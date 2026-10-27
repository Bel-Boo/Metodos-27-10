"""
Desafío de Integración Numérica · La Paz, Bolivia
Trapecio compuesto, Simpson 1/3 y Simpson 3/8 (implementados desde cero).

Ejecutar localmente:  streamlit run app.py
"""
import ast
import operator
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# 1. EVALUADOR SEGURO DE EXPRESIONES (sin eval)
# =====================================================================
_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "ln": np.log, "log": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs,
}
_CONSTS = {"pi": np.pi, "e": np.e}
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ExpresionError(ValueError):
    """Error al interpretar una expresión matemática."""


class DatosError(ValueError):
    """Error de validación de datos o de parámetros."""


def evaluar(expr, x=None):
    """Evalúa una expresión matemática de forma segura. Variable: x (o t)."""
    expr = str(expr).strip().replace("^", "**").replace("π", "pi").replace(",", ".")
    if not expr:
        raise ExpresionError("La expresión está vacía.")
    if len(expr) > 300:
        raise ExpresionError("La expresión es demasiado larga.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ExpresionError(f"Sintaxis inválida en la expresión: «{expr}».")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return np.float64(node.value)
            raise ExpresionError("Solo se permiten números.")
        if isinstance(node, ast.Name):
            if node.id in ("x", "t"):
                if x is None:
                    raise ExpresionError("Este campo no puede depender de x.")
                return x
            if node.id in _CONSTS:
                return np.float64(_CONSTS[node.id])
            raise ExpresionError(f"Nombre no permitido: «{node.id}».")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and len(node.args) == 1 and not node.keywords):
            return _FUNCS[node.func.id](ev(node.args[0]))
        raise ExpresionError(
            "Operación o función no permitida. Usa + - * / ** ^ y: "
            + ", ".join(_FUNCS) + ", pi, e."
        )

    try:
        with np.errstate(all="ignore"):
            return ev(tree)
    except (OverflowError, ZeroDivisionError):
        raise ExpresionError("Desbordamiento o división por cero al evaluar.")


def evaluar_escalar(expr, nombre="valor"):
    try:
        v = float(evaluar(expr))
    except ExpresionError as err:
        raise ExpresionError(f"{nombre}: {err}")
    except (TypeError, ValueError):
        raise ExpresionError(f"{nombre}: no es un número válido.")
    if not np.isfinite(v):
        raise ExpresionError(f"{nombre}: el resultado no es finito.")
    return v


def evaluar_funcion(expr, xs):
    val = evaluar(expr, xs)
    val = np.broadcast_to(np.asarray(val, dtype=float), xs.shape).copy()
    if not np.all(np.isfinite(val)):
        malos = xs[~np.isfinite(val)]
        raise DatosError(
            f"f(x) no es real/finita en {len(malos)} nodo(s), p. ej. x = {malos[0]:g}."
        )
    return val


# =====================================================================
# 2. VALIDACIÓN Y LECTURA DE DATOS
# =====================================================================
def parse_lista(texto, nombre):
    tokens = [t for t in re.split(r"[;,\s]+", str(texto).strip()) if t]
    vals = []
    for i, t in enumerate(tokens, 1):
        try:
            vals.append(float(t))
        except ValueError:
            raise DatosError(f"{nombre}: el valor #{i} («{t}») no es numérico.")
    return vals


def validar_tabla(x, y):
    """Valida pares (x, y). Devuelve (x, y, n, h)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise DatosError(
            f"Las listas tienen distinta longitud: {len(x)} valores de x y {len(y)} de y."
        )
    if len(x) < 2:
        raise DatosError("Se necesitan al menos 2 puntos (n ≥ 1).")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise DatosError("Hay valores vacíos, NaN o infinitos en los datos.")
    d = np.diff(x)
    if np.any(d <= 0):
        i = int(np.argmax(d <= 0)) + 1
        raise DatosError(
            f"x debe ser estrictamente creciente (falla en la posición {i + 1}: "
            f"x[{i}] = {x[i - 1]:g} → x[{i + 1}] = {x[i]:g})."
        )
    h = (x[-1] - x[0]) / (len(x) - 1)
    if np.max(np.abs(d - h)) > 1e-6 * abs(h):
        raise DatosError(
            "Espaciamiento no uniforme: los métodos requieren nodos igualmente "
            f"espaciados. h promedio = {h:g}, pero hay diferencias entre "
            f"{d.min():g} y {d.max():g}."
        )
    return x, y, len(x) - 1, h


def tabla_a_arrays(df):
    """Convierte la tabla editable en arrays; ignora filas totalmente vacías."""
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    crudo_x, crudo_y = df["x"], df["y"]
    vacias = crudo_x.isna() & crudo_y.isna()
    x, y = x[~vacias], y[~vacias]
    crudo_x, crudo_y = crudo_x[~vacias], crudo_y[~vacias]
    if (x.isna() & crudo_x.notna()).any() or (y.isna() & crudo_y.notna()).any():
        raise DatosError("Hay datos no numéricos en la tabla.")
    if x.isna().any() or y.isna().any():
        raise DatosError("Hay filas incompletas en la tabla (falta x o y).")
    return x.to_numpy(float), y.to_numpy(float)


# =====================================================================
# 3. MÉTODOS DE INTEGRACIÓN (implementación propia)
# =====================================================================
M_TRAP = "Trapecio compuesto"
M_S13 = "Simpson 1/3"
M_S38 = "Simpson 3/8"
M_COMB = "Simpson combinado (1/3 + 3/8)"
M_S13T = "Simpson 1/3 + trapecio (n impar)"
METODOS = [M_TRAP, M_S13, M_S38, M_COMB, M_S13T]


def requisito(metodo, n):
    """Devuelve None si el método es aplicable con n, o un mensaje de error."""
    if metodo == M_TRAP:
        return None if n >= 1 else "Requiere n ≥ 1."
    if metodo == M_S13:
        return None if (n >= 2 and n % 2 == 0) else f"Requiere n par (n = {n})."
    if metodo == M_S38:
        return None if (n >= 3 and n % 3 == 0) else f"Requiere n múltiplo de 3 (n = {n})."
    if metodo in (M_COMB, M_S13T):
        return None if (n >= 3 and n % 2 == 1) else f"Requiere n impar ≥ 3 (n = {n})."
    return "Método desconocido."


def pesos(metodo, n, h):
    """Vector de pesos w_i tal que I ≈ Σ w_i f_i. Devuelve (w, escala)."""
    err = requisito(metodo, n)
    if err:
        raise DatosError(f"{metodo}: {err}")
    w = np.zeros(n + 1)

    def trap(i0, i1):
        for i in range(i0, i1):
            w[i] += h / 2
            w[i + 1] += h / 2

    def s13(i0, i1):
        for k in range(i0, i1, 2):
            w[k] += h / 3
            w[k + 1] += 4 * h / 3
            w[k + 2] += h / 3

    def s38(i0, i1):
        for k in range(i0, i1, 3):
            w[k] += 3 * h / 8
            w[k + 1] += 9 * h / 8
            w[k + 2] += 9 * h / 8
            w[k + 3] += 3 * h / 8

    if metodo == M_TRAP:
        trap(0, n); escala = h / 2
    elif metodo == M_S13:
        s13(0, n); escala = h / 3
    elif metodo == M_S38:
        s38(0, n); escala = 3 * h / 8
    elif metodo == M_COMB:
        s13(0, n - 3); s38(n - 3, n); escala = h
    else:  # M_S13T
        s13(0, n - 1); trap(n - 1, n); escala = h
    return w, escala


def integrar(metodo, y, n, h):
    w, _ = pesos(metodo, n, h)
    return float(np.sum(w * y))


# =====================================================================
# 4. TRANSFORMACIONES PARA DATOS TABULADOS (p. ej. ejercicio 27)
# =====================================================================
T_NONE = "Ninguna: integrar y dx"
T_VOL = "Volumen de revolución: π·r² con r = y/2 (y = diámetro)"
T_SUP_S = "Área lateral simplificada: 2π·r con r = y/2 (omite r′)"
T_SUP_R = "Área lateral real: 2π·r·√(1+r′²), r′ por diferencias finitas"
TRANSFORMS = [T_NONE, T_VOL, T_SUP_S, T_SUP_R]


def transformar(x, y, modo):
    if modo == T_VOL:
        return np.pi * (y / 2) ** 2
    if modo == T_SUP_S:
        return 2 * np.pi * (y / 2)
    if modo == T_SUP_R:
        r = y / 2
        rp = np.gradient(r, x)
        return 2 * np.pi * r * np.sqrt(1 + rp ** 2)
    return y


# =====================================================================
# 5. TABLAS DE RESULTADOS Y GRÁFICOS
# =====================================================================
def fmt6(v):
    return f"{v:.6f}"


def calcular_tabla(metodos, y, n, h, factor, ref, extra_f, extra_u):
    filas, avisos = [], []
    for m in metodos:
        err = requisito(m, n)
        if err:
            avisos.append(f"**{m}** no aplicable: {err}")
            continue
        val = integrar(m, y, n, h) * factor
        fila = {"Método": m, "Resultado": fmt6(val), "Notación científica": f"{val:.8e}"}
        if extra_f is not None and extra_u:
            fila[f"Resultado en {extra_u}"] = fmt6(val * extra_f)
        if ref is not None:
            ea = abs(ref - val)
            fila["Error absoluto"] = f"{ea:.6e}"
            fila["Error %"] = fmt6(100 * ea / abs(ref)) if ref != 0 else "—"
        filas.append(fila)
    return pd.DataFrame(filas), avisos


def graficar(x_nodos, f_nodos, xlabel, ylabel, curva=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    if curva is not None:
        ax.plot(curva[0], curva[1], color="#1f77b4", lw=2, label="f(x) exacta")
    ax.fill_between(x_nodos, f_nodos, alpha=0.25, color="#ff7f0e",
                    label="Área bajo la poligonal (trapecio)")
    ax.plot(x_nodos, f_nodos, "o-", color="#d62728", ms=4, lw=1, label="Nodos")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# 6. EJERCICIOS PRECARGADOS
# =====================================================================
TAB = "Datos tabulados (x, y)"
FUN = "Función f(x)"
PERS = "Personalizado (ejemplo editable)"

BASE = dict(
    modo=TAB, entrada="Tabla editable", x=[], y=[], func="x**2", a="0", b="2",
    n=4, n_comp="4, 8", ref="", factor="1", unit="", extra_f="", extra_u="",
    xlabel="x", ylabel="y", transform=T_NONE, nota="",
)

PRESETS = {
    PERS: dict(x=[0, 0.5, 1, 1.5, 2], y=[0, 0.25, 1, 2.25, 4], ref="8/3", unit="u²",
               nota="Ejemplo f(x)=x² en [0, 2]. Edita la tabla o cambia a «Función f(x)»."),
    "Prueba: f(x)=x² en [0, 2]": dict(
        modo=FUN, func="x**2", a="0", b="2", n=4, n_comp="4, 6, 8", ref="8/3", unit="u²",
        nota="Verificación mínima exigida: la integral exacta es 8/3."),
    "Ej. 19 · Distancia de un vehículo": dict(
        x=list(range(8)), y=[0, 14, 39, 69, 95, 114, 129, 139],
        xlabel="t (s)", ylabel="v (mi/h)", factor="1/3600", unit="mi",
        extra_f="1609.344", extra_u="m", ref="",
        nota="d = ∫v dt. v está en mi/h y t en s, por eso el factor 1/3600. "
             "Con n = 7 usa «Simpson combinado» (1/3 en 4 subintervalos + 3/8 en 3)."),
    "Ej. 20 · Trabajo de una banda elástica": dict(
        x=[0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8], y=[0, 0.85, 1.30, 1.60, 1.87, 2.14, 2.34, 2.52],
        xlabel="x (in)", ylabel="F (lb)", unit="lb·in", extra_f="0.112984829", extra_u="J",
        nota="W = ∫F dx. n = 7: usa «Simpson combinado»."),
    "Ej. 21 · Área de un perfil topográfico": dict(
        x=list(range(1, 11)), y=[1.3, 3.7, 4.6, 4.4, 2.7, 2.5, 3.6, 5.3, 6.0, 5.9],
        xlabel="x", ylabel="y", unit="u²",
        nota="Datos ilustrativos digitalizados (no exactos). n = 9: el solucionario usa "
             "«Simpson 1/3 + trapecio»."),
    "Ej. 22 · Longitud de una catenaria": dict(
        modo=FUN, func="cosh(x/24)", a="-12", b="26", n=8, n_comp="4, 8, 16",
        ref="24*(sinh(26/24)-sinh(-12/24))", unit="ft",
        nota="y = 24·cosh(x/24). Como 1+(y′)² = cosh², el integrando de la longitud es cosh(x/24)."),
    "Ej. 23 · Caudal de una tubería": dict(
        modo=FUN, func="160*pi*x*(1-x/0.25)**(1/7)", a="0", b="0.25", n=8, n_comp="4, 8, 16",
        ref="49*pi/12", unit="in³/s",
        nota="Q = ∫2πr·v(r)dr con v = 80(1−r/R)^(1/7), R = 0.25 in. Usa x en lugar de r."),
    "Ej. 24 · Longitud de la cardioide": dict(
        modo=FUN, func="20*sin(x/2)", a="0", b="2*pi", n=8, n_comp="4, 8, 16",
        ref="80", unit="u",
        nota="Con b = 5: √(x′²+y′²) = 20·sin(t/2) en [0, 2π]. Usa x en lugar de t."),
    "Ej. 25 · Energía potencial de un satélite": dict(
        modo=FUN, func="500*9.81*6371000**2/(6371000+x)**2", a="0", b="800000", n=8,
        n_comp="4, 8, 16", ref="500*9.81*6371000*800000/(6371000+800000)", unit="J",
        nota="ΔU = m·g₀ ∫R²/(R+y)² dy, con m=500 kg, g₀=9.81 m/s², R=6 371 000 m, h=800 000 m."),
    "Ej. 26 · Superficie de un lago": dict(
        x=list(range(40, 261, 20)), y=[None] * 12, xlabel="x (mi)", ylabel="w(x) (mi)",
        unit="mi²", ref="9940",
        nota="Completa la columna w(x) con las anchuras medidas del mapa (cada 20 mi). "
             "n = 11: usa «Simpson combinado». Referencia del problema original: 9 940 mi²."),
    "Ej. 27 · Volumen de un objeto alargado": dict(
        x=list(range(13)), y=[0, 2.6, 3.2, 4.8, 5.6, 6.0, 6.2, 6.0, 5.6, 4.8, 3.3, 2.6, 0],
        xlabel="z (in)", ylabel="d (in)", unit="in³", transform=T_VOL,
        nota="y = diámetro. V = π∫r² dz. Cambia la transformación para ver el área simplificada "
             "(unidad in²) y el área lateral real."),
}


def _fmt_lista(vals):
    return ", ".join("" if v is None else f"{v:g}" for v in vals if v is not None)


def cargar_preset():
    p = {**BASE, **PRESETS[st.session_state.preset]}
    ss = st.session_state
    ss.df_base = pd.DataFrame({"x": pd.Series(p["x"], dtype=float),
                               "y": pd.Series(p["y"], dtype=float)})
    ss.ver = ss.get("ver", 0) + 1
    ss.xs_text = _fmt_lista(p["x"])
    ss.ys_text = _fmt_lista(p["y"])
    for k_state, k_p in [("modo", "modo"), ("entrada", "entrada"), ("func", "func"),
                         ("a_txt", "a"), ("b_txt", "b"), ("n_main", "n"), ("n_comp", "n_comp"),
                         ("ref_txt", "ref"), ("factor_txt", "factor"), ("unit", "unit"),
                         ("extra_f", "extra_f"), ("extra_u", "extra_u"), ("xlabel", "xlabel"),
                         ("ylabel", "ylabel"), ("transform", "transform"), ("nota", "nota")]:
        ss[k_state] = p[k_p]
    ss.metodos = list(METODOS)


# =====================================================================
# 7. INTERFAZ
# =====================================================================
st.set_page_config(page_title="Integración numérica · La Paz", page_icon="📐", layout="wide")

if "preset" not in st.session_state:
    st.session_state.preset = "Ej. 19 · Distancia de un vehículo"
    cargar_preset()

# Evita que Streamlit borre valores de widgets que no se muestran en esta ejecución
for _k in ("xs_text", "ys_text", "func", "a_txt", "b_txt", "n_main", "n_comp",
           "entrada", "transform", "m_peso"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

st.title("📐 Integración numérica aplicada a La Paz, Bolivia")
st.caption("Trapecio compuesto · Simpson 1/3 · Simpson 3/8 — Métodos Numéricos")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.selectbox("Cargar ejercicio", list(PRESETS), key="preset", on_change=cargar_preset)
    st.radio("Tipo de datos", [TAB, FUN], key="modo")
    st.multiselect("Métodos a calcular", METODOS, key="metodos")
    st.markdown("---")
    st.caption("Usa **punto** como separador decimal. En funciones: `x`, `pi`, `e`, "
               "`sin cos tan sinh cosh exp ln sqrt abs`, potencias con `**` o `^`.")

tab_calc, tab_val, tab_ayuda = st.tabs(["🧮 Calculadora", "✅ Prueba de validación", "📖 Fórmulas"])

# ------------------------------- CALCULADORA -------------------------------
with tab_calc:
    if st.session_state.nota:
        st.info(st.session_state.nota)

    with st.expander("🏷️ Unidades, conversión y valor de referencia", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.text_input("Etiqueta de x", key="xlabel")
        c2.text_input("Etiqueta de y / f(x)", key="ylabel")
        c3.text_input("Factor de conversión del resultado", key="factor_txt",
                      help="Se multiplica al resultado. Admite expresiones, p. ej. 1/3600.")
        c4.text_input("Unidad del resultado", key="unit")
        c5, c6, c7 = st.columns(3)
        c5.text_input("Conversión adicional (factor)", key="extra_f",
                      help="Opcional: muestra el resultado también en otra unidad.")
        c6.text_input("Unidad adicional", key="extra_u")
        c7.text_input("Valor de referencia / exacto (opcional)", key="ref_txt",
                      help="En unidades del resultado. Admite expresiones, p. ej. 49*pi/12.")

    try:
        factor = evaluar_escalar(st.session_state.factor_txt or "1", "Factor de conversión")
        extra_f = (evaluar_escalar(st.session_state.extra_f, "Conversión adicional")
                   if st.session_state.extra_f.strip() else None)
        ref = (evaluar_escalar(st.session_state.ref_txt, "Valor de referencia")
               if st.session_state.ref_txt.strip() else None)
    except ExpresionError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    unit = st.session_state.unit
    xlabel, ylabel = st.session_state.xlabel, st.session_state.ylabel
    metodos_sel = st.session_state.metodos
    if not metodos_sel:
        st.warning("Selecciona al menos un método en la barra lateral.")
        st.stop()

    # ---------------------------- Datos tabulados ----------------------------
    if st.session_state.modo == TAB:
        st.subheader("1. Datos")
        st.radio("Forma de ingreso", ["Tabla editable", "Texto separado por comas"],
                 key="entrada", horizontal=True)
        try:
            if st.session_state.entrada == "Tabla editable":
                df_ed = st.data_editor(
                    st.session_state.df_base, num_rows="dynamic", width="stretch",
                    key=f"editor_{st.session_state.ver}",
                    column_config={"x": st.column_config.NumberColumn(xlabel, format="%g"),
                                   "y": st.column_config.NumberColumn(ylabel, format="%g")},
                )
                x_raw, y_raw = tabla_a_arrays(df_ed)
            else:
                t1, t2 = st.columns(2)
                t1.text_area(f"Valores de x — {xlabel}", key="xs_text", height=110)
                t2.text_area(f"Valores de y — {ylabel}", key="ys_text", height=110)
                x_raw = parse_lista(st.session_state.xs_text, "x")
                y_raw = parse_lista(st.session_state.ys_text, "y")

            x, y, n, h = validar_tabla(x_raw, y_raw)
        except (DatosError, ExpresionError) as e:
            st.error(f"⚠️ {e}")
            st.stop()

        st.selectbox("Transformación del integrando", TRANSFORMS, key="transform")
        f_nodos = transformar(x, y, st.session_state.transform)
        curva = None
        a, b = float(x[0]), float(x[-1])

    # ------------------------------- Función --------------------------------
    else:
        st.subheader("1. Función e intervalo")
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.text_input("f(x)", key="func", help="Ejemplo: 160*pi*x*(1-x/0.25)**(1/7)")
        c2.text_input("a (límite inferior)", key="a_txt")
        c3.text_input("b (límite superior)", key="b_txt")
        c4, c5 = st.columns(2)
        c4.number_input("n (subintervalos)", min_value=1, step=1, key="n_main")
        c5.text_input("Comparar con otros n (separados por comas)", key="n_comp")
        usar_num = st.checkbox("Si no hay referencia, usar Simpson 1/3 con n = 20 000 como referencia numérica",
                               value=False)
        try:
            a = evaluar_escalar(st.session_state.a_txt, "a")
            b = evaluar_escalar(st.session_state.b_txt, "b")
            if a >= b:
                raise DatosError("Se requiere a < b.")
            n = int(st.session_state.n_main)
            h = (b - a) / n
            x = np.linspace(a, b, n + 1)
            y = f_nodos = evaluar_funcion(st.session_state.func, x)
            xx = np.linspace(a, b, 500)
            curva = (xx, evaluar_funcion(st.session_state.func, xx))
            try:
                n_extra = sorted({int(v) for v in parse_lista(st.session_state.n_comp, "n")})
            except ValueError:
                raise DatosError("Los valores de n para comparar deben ser enteros.")
            if any(v < 1 for v in n_extra):
                raise DatosError("Todos los n deben ser ≥ 1.")
            if ref is None and usar_num:
                N = 20000
                xr = np.linspace(a, b, N + 1)
                ref = integrar(M_S13, evaluar_funcion(st.session_state.func, xr), N, (b - a) / N) * factor
                st.caption(f"Referencia numérica usada: {ref:.8f} {unit}")
        except (DatosError, ExpresionError) as e:
            st.error(f"⚠️ {e}")
            st.stop()

    # ------------------------------ Resultados -------------------------------
    st.subheader("2. Parámetros")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("a", f"{a:g}")
    m2.metric("b", f"{b:g}")
    m3.metric("n (subintervalos)", n)
    m4.metric("h = (b − a)/n", f"{h:.6g}")
    if st.session_state.modo == TAB:
        st.caption(f"✔️ Datos válidos: {n + 1} puntos con espaciamiento uniforme.")

    st.subheader(f"3. Resultados con n = {n}" + (f" ({unit})" if unit else ""))
    try:
        df_res, avisos = calcular_tabla(metodos_sel, f_nodos, n, h, factor, ref, extra_f,
                                        st.session_state.extra_u.strip())
    except DatosError as e:
        st.error(f"⚠️ {e}")
        st.stop()
    for av in avisos:
        st.warning(av)
    if not df_res.empty:
        st.dataframe(df_res, width="stretch", hide_index=True)
        if ref is None:
            st.caption("Sin valor de referencia no se puede calcular el error verdadero.")
        else:
            st.caption(f"Error absoluto = |I_ref − I_aprox| · Error % = 100·Eₐ/|I_ref| · "
                       f"I_ref = {ref:.8g} {unit}")
        st.download_button("⬇️ Descargar resultados (CSV)",
                           df_res.to_csv(index=False).encode("utf-8"),
                           file_name="resultados_integracion.csv", mime="text/csv")
    else:
        st.error("Ningún método seleccionado es aplicable con este n.")

    # Comparación de n (solo función)
    if st.session_state.modo == FUN and n_extra:
        st.subheader("4. Comparación para distintos n")
        filas = []
        try:
            for nn in sorted({n, *n_extra}):
                hh = (b - a) / nn
                yy = evaluar_funcion(st.session_state.func, np.linspace(a, b, nn + 1))
                for m in metodos_sel:
                    if requisito(m, nn):
                        continue
                    v = integrar(m, yy, nn, hh) * factor
                    f_ = {"n": nn, "h": f"{hh:.6g}", "Método": m, "Resultado": fmt6(v)}
                    if ref is not None:
                        ea = abs(ref - v)
                        f_["Error absoluto"] = f"{ea:.6e}"
                        f_["Error %"] = fmt6(100 * ea / abs(ref)) if ref != 0 else "—"
                    filas.append(f_)
        except DatosError as e:
            st.error(f"⚠️ {e}")
        if filas:
            st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
            st.caption("Los métodos de Simpson solo aparecen cuando n cumple su condición.")

    # Gráfico y pesos
    st.subheader("5. Gráfico" if st.session_state.modo == TAB else "5. Gráfico")
    st.caption("La región sombreada es el área bajo la poligonal que une los nodos.")
    fig = graficar(x, f_nodos, xlabel if st.session_state.modo == TAB else "x",
                   ylabel if st.session_state.modo == TAB else "f(x)", curva)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("6. Tabla de nodos y pesos de integración")
    aplicables = [m for m in metodos_sel if requisito(m, n) is None]
    if aplicables:
        m_peso = st.selectbox("Método", aplicables, key="m_peso")
        w, escala = pesos(m_peso, n, h)
        df_w = pd.DataFrame({
            "i": np.arange(n + 1), "xᵢ": x, "f(xᵢ)": f_nodos,
            "Coeficiente cᵢ": w / escala, "Peso wᵢ": w, "wᵢ·f(xᵢ)": w * f_nodos,
        })
        st.dataframe(df_w.style.format({"xᵢ": "{:.6f}", "f(xᵢ)": "{:.6f}", "Coeficiente cᵢ": "{:.4g}",
                                        "Peso wᵢ": "{:.6f}", "wᵢ·f(xᵢ)": "{:.6f}"}),
                     width="stretch", hide_index=True)
        if m_peso in (M_COMB, M_S13T):
            st.caption("I ≈ h·Σ cᵢ fᵢ  (los coeficientes mezclan los factores de cada tramo).")
        else:
            nombre_esc = {M_TRAP: "h/2", M_S13: "h/3", M_S38: "3h/8"}[m_peso]
            st.caption(f"I ≈ ({nombre_esc})·Σ cᵢ fᵢ = {escala:.6f}·Σ cᵢ fᵢ  ·  "
                       f"Σ wᵢ fᵢ = {np.sum(w * f_nodos):.6f}")

# ------------------------------ VALIDACIÓN ------------------------------
with tab_val:
    st.subheader("Prueba obligatoria: ∫₀² x² dx = 8/3")
    exacto = 8 / 3
    filas = []
    for nn in (4, 6, 8, 12):
        xv = np.linspace(0, 2, nn + 1)
        yv = xv ** 2
        hv = 2 / nn
        for m in (M_TRAP, M_S13, M_S38):
            if requisito(m, nn):
                continue
            v = integrar(m, yv, nn, hv)
            filas.append({"n": nn, "h": f"{hv:.6g}", "Método": m, "Resultado": fmt6(v),
                          "Error absoluto": f"{abs(exacto - v):.3e}",
                          "Error %": fmt6(100 * abs(exacto - v) / exacto)})
    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
    ok = all(abs(exacto - integrar(M_S13, np.linspace(0, 2, k + 1) ** 2, k, 2 / k)) < 1e-12
             for k in (4, 8)) and abs(exacto - integrar(M_S38, np.linspace(0, 2, 7) ** 2, 6, 2 / 6)) < 1e-12
    if ok:
        st.success("✅ Simpson integra x² de forma exacta (hasta redondeo) y el trapecio converge "
                   "con error O(h²), como se espera.")
    else:
        st.error("❌ La verificación falló. Revisa la implementación.")
    st.caption("El trapecio tiene error positivo (sobreestima) porque x² es convexa; "
               "Simpson es exacto para polinomios de grado ≤ 3.")

# -------------------------------- FÓRMULAS --------------------------------
with tab_ayuda:
    st.subheader("Formulario")
    st.latex(r"h=\frac{b-a}{n},\qquad x_i=a+ih")
    st.markdown("**Trapecio compuesto**")
    st.latex(r"T=h\left[\frac{f_0+f_n}{2}+\sum_{i=1}^{n-1}f_i\right]")
    st.markdown("**Simpson 1/3** (n par)")
    st.latex(r"S=\frac{h}{3}\left[f_0+f_n+4\sum_{i\ \mathrm{impar}}f_i+2\sum_{i\ \mathrm{par\ interior}}f_i\right]")
    st.markdown("**Simpson 3/8** (n múltiplo de 3)")
    st.latex(r"S=\frac{3h}{8}\left[f_0+f_n+3\sum_{i\not\equiv0\ (3)}f_i+2\sum_{i\equiv0\ (3)\ \mathrm{int.}}f_i\right]")
    st.markdown("**Errores** (si se conoce un valor de referencia)")
    st.latex(r"E_a=|I_{ref}-I_{aprox}|,\qquad E\%=100\,\frac{E_a}{|I_{ref}|}")
    st.markdown(
        "- **n impar ≥ 3**: «Simpson combinado» aplica 1/3 en los primeros n−3 subintervalos y 3/8 en los últimos 3 "
        "(con n = 7: 4 + 3).\n"
        "- **«Simpson 1/3 + trapecio»** aplica 1/3 en los primeros n−1 y trapecio en el último "
        "(es lo que usa el solucionario del ejercicio 21).\n"
        "- Se valida que x sea creciente, el espaciamiento uniforme, que no haya datos vacíos o no "
        "numéricos y que n cumpla la condición de cada método."
    )
