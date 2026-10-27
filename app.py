# -*- coding: utf-8 -*-
"""Desafío de Integración Numérica · La Paz, Bolivia  (Streamlit)
Ejecutar:  streamlit run app.py
"""
import ast
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Integración numérica · La Paz", page_icon="📐", layout="wide")

# ----------------------------------------------------------------------------
# 1) NÚCLEO NUMÉRICO
# ----------------------------------------------------------------------------
METODOS = ["Trapecio compuesto", "Simpson 1/3", "Simpson 3/8",
           "Simpson combinado (1/3 + 3/8)", "Simpson 1/3 + trapecio final"]


def ahora():
    """Fecha y hora de La Paz (UTC-4)."""
    return datetime.now(timezone(timedelta(hours=-4))).strftime("%d/%m/%Y %H:%M:%S")


def fmt(v):
    if v is None:
        return ""
    return f"{v:.8e}" if (abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-4)) else f"{v:.6f}"


def check_uniform(x):
    """Valida x y devuelve h."""
    x = np.asarray(x, float)
    if len(x) < 3:
        raise ValueError("Se necesitan al menos 3 puntos (n ≥ 2 subintervalos).")
    if not np.all(np.isfinite(x)):
        raise ValueError("Hay datos no numéricos o vacíos en x.")
    d = np.diff(x)
    if np.any(d <= 0):
        raise ValueError("Los valores de x deben ser estrictamente crecientes.")
    if not np.allclose(d, d[0], rtol=1e-6, atol=0):
        raise ValueError("El espaciamiento no es uniforme: todos los pasos h deben ser iguales.")
    return float(d[0])


def seg13(m):  # coeficientes 1,4,2,4,...,4,1  (m par)
    c = np.ones(m + 1); c[1:-1:2] = 4; c[2:-1:2] = 2
    return c


def seg38(m):  # coeficientes 1,3,3,2,3,3,...,1  (m múltiplo de 3)
    c = np.full(m + 1, 3.0); c[0] = c[-1] = 1; c[3:-1:3] = 2
    return c


def pesos(m, n, h):
    """Devuelve (pesos w_i, coeficientes, factor). Integral = Σ w_i·f_i."""
    if m == METODOS[0]:
        c = np.full(n + 1, 2.0); c[[0, n]] = 1
        return c * h / 2, c, "h/2"
    if m == METODOS[1]:
        if n % 2:
            raise ValueError(f"n={n} es impar; Simpson 1/3 necesita n par")
        return seg13(n) * h / 3, seg13(n), "h/3"
    if m == METODOS[2]:
        if n % 3:
            raise ValueError(f"n={n} no es múltiplo de 3")
        return seg38(n) * 3 * h / 8, seg38(n), "3h/8"
    if n % 2 == 0:
        raise ValueError("n es par: se usa Simpson 1/3")
    w = np.zeros(n + 1)
    if m == METODOS[3]:
        k = n - 3                      # 1/3 en los primeros k, 3/8 en los últimos 3
        if k > 0:
            w[:k + 1] += seg13(k) * h / 3
        w[k:] += seg38(3) * 3 * h / 8
    else:
        k = n - 1                      # 1/3 en los primeros n-1, trapecio en el último
        if k > 0:
            w[:k + 1] += seg13(k) * h / 3
        w[k:] += h / 2
    return w, None, "mixto"


def resolver(x, y, h):
    n = len(x) - 1
    out = {}
    for m in METODOS:
        try:
            w, c, fac = pesos(m, n, h)
            out[m] = (float(np.sum(w * y)), w, c, fac)
        except ValueError as e:
            out[m] = (None, str(e))
    return out


def mejor(vals, pref):
    if METODOS[pref] in vals:
        return METODOS[pref]
    for m in METODOS[1:]:
        if m in vals:
            return m
    return METODOS[0]


def tabla_res(p, res):
    """Tabla de resultados de un problema. Devuelve (DataFrame, {método: valor})."""
    ref, hora, filas, vals = p.get("exacto"), ahora(), [], {}
    for m in METODOS:
        r = res[m]
        if r[0] is None:
            filas.append({"Método": m, "Resultado": "No aplica: " + r[1], "Unidad": "", "Referencia": "",
                          "Error absoluto": "", "Error %": "", "Hora del cálculo": hora})
            continue
        v = p["fin"](r[0]); vals[m] = v
        f = {"Método": m, "Resultado": fmt(v), "Unidad": p["unidad"],
             "Referencia": fmt(ref) if ref is not None else "—", "Error absoluto": "—", "Error %": "—",
             "Hora del cálculo": hora}
        if ref:
            ea = abs(ref - v)
            f["Error absoluto"], f["Error %"] = fmt(ea), f"{100 * ea / abs(ref):.6f} %"
        filas.append(f)
    return pd.DataFrame(filas), vals


def evaluar(expr, vals):
    """Evalúa f(x) escrita por el usuario de forma segura."""
    fun = {"sin": np.sin, "sen": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "ln": np.log,
           "log": np.log, "sqrt": np.sqrt, "raiz": np.sqrt, "abs": np.abs, "sinh": np.sinh,
           "cosh": np.cosh, "tanh": np.tanh, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan}
    cte = {"pi": np.pi, "e": np.e}
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}

    def ev(nd):
        if isinstance(nd, ast.Expression):
            return ev(nd.body)
        if isinstance(nd, ast.Constant) and isinstance(nd.value, (int, float)):
            return np.float64(nd.value)
        if isinstance(nd, ast.Name):
            if nd.id == "x":
                return vals
            if nd.id in cte:
                return np.float64(cte[nd.id])
        if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, (ast.USub, ast.UAdd)):
            v = ev(nd.operand)
            return -v if isinstance(nd.op, ast.USub) else v
        if isinstance(nd, ast.BinOp) and type(nd.op) in ops:
            return ops[type(nd.op)](ev(nd.left), ev(nd.right))
        if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name) and nd.func.id in fun \
                and len(nd.args) == 1 and not nd.keywords:
            return fun[nd.func.id](ev(nd.args[0]))
        raise ValueError("Expresión no permitida. Usa x, números, + - * / ^, pi, e y funciones "
                         "como sen, cos, exp, sqrt, ln.")

    try:
        arbol = ast.parse(expr.replace("^", "**").replace(",", ".").strip(), mode="eval")
    except SyntaxError:
        raise ValueError("No pude entender la función. Ejemplo válido: x^2 + 3*sen(x)")
    with np.errstate(all="ignore"):
        r = ev(arbol)
    return np.broadcast_to(np.asarray(r, float), np.shape(vals)).copy()


# ----------------------------------------------------------------------------
# 2) PRESENTACIÓN DE RESULTADOS (se reutiliza en todas las páginas)
# ----------------------------------------------------------------------------
def mostrar(p):
    x, y = np.asarray(p["x"], float), np.asarray(p["y"], float)
    h = check_uniform(x)
    if not np.all(np.isfinite(y)):
        raise ValueError("Hay datos vacíos, no numéricos o infinitos en y.")
    n = len(x) - 1
    res = resolver(x, y, h)
    c1, c2, c3 = st.columns(3)
    c1.metric("Subintervalos n", n); c2.metric("Paso h", f"{h:g}"); c3.metric("Intervalo", f"[{x[0]:g}, {x[-1]:g}]")
    df, vals = tabla_res(p, res)
    st.markdown(f"##### 📊 {p['nombre']}")
    st.dataframe(df, hide_index=True)
    m = mejor(vals, p.get("pref", 1))
    if m in vals:
        txt = f"**Resultado ({m}): {fmt(vals[m])} {p['unidad']}**"
        if p.get("nota"):
            txt += f"  ·  {p['nota'](vals[m])}"
        st.success(txt)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, "o-", color="#1f77b4"); ax.fill_between(x, y, alpha=0.25)
    ax.set_xlabel(p.get("xl", "x")); ax.set_ylabel(p.get("yl", "f(x)")); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close(fig)
    ok = [k for k in METODOS if res[k][0] is not None]
    with st.expander("🔍 Ver tabla de pesos (cómo se armó la suma)"):
        mm = st.selectbox("Método", ok, key=f"w_{p['nombre']}_{id(p) % 9973}")
        _, w, c, fac = res[mm][0], res[mm][1], res[mm][2], res[mm][3]
        t = {"i": range(n + 1), "x_i": x, "f(x_i)": y}
        if c is not None:
            t["Coeficiente"] = c
        t["Peso w_i"] = w; t["Aporte w_i·f_i"] = w * y
        st.dataframe(pd.DataFrame(t), hide_index=True)
        st.caption(f"Factor del método: {fac}.  La integral es la suma de la última columna = "
                   f"{fmt(float(np.sum(w * y)))}")
    return res, vals


# ----------------------------------------------------------------------------
# 3) LOS EJERCICIOS 19–27
# ----------------------------------------------------------------------------
def fn(nombre, f, a, b, n, unidad, exacto, xl="x", yl="f(x)"):
    x = np.linspace(a, b, int(n) + 1)
    return dict(nombre=nombre, x=x, y=f(x), fin=lambda I: I, unidad=unidad, exacto=exacto, xl=xl, yl=yl)


def tb(nombre, xy, unidad, fin=lambda I: I, **kw):
    return dict(nombre=nombre, x=xy[0], y=xy[1], fin=fin, unidad=unidad, exacto=None, **kw)


EJ = {
    19: dict(
        t="Transporte urbano · distancia de un vehículo", tipo="tabla", pref=3,
        cols=["t (s)", "v (mi/h)"],
        df=[[0, 0], [1, 14], [2, 39], [3, 69], [4, 95], [5, 114], [6, 129], [7, 139]],
        que="Un vehículo experimental anota su velocidad cada segundo durante 7 s. **La distancia es el área bajo la "
            "curva v(t)**. Como la velocidad está en mi/h y el tiempo en segundos, el resultado se divide entre 3600.",
        modelo=r"d=\frac{1}{3600}\int_0^7 v(t)\,dt",
        ing="**Telemetría y flotas (GPS/IoT):** los dispositivos envían velocidad cada cierto tiempo y el sistema "
            "calcula la distancia integrando esos datos. Es la base de apps de transporte, logística y monitoreo.",
        prob=lambda xy, p, n: [tb("Distancia recorrida", xy, "mi", lambda I: I / 3600, xl="t (s)", yl="v (mi/h)",
                                  nota=lambda v: f"≈ {v * 1609.344:.2f} m")]),
    20: dict(
        t="Ensayo de elasticidad · trabajo de la banda", tipo="tabla", pref=3,
        cols=["x (in)", "F (lb)"],
        df=[[0, 0], [0.4, 0.85], [0.8, 1.30], [1.2, 1.60], [1.6, 1.87], [2.0, 2.14], [2.4, 2.34], [2.8, 2.52]],
        que="Se estira una banda elástica y se mide la fuerza a distintos desplazamientos. **El trabajo es el área bajo "
            "la curva fuerza–desplazamiento**: W = ∫F(x)dx. (1 lb·in ≈ 0.112985 J).",
        modelo=r"W=\int_0^{2.8} F(x)\,dx",
        ing="**Sensores y calibración:** sistemas embebidos leen una magnitud (fuerza, voltaje, consumo) y calculan "
            "energía o trabajo acumulado integrando la lectura. Es igual a medir el consumo de energía de un equipo.",
        prob=lambda xy, p, n: [tb("Trabajo realizado", xy, "lb·in", xl="x (in)", yl="F (lb)",
                                  nota=lambda v: f"≈ {v * 0.112985:.4f} J")]),
    21: dict(
        t="Perfil topográfico · área bajo el perfil", tipo="tabla", pref=4,
        cols=["x", "y (aprox.)"],
        df=[[1, 1.3], [2, 3.7], [3, 4.6], [4, 4.4], [5, 2.7], [6, 2.5], [7, 3.6], [8, 5.3], [9, 6.0], [10, 5.9]],
        que="Se digitaliza una gráfica de altura vs posición (x de 1 a 10) y se estima el área bajo el perfil. Son "
            "**datos ilustrativos, no exactos**: la precisión depende de cómo se digitalizó. Hay 9 subintervalos "
            "(impar), por eso el solucionario usa Simpson 1/3 en 8 y trapecio en el último.",
        modelo=r"A=\int_1^{10} y(x)\,dx",
        ing="**Procesamiento de datos y señales:** el área bajo una curva de carga (CPU, tráfico de red, usuarios "
            "conectados) mide el consumo total de recursos. También se usa para digitalizar gráficas a datos.",
        prob=lambda xy, p, n: [tb("Área bajo el perfil", xy, "u²", yl="y")]),
    22: dict(
        t="Cable suspendido · longitud de la catenaria", tipo="funcion", pref=1, n=8,
        params=[("a", "Parámetro a (ft)", 24.0), ("x1", "Apoyo izquierdo x₁ (ft)", -12.0),
                ("x2", "Apoyo derecho x₂ (ft)", 26.0)],
        que="Un cable cuelga con forma y = a·cosh(x/a). **Su longitud** entre dos apoyos sale de L = ∫√(1+y′²)dx, y "
            "como y′ = senh(x/a) la integral se simplifica a ∫cosh(x/a)dx. Es un modelo simplificado, no el cálculo "
            "de ingeniería de un teleférico.",
        modelo=r"L=\int_{x_1}^{x_2}\cosh\!\left(\frac{x}{a}\right)dx = a\left[\sinh\frac{x_2}{a}-\sinh\frac{x_1}{a}\right]",
        ing="**Redes e infraestructura:** estimar la longitud de cable (fibra óptica, tendido eléctrico) entre postes "
            "es parte del diseño y presupuesto de redes de telecomunicaciones.",
        prob=lambda xy, p, n: [fn("Longitud del cable", lambda x: np.cosh(x / p["a"]), p["x1"], p["x2"], n, "ft",
                                  p["a"] * (np.sinh(p["x2"] / p["a"]) - np.sinh(p["x1"] / p["a"])), yl="cosh(x/a)")]),
    23: dict(
        t="Red de agua · caudal de una tubería", tipo="funcion", pref=1, n=8,
        params=[("v0", "Velocidad central v₀ (in/s)", 80.0), ("R", "Radio R (in)", 0.25)],
        que="El agua en una tubería circular tiene velocidad v(r) = v₀(1 − r/R)^(1/7). **El caudal** se obtiene sumando "
            "los anillos de área 2πr·dr. Cerca de la pared la curva cambia muy rápido, por eso converge lentamente.",
        modelo=r"Q=\int_0^R 2\pi r\,v_0\left(1-\frac{r}{R}\right)^{1/7}dr=\frac{49\pi}{120}\,v_0R^2\cdot 2",
        ing="**Simulación de sistemas:** modelar flujos (agua, tráfico, paquetes de datos) es clave en simulación. "
            "El caudal en una tubería es análogo al ancho de banda en una red.",
        prob=lambda xy, p, n: [fn("Caudal Q", lambda r: 2 * np.pi * r * p["v0"] * np.clip(1 - r / p["R"], 0, None) ** (1 / 7),
                                  0, p["R"], n, "in³/s", 2 * np.pi * p["v0"] * p["R"] ** 2 * 49 / 120,
                                  xl="r (in)", yl="2πr·v(r)")]),
    24: dict(
        t="Diseño de trayectoria · longitud de la cardioide", tipo="funcion", pref=1, n=8,
        params=[("b", "Parámetro b", 5.0)],
        que="Una trayectoria en forma de cardioide: x(t)=2b·cos t − b·cos 2t, y(t)=2b·sen t − b·sen 2t, con t de 0 a 2π. "
            "**La longitud** es L = ∫√(x′²+y′²)dt, que vale exactamente 16b (80 si b=5).",
        modelo=r"L=\int_0^{2\pi}\sqrt{x'(t)^2+y'(t)^2}\,dt = 16\,b",
        ing="**Gráficos por computador y robótica:** las curvas paramétricas definen trayectorias de robots, drones y "
            "animaciones. Conocer su longitud sirve para estimar tiempo y energía del recorrido.",
        prob=lambda xy, p, n: [fn("Longitud de la trayectoria", lambda t: np.sqrt(
            (-2 * p["b"] * np.sin(t) + 2 * p["b"] * np.sin(2 * t)) ** 2 +
            (2 * p["b"] * np.cos(t) - 2 * p["b"] * np.cos(2 * t)) ** 2), 0, 2 * np.pi, n, "u", 16 * p["b"],
            xl="t (rad)", yl="√(x′²+y′²)")]),
    25: dict(
        t="Energía gravitatoria · variación de energía potencial", tipo="funcion", pref=1, n=8,
        params=[("m", "Masa m (kg)", 500.0), ("g0", "g₀ (m/s²)", 9.81), ("R", "Radio de la Tierra R (m)", 6371000.0),
                ("h", "Altura h (m)", 800000.0)],
        que="Un satélite de 500 kg sube 800 km. La gravedad baja con la altura: g(y)=g₀R²/(R+y)². **La energía** es "
            "ΔU = ∫ m·g(y) dy, desde y=0 hasta y=h.",
        modelo=r"\Delta U=m g_0\int_0^h \frac{R^2}{(R+y)^2}dy = \frac{m g_0 R h}{R+h}",
        ing="**Simulación y comunicaciones:** los sistemas satelitales (internet, GPS) se planifican con modelos físicos "
            "integrados numéricamente. Esto es simulación computacional aplicada.",
        prob=lambda xy, p, n: [fn("Variación de energía potencial", lambda y: p["m"] * p["g0"] * p["R"] ** 2 / (p["R"] + y) ** 2,
                                  0, p["h"], n, "J", p["m"] * p["g0"] * p["R"] * p["h"] / (p["R"] + p["h"]),
                                  xl="y (m)", yl="m·g(y) (N)")]),
    26: dict(
        t="Superficie del lago · área con anchuras medidas", tipo="tabla", pref=3,
        cols=["x (mi)", "Ancho w (mi)"],
        df=[[40 + 20 * i, None] for i in range(12)],
        que="Se mide el ancho del lago Erie cada 20 millas (x de 40 a 260) y se estima su área A = ∫w(x)dx. "
            "**Debes escribir tú las anchuras** medidas en el mapa: el solucionario no da una tabla oficial. "
            "Referencia: 9 940 mi². Luego puedes adaptar el ejercicio al lago Titicaca con datos de una fuente confiable.",
        modelo=r"A=\int_{40}^{260} w(x)\,dx \qquad E\%=100\,\frac{|A_{num}-9940|}{9940}",
        ing="**Sistemas de Información Geográfica (SIG):** estimar áreas a partir de mediciones es la base del análisis "
            "geoespacial y del procesamiento de imágenes satelitales.",
        prob=lambda xy, p, n: [dict(tb("Área del lago", xy, "mi²", xl="x (mi)", yl="Ancho (mi)"), exacto=9940.0)]),
    27: dict(
        t="Objeto alargado · volumen y superficie", tipo="tabla", pref=1,
        cols=["z (in)", "d (in)"],
        df=[[0, 0], [1, 2.6], [2, 3.2], [3, 4.8], [4, 5.6], [5, 6.0], [6, 6.2], [7, 6.0], [8, 5.6], [9, 4.8],
            [10, 3.3], [11, 2.6], [12, 0]],
        que="Se mide el diámetro d a lo largo del eje z; el radio es r = d/2. **Volumen:** V = π∫r²dz. La fórmula del "
            "enunciado S = 2π∫r dz **omite la pendiente**, así que no es el área superficial exacta; el área real de "
            "revolución es S = 2π∫r·√(1+r′²)dz y necesita estimar r′.",
        modelo=r"V=\pi\int_0^{12} r^2\,dz \qquad S_{enunciado}=2\pi\int_0^{12} r\,dz",
        ing="**Modelado 3D y CAD:** calcular volúmenes de piezas a partir de perfiles es útil en diseño asistido por "
            "computador e impresión 3D (cuánto material se necesita).",
        prob=lambda xy, p, n: [
            tb("Volumen V", (xy[0], np.pi * (xy[1] / 2) ** 2), "in³", xl="z (in)", yl="π·r²"),
            tb("Superficie con la fórmula del enunciado (simplificada)", (xy[0], 2 * np.pi * xy[1] / 2), "in²",
               xl="z (in)", yl="2π·r")]),
}


def entradas(k, widget=True):
    """Lee datos por teclado (widget=True) o toma los valores por defecto."""
    e = EJ[k]
    xy, p, n = None, {}, e.get("n", 0)
    if e["tipo"] == "tabla":
        base = pd.DataFrame(e["df"], columns=e["cols"], dtype=float)
        d = st.data_editor(base, num_rows="dynamic", key=f"de{k}") if widget else base
        d = d.dropna(how="all").apply(pd.to_numeric, errors="coerce")
        xy = (d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float))
    else:
        cols = st.columns(len(e["params"]) + 1) if widget else None
        for i, (key, lab, val) in enumerate(e["params"]):
            p[key] = cols[i].number_input(lab, value=float(val), format="%g", key=f"p{k}{key}") if widget else val
        n = int(cols[-1].number_input("Subintervalos n", min_value=2, value=n, step=1, key=f"n{k}")) if widget else n
    return xy, p, n


def problemas(k, xy, p, n):
    ps = EJ[k]["prob"](xy, p, n)
    for q in ps:
        q["pref"] = EJ[k]["pref"]
    return ps


# ----------------------------------------------------------------------------
# 4) PÁGINAS
# ----------------------------------------------------------------------------
def pag_inicio():
    st.title("📐 Integración numérica aplicada a La Paz")
    st.caption("Desafío grupal de Métodos Numéricos · Trapecio compuesto, Simpson 1/3 y Simpson 3/8")
    st.markdown("""
**¿Qué hace esta página?** Calcula integrales a partir de datos que **tú escribes con el teclado** y te muestra
el resultado, el error, una tabla y un gráfico. Es solo para verificar tus cálculos manuales.

**Cómo usarla en 3 pasos**
1. Elige una opción en el **menú de la izquierda**.
2. Escribe tus datos (o usa los que ya vienen del solucionario).
3. Mira la tabla de resultados: ¿cuánto salió con cada método y qué tan cerca está del valor exacto?
""")
    st.subheader("Los 3 métodos en palabras simples")
    st.markdown("""
Integrar = **calcular el área bajo una curva**. Dividimos el intervalo [a, b] en *n* tramos iguales de ancho
**h = (b − a)/n** y sumamos áreas pequeñas.
""")
    st.dataframe(pd.DataFrame({
        "Método": ["Trapecio compuesto", "Simpson 1/3", "Simpson 3/8"],
        "Idea": ["Une los puntos con rectas", "Une de a 2 tramos con una parábola", "Une de a 3 tramos con una cúbica"],
        "Condición sobre n": ["Cualquier n", "n par", "n múltiplo de 3"],
        "Coeficientes": ["1, 2, 2, …, 2, 1", "1, 4, 2, 4, …, 4, 1", "1, 3, 3, 2, 3, 3, …, 3, 1"],
        "Factor": ["h/2", "h/3", "3h/8"]}), hide_index=True)
    st.latex(r"E_a=|I_{ref}-I_{aprox}| \qquad E\%=100\,\frac{E_a}{|I_{ref}|}")
    st.info("Con n = 7 (como en los ejercicios 19 y 20) se combina Simpson 1/3 en 4 tramos + Simpson 3/8 en 3 tramos.")
    st.subheader("Ejercicios incluidos")
    st.dataframe(pd.DataFrame({"Ejercicio": list(EJ), "Tema": [v["t"] for v in EJ.values()]}), hide_index=True)


def pag_calc():
    st.title("🧮 Calculadora libre")
    st.write("Escribe tus propios datos. Elige si tienes una **tabla (x, y)** o una **función f(x)**.")
    modo = st.radio("Tengo…", ["Una tabla de datos (x, y)", "Una función f(x)"], horizontal=True)
    unidad = st.text_input("Unidad del resultado (opcional)", value="")
    try:
        if modo.startswith("Una tabla"):
            st.caption("Escribe en la tabla (puedes agregar filas abajo). x debe ir creciente y con paso igual.")
            d = st.data_editor(pd.DataFrame({"x": [0.0, 0.5, 1.0, 1.5, 2.0], "y": [0.0, 0.25, 1.0, 2.25, 4.0]}),
                               num_rows="dynamic", key="calc_df").apply(pd.to_numeric, errors="coerce")
            x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
        else:
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            expr = c1.text_input("f(x) =", value="x^2", help="Ej: x^2, sen(x)+3, exp(-x)/(1+x^2)")
            a = c2.number_input("a", value=0.0, format="%g"); b = c3.number_input("b", value=2.0, format="%g")
            n = int(c4.number_input("n", min_value=2, value=4, step=1))
            x = np.linspace(a, b, n + 1); y = evaluar(expr, x)
        conoce = st.checkbox("Conozco el valor exacto o de referencia")
        ref = st.number_input("Valor exacto", value=8 / 3, format="%.10g") if conoce else None
        mostrar(dict(nombre="Resultados", x=x, y=y, fin=lambda I: I, unidad=unidad, exacto=ref, pref=1))
    except ValueError as e:
        st.error(f"⚠️ {e}")
    with st.expander("✅ Prueba de validación: f(x)=x² en [0, 2] (exacto = 8/3)"):
        filas = []
        for n in (4, 8):
            xx = np.linspace(0, 2, n + 1); r = resolver(xx, xx ** 2, 2 / n)
            for m in METODOS[:2]:
                v = r[m][0]
                filas.append({"n": n, "Método": m, "Resultado": fmt(v), "Exacto (8/3)": fmt(8 / 3),
                              "Error absoluto": fmt(abs(v - 8 / 3))})
        st.dataframe(pd.DataFrame(filas), hide_index=True)
        st.caption("Simpson es exacto para polinomios de grado ≤ 3, por eso su error es casi 0. El trapecio mejora al duplicar n.")


def pag_ejercicios():
    st.title("📚 Ejercicios resueltos")
    k = st.selectbox("Elige el ejercicio", list(EJ), format_func=lambda k: f"Ejercicio {k} · {EJ[k]['t']}")
    e = EJ[k]
    t1, t2, t3 = st.tabs(["1️⃣ ¿De qué trata?", "2️⃣ Datos y resultados", "3️⃣ Ingeniería de Sistemas"])
    with t1:
        st.subheader(e["t"]); st.markdown(e["que"]); st.latex(e["modelo"])
    with t2:
        st.caption("✍️ Puedes cambiar los datos con el teclado: la tabla se recalcula sola.")
        try:
            xy, p, n = entradas(k)
            if e["tipo"] == "tabla" and (np.isnan(xy[0]).any() or np.isnan(xy[1]).any()):
                st.info("Completa todas las celdas con números para calcular.")
            else:
                for q in problemas(k, xy, p, n):
                    mostrar(q)
        except (ValueError, KeyError) as er:
            st.error(f"⚠️ {er}")
    with t3:
        st.markdown(e["ing"])


def conclusion(vals, ref):
    if not vals:
        return "No hubo método aplicable."
    if ref:
        m = min(vals, key=lambda k: abs(vals[k] - ref))
        return (f"El método más cercano al valor de referencia fue **{m}** "
                f"(error {100 * abs(vals[m] - ref) / abs(ref):.4f} %).")
    return ("No hay valor exacto, por eso no se calcula el error verdadero; los métodos solo se comparan entre sí. "
            "Los resultados dependen de la calidad de los datos.")


def pag_informe():
    st.title("📄 Informe técnico")
    st.write("Resumen con el formato pedido: **Problema · Modelo · Datos · Cálculo · Resultados · Conclusión**.")
    sel = st.multiselect("Ejercicios a incluir", list(EJ), default=[19, 23], format_func=lambda k: f"Ejercicio {k}")
    md = [f"# Informe técnico · Integración numérica\n_Generado el {ahora()} (hora de La Paz)_\n"]
    resumen = []
    for k in sel:
        e = EJ[k]
        xy, p, n = entradas(k, widget=False)
        if e["tipo"] == "tabla" and np.isnan(xy[1]).any():
            st.warning(f"Ejercicio {k}: faltan datos medidos. Complétalos en la página de Ejercicios.")
            continue
        for q in problemas(k, xy, p, n):
            h = check_uniform(q["x"]); res = resolver(np.asarray(q["x"], float), np.asarray(q["y"], float), h)
            df, vals = tabla_res(q, res)
            m = mejor(vals, q["pref"]); ref = q["exacto"]
            resumen.append({"Ejercicio": k, "Problema": q["nombre"], "Trapecio": fmt(vals.get(METODOS[0])),
                            "Simpson": fmt(vals.get(m)), "Unidad": q["unidad"],
                            "Exacto/ref.": fmt(ref) if ref else "—", "Método Simpson": m})
            with st.expander(f"Ejercicio {k} · {q['nombre']}", expanded=False):
                st.markdown(f"**1. Problema.** {e['que']}")
                st.markdown("**2. Modelo.**"); st.latex(e["modelo"])
                st.markdown(f"**3. Datos.** n = {len(q['x']) - 1}, h = {h:g}, intervalo [{q['x'][0]:g}, {q['x'][-1]:g}].")
                st.markdown("**4. Cálculo.** Trapecio (cualquier n), Simpson 1/3 (n par), Simpson 3/8 (n múltiplo de 3).")
                st.markdown("**5. Resultados.**"); st.dataframe(df, hide_index=True)
                st.markdown(f"**6. Conclusión.** {conclusion(vals, ref)}")
            md += [f"## Ejercicio {k} · {q['nombre']}", f"**Problema.** {e['que']}", f"**Modelo.** ${e['modelo']}$",
                   f"**Datos.** n = {len(q['x']) - 1}, h = {h:g}", "**Resultados.**",
                   "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
            md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values.tolist()]
            md += [f"**Conclusión.** {conclusion(vals, ref)}", f"**Aplicación en Ing. de Sistemas.** {e['ing']}", ""]
    if resumen:
        st.subheader("Resumen de resultados")
        st.dataframe(pd.DataFrame(resumen), hide_index=True)
        st.download_button("⬇️ Descargar informe (.md)", "\n".join(md), file_name="informe_integracion.md")


def pag_sistemas():
    st.title("💻 Aplicación en Ingeniería de Sistemas")
    st.write("La integración numérica aparece cada vez que un sistema **acumula** una cantidad que cambia en el tiempo.")
    st.dataframe(pd.DataFrame({
        "Concepto del desafío": ["Integrar velocidad → distancia", "Integrar potencia → energía", "Área bajo una curva de carga",
                                 "Caudal en una tubería", "Longitud de curvas", "Área medida en mapas", "Volumen de perfiles"],
        "Aplicación en Sistemas": ["Telemetría y GPS en flotas (IoT)", "Consumo de energía en servidores y centros de datos",
                                   "Uso total de CPU, memoria o ancho de banda", "Simulación de flujo de datos y tráfico en redes",
                                   "Trayectorias de robots/drones, gráficos por computador", "Sistemas de Información Geográfica (SIG)",
                                   "CAD, modelado 3D e impresión 3D"]}), hide_index=True)
    st.subheader("Más conexiones")
    st.markdown("""
- **Ciencia de datos / IA:** el área bajo la curva ROC (AUC) de un modelo se calcula con la regla del trapecio.
- **Calidad de software:** validar con una prueba conocida (x² en [0,2] = 8/3) es una prueba unitaria; comparar n=4 y n=8 mide la convergencia.
- **Eficiencia:** más subintervalos = más precisión pero más tiempo de cómputo (compromiso precisión vs costo).
- **Manejo de errores:** validar datos (x creciente, paso uniforme, n compatible) evita resultados falsos, como en cualquier sistema.
""")
    st.subheader("Consejo para la defensa oral")
    st.info("Explica cada método con una frase: *trapecio = rectas, Simpson 1/3 = parábolas (n par), Simpson 3/8 = cúbicas "
            "(n múltiplo de 3)*, y muestra cómo el error baja al aumentar n.")


PAGS = {"🏠 Inicio": pag_inicio, "🧮 Calculadora": pag_calc, "📚 Ejercicios": pag_ejercicios,
        "📄 Informe": pag_informe, "💻 Ing. de Sistemas": pag_sistemas}
st.sidebar.title("Menú")
PAGS[st.sidebar.radio("Ir a:", list(PAGS))]()
st.sidebar.caption("Métodos Numéricos · La Paz, Bolivia")
