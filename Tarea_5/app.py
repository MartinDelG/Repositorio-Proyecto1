# app.py  →  python app.py    (Dash >= 2.16)
# Pestaña 1: Visión General (KPIs + 3 gráficas; la tercera es Prioridad × SLA)
# Pestaña 2: Predicción (solo TIEMPO DE DURACIÓN usando tu modelo)

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px

# ===================== RUTAS / CONFIG =====================
ROOT = Path(__file__).resolve().parent

# Datos (para KPIs y poblar dropdowns)
CSV_CANDIDATES = [
    ROOT / "datos_procesados.csv",
    ROOT / "data" / "datos_procesados.csv",
]
XLSX_CANDIDATES = [
    ROOT / "datos_procesados.xlsx",
    ROOT / "data" / "datos_procesados.xlsx",
]

# Artefactos del modelo (ajusta si están en otra ruta)
MODEL_CANDIDATES = [ROOT / "model.pkl", ROOT / "linreg.joblib", ROOT / "model.joblib"]
CMAP_PATH  = ROOT / "cluster_maps.json"
HIGH_PATH  = ROOT / "high_card_cols.json"
LOW_PATH   = ROOT / "low_card_cols.json"
UNION_PATH = ROOT / "cols_union.json"
CFG_PATH   = ROOT / "predict_config.json"  # opcional

# Columnas que pide la UI para predecir (las que usa tu modelo)
PRED_COLS = [
    "caller_id", "opened_by", "location", "category", "subcategory",
    "u_symptom", "priority", "assignment_group", "assigned_to",
    "u_priority_confirmation", "contact_type"
]

# Defaults si no existe predict_config.json
USE_LOG_TARGET_DEFAULT    = False
CLIP_NON_NEGATIVE_DEFAULT = True
SLA_TARGET_HOURS_DEFAULT  = 8.0

# ===================== CARGA DE DATOS =====================
def load_df_opts():
    # 1) CSV
    for p in CSV_CANDIDATES:
        if p.exists():
            try:
                return pd.read_csv(p, encoding="ISO-8859-1", low_memory=False)
            except Exception:
                pass
    # 2) Excel
    for p in XLSX_CANDIDATES:
        if p.exists():
            try:
                return pd.read_excel(p)
            except Exception:
                pass
    # 3) Fallback mínimo para no romper
    return pd.DataFrame({
        "duracion_horas":[2,4,6,8,5,9,3,12],
        "priority":["2 - High","3 - Moderate","1 - Critical","2 - High","3 - Moderate","2 - High","1 - Critical","4 - Low"],
        "incident_state":["Closed","Closed","Resolved","Closed","Resolved","Closed","In Progress","Closed"],
        "caller_id":["Caller 1"]*8,
        "opened_by":["Opened by 1"]*8,
        "location":["Location 1"]*8,
        "category":["Category 1"]*8,
        "subcategory":["Subcategory 1"]*8,
        "u_symptom":["Symptom 1"]*8,
        "assignment_group":["Group 1"]*8,
        "assigned_to":["Resolver 1"]*8,
        "u_priority_confirmation":["true"]*8,
        "contact_type":["Phone"]*8
    })

df_opts = load_df_opts()

# Normaliza nombre de duración si vino distinto
if "duracion_horas" not in df_opts.columns:
    for cand in ["duration_hours", "duracion", "ttr_horas"]:
        if cand in df_opts.columns:
            df_opts = df_opts.rename(columns={cand: "duracion_horas"})
            break

# ===================== CARGA DEL MODELO =====================
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            try:
                return load(p)
            except Exception:
                continue
    raise FileNotFoundError("No encontré model.pkl / linreg.joblib / model.joblib junto a app.py")

def load_json_or_raise(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Falta archivo: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

linreg = load_model()
cluster_maps   = load_json_or_raise(CMAP_PATH)            # dict: {col: {valor: cluster_id}}
high_card_cols = load_json_or_raise(HIGH_PATH)            # lista alta cardinalidad
low_card_cols  = load_json_or_raise(LOW_PATH)             # lista baja cardinalidad
cols_union     = pd.Index(load_json_or_raise(UNION_PATH)) # columnas finales (dummies)

# Config opcional
if CFG_PATH.exists():
    cfg = json.load(open(CFG_PATH, "r"))
    USE_LOG_TARGET    = bool(cfg.get("USE_LOG_TARGET", USE_LOG_TARGET_DEFAULT))
    CLIP_NON_NEGATIVE = bool(cfg.get("CLIP_NON_NEGATIVE", CLIP_NON_NEGATIVE_DEFAULT))
    SLA_TARGET_HOURS  = float(cfg.get("SLA_TARGET_HOURS", SLA_TARGET_HOURS_DEFAULT))
else:
    USE_LOG_TARGET    = USE_LOG_TARGET_DEFAULT
    CLIP_NON_NEGATIVE = CLIP_NON_NEGATIVE_DEFAULT
    SLA_TARGET_HOURS  = SLA_TARGET_HOURS_DEFAULT

# ===================== TRANSFORMACIÓN (MISMA QUE ENTRENAMIENTO) =====================
def transformar_nuevo(df_new: pd.DataFrame,
                      cluster_maps: dict,
                      high_card_cols: list,
                      low_card_cols: list,
                      cols_union: pd.Index) -> pd.DataFrame:
    Z = df_new.copy()

    # 1) Alta cardinalidad → *_cluster (valores no vistos => -1)
    for col in high_card_cols:
        newc = f"{col}_cluster"
        mapa = cluster_maps.get(col, {})
        Z[newc] = Z[col].astype(str).map(mapa).fillna(-1).astype(int).astype(str)

    # 2) One-hot (baja card + *_cluster) con drop_first=True
    cluster_cols = [f"{c}_cluster" for c in high_card_cols]
    base_cols = [c for c in (low_card_cols + cluster_cols) if c in Z.columns]
    Z_cat = pd.get_dummies(Z[base_cols], drop_first=True)

    # 3) Alinear a columnas de entrenamiento
    Z_cat = Z_cat.reindex(columns=cols_union, fill_value=0)
    return Z_cat

# ===================== OPCIONES PARA DROPDOWNS =====================
def opciones_columna(col: str):
    # Primero intenta con los valores del dataset
    if col in df_opts.columns:
        vals = df_opts[col].dropna().astype(str).unique().tolist()
        if vals:
            return sorted(vals)[:1000]
    # Si es alta cardinalidad, usa las llaves del cluster_map como fallback
    if col in high_card_cols and col in cluster_maps:
        return sorted(list(map(str, cluster_maps[col].keys())))[:1000]
    # Fallback genérico
    return []

controls_pred = []
for col in PRED_COLS:
    opts = [{"label": v, "value": v} for v in opciones_columna(col)]
    default = opts[0]["value"] if opts else ""
    controls_pred.append(
        html.Div([
            html.Label(col),
            dcc.Dropdown(id=f"pred-{col}", options=opts, value=default, clearable=False)
        ], style={"marginBottom": "10px"})
    )

# ===================== VISIÓN GENERAL: KPIs + GRÁFICAS =====================
def kpis(d: pd.DataFrame):
    total = len(d)
    ttr   = float(d["duracion_horas"].mean()) if "duracion_horas" in d.columns else 0.0
    sla_p = float(((d["duracion_horas"] <= SLA_TARGET_HOURS).mean()*100)) if "duracion_horas" in d.columns else 0.0
    return int(total), ttr, sla_p

total_inc, ttr_avg, sla_pct = kpis(df_opts)

# 1) Histograma de duración
fig_hist = px.histogram(
    df_opts.dropna(subset=["duracion_horas"]) if "duracion_horas" in df_opts.columns else pd.DataFrame({"duracion_horas":[]}),
    x="duracion_horas", nbins=10, title="Distribución de la duración de incidentes (horas)",
    labels={"duracion_horas":"Horas"}
)

# 2) Pie de prioridad
fig_pie = px.pie(df_opts, names="priority", title="Distribución de Prioridad", hole=0.3) \
    if "priority" in df_opts.columns else px.pie(pd.DataFrame({"x":[],"y":[]}), names="x", values="y")

# 3) NUEVA: Barras Prioridad × SLA (Cumple / No cumple)
if "duracion_horas" in df_opts.columns:
    df_opts["sla_met_calc"] = (df_opts["duracion_horas"] <= SLA_TARGET_HOURS).astype("boolean")
else:
    df_opts["sla_met_calc"] = pd.NA

if {"priority", "sla_met_calc"}.issubset(df_opts.columns) and df_opts["priority"].notna().any():
    dff = df_opts.dropna(subset=["priority", "sla_met_calc"]).copy()
    dff["SLA"] = np.where(dff["sla_met_calc"], "Cumple", "No cumple")
    # Ordena prioridades tipo "1 - Critical"..."4 - Low"
    prio_order = sorted(
        dff["priority"].astype(str).unique(),
        key=lambda s: (s.split(" - ")[0] if s and s[0].isdigit() else "9", s)
    )
    bar_counts = dff.groupby(["priority", "SLA"]).size().reset_index(name="count")
    fig_prio_sla = px.bar(
        bar_counts, x="priority", y="count", color="SLA",
        barmode="group",
        category_orders={"priority": prio_order, "SLA": ["Cumple", "No cumple"]},
        title="Incidentes por Prioridad (separado por cumplimiento de SLA)",
        labels={"priority": "Prioridad", "count": "# Incidentes"}
    )
else:
    fig_prio_sla = px.bar(title="Incidentes por Prioridad (SLA)")

# ===================== APP / LAYOUT =====================
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

card = {"background":"#f2f2f2","borderRadius":"12px","padding":"18px","textAlign":"center"}
kpi_val = {"fontSize":"28px","fontWeight":"700","marginTop":"6px"}

app.layout = html.Div(style={"padding":"14px"}, children=[
    dcc.Tabs(id="tabs", value="tab-overview", children=[
        # ---------- PESTAÑA 1: VISIÓN GENERAL ----------
        dcc.Tab(label="Visión General", value="tab-overview", children=[
            html.H2("Resumen del estado del servicio: TTR, SLA y volumen de incidentes"),
            html.Div([
                html.Div([html.Div("# Total de Incidentes"), html.Div(f"{total_inc:,}", style=kpi_val)], style=card),
                html.Div([html.Div("Tiempo Promedio de resolución"), html.Div(f"{ttr_avg:,.2f} h", style=kpi_val)], style=card),
                html.Div([html.Div("%SLA cumplido"), html.Div(f"{sla_pct:,.1f}%", style=kpi_val)], style=card),
            ], style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"12px","margin":"12px 0"}),

            html.Div([
                html.Div([html.H3("Distribución de la duración de incidentes"), dcc.Graph(figure=fig_hist)], style={"flex":1}),
                html.Div([html.H3("Distribución de Prioridad"), dcc.Graph(figure=fig_pie)], style={"flex":1}),
                html.Div([html.H3("Incidentes por Prioridad y SLA"), dcc.Graph(figure=fig_prio_sla)], style={"flex":1}),
            ], style={"display":"flex","gap":"12px"})
        ]),

        # ---------- PESTAÑA 2: PREDICCIÓN ----------
        dcc.Tab(label="Predicción de Incidente", value="tab-predict", children=[
            html.H2("Ingrese características del incidente para estimar duración"),
            html.Div([
                html.Div(controls_pred, style={"flex": 1, "paddingRight": "16px"}),

                # Panel derecho: SOLO tiempo (eliminado 'riesgo')
                html.Div([
                    html.Div([
                        html.Div("TIEMPO DE DURACIÓN (predicho)"),
                        html.H1(id="pred-tiempo", children="—")
                    ], style={**card, "minHeight":"160px", "marginBottom":"12px"}),

                    html.Button("Predecir", id="btn-predict", n_clicks=0, style={"marginTop":"14px","width":"100%"}),
                    html.Div(id="pred-error", style={"color":"crimson","marginTop":"8px"})
                ], style={"flex": 1})
            ], style={"display": "flex", "gap": "24px"})
        ]),
    ])
])

# ===================== CALLBACK: PREDICCIÓN =====================
@app.callback(
    Output("pred-tiempo","children"),
    Output("pred-error","children"),
    Input("btn-predict","n_clicks"),
    [State(f"pred-{c}", "value") for c in PRED_COLS]
)
def hacer_prediccion(n_clicks, *vals):
    if not n_clicks:
        return "—", ""
    try:
        # 1) Registro EXACTO que espera el modelo
        registro = {col: ("" if v is None else str(v)) for col, v in zip(PRED_COLS, vals)}
        df_new = pd.DataFrame([registro])

        # 2) Transformación igual que en entrenamiento
        X_nuevo = transformar_nuevo(
            df_new=df_new,
            cluster_maps=cluster_maps,
            high_card_cols=high_card_cols,
            low_card_cols=low_card_cols,
            cols_union=cols_union
        )

        # 3) Predicción
        y_hat = linreg.predict(X_nuevo)
        if USE_LOG_TARGET:
            y_hat = np.expm1(y_hat)
        if CLIP_NON_NEGATIVE and not USE_LOG_TARGET:
            y_hat = np.maximum(y_hat, 0)

        y = float(np.ravel(y_hat)[0])

        # DEBUG opcional: cuántas features entran distintas de 0
        # print("Features != 0:", int((X_nuevo != 0).sum().sum()))

        return f"{y:.2f} h", ""

    except Exception as e:
        return "—", f"Error al predecir: {e}"

# ===================== RUN =====================
if __name__ == "__main__":
    app.run(debug=True)
