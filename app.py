import os
import re
import json
import glob

import numpy as np
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MODEL_DIR", _HERE)

st.set_page_config(page_title="Predicción de desarrollo", layout="wide")
TARGETS = ["extensivo", "redesarrollo", "total"]

# Data Loading
@st.cache_data(show_spinner="Cargando datos...")
def load_parquets(data_dir: str):
    """
    Find and load the scoring and definitions parquet files by content
    """
    scoring, defs = None, None
    for path in glob.glob(os.path.join(data_dir, "*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        cols = set(df.columns)
        if "geometry_geojson" in cols or any(c.startswith("prob_") for c in cols):
            scoring = df
        elif {"target", "definicion"}.issubset(cols):
            defs = df
    return scoring, defs

def norm_target(value: str) -> str:
    """
    Normalise the target label to the bare suffix: 'target_Redesarrollo' -> 'redesarrollo'
    """
    v = str(value).strip().lower()
    v = re.sub(r"^target[_\s]*", "", v)
    return v

def build_defintions(defs: pd.DataFrame) -> dict:
    """
    Map normalised target -> {'nombre': ..., 'definicion': ...}
    """
    out = {}
    if defs is None:
        return out
    name_col = "nombre_display" if "nombre_display" in defs.columns else None
    for _, row in defs.iterrows():
        key = norm_target(row['target'])
        out[key] = {
            "nombre": str(row[name_col]) if name_col else key.capitalize(),
            "definicion": str(row['definicion']),
        }
    return out

def geojson_to_polygon(geojson_str):
    """
    Convert a GeoJSON geometry string to deck.gl polygon coords (list of [lon, lat])
    """
    try:
        geom = json.loads(geojson_str)
    except (TypeError, json.JSONDecodeError):
        return None
    t = geom.get("type")
    coords = geom.get('coordinates')
    if t == "Polygon":
        return coords[0] if coords else None
    if t == "MultiPolygon":
        return coords[0][0] if coords and coords[0] else None
    return None

def parse_shap(shap_str):
    """
    Parse a shap_yop_* JSON string into a list of {variable, valor, shap} dicts
    """
    try:
        data = json.loads(shap_str)
    except (TypeError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []

# Load
scoring, defs_df = load_parquets(DATA_DIR)

if scoring is None:
    st.error(
        f"No se encontró el Parquet de scoring en {DATA_DIR}.\n\n"
        "Coloca 'scoring_enriquecido.parquet' en esa carpeta o define 'MODEL_DIR'"
    )
    st.stop()
definciones = build_defintions(defs_df)
available = [t for t in TARGETS if f"clasificacion_{t}" in scoring.columns]
if not available:
    st.error("El Parquet no contiene columnas por target")
    st.stop()

# Sidebar: target selection + definition
st.sidebar.header("Target")

def target_label(t):
    return definciones.get(t, {}).get("nombre", t.capitalize())

target = st.sidebar.selectbox("Selecciona un target", available, format_func=target_label)

info = definciones.get(target)
if info:
    st.sidebar.markdown(f"**{info['nombre']}**")
    st.sidebar.caption(info["definicion"])

# Assemble the working frame for the selected target
probs_col = f"prob_{target}"
clas_col = f"clasificacion_{target}"

df = pd.DataFrame({
    "s2_token": scoring["s2_token"].astype(str),
    "prob": scoring[probs_col] if probs_col in scoring.columns else np.nan,
    "clasificacion": scoring[clas_col],
    "row_ix": scoring.index.to_numpy(),
})
for field in ["frase_porque", "frase_confiabilidad", "frase_frecuencia", "frase_percentil"]:
    col = f"{field}_{target}"
    df[field] = scoring[col] if col in scoring.columns else ""

df["polygon"] = scoring["geometry_geojson"].apply(geojson_to_polygon)
df = df[df["polygon"].notna()].reset_index(drop=True)

# Header + Summary
st.title("Predicción de Desarrollo")
if info:
    st.caption(info["definicion"])

counts = df["clasificacion"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Celads", f"{len(df):,}")
c2.metric("Alto", f"{int(counts.get('Alto', 0)):,}")
c3.metric("Medio", f"{int(counts.get('Medio', 0)):,}")
c4.metric("Bajo", f"{int(counts.get('Bajo', 0)):,}")

# Map
import pydeck as pdk

st.subheader("Mapa")
color_mode = st.radio(
    "Coloreado",
    ["Clasificación (Alt0/Medio/Bajo)", "Continuo (probabilidad)"],
    horizontal = True
)

df["prob_pct"] = (pd.to_numeric(df["prob"], errors="coerce") * 100).round(1).astype("Float64").astype(str) + "%"

CLASS_COLORS = {
    "Alto": [50, 180, 50],
    "Medio": [230, 180, 40],
    "Bajo": [220, 50, 47],
}

if color_mode.startswith("Clasificación"):
    rgb = df["clasificacion"].map(lambda c: CLASS_COLORS.get(c, [140, 140, 140]))
    df["r"] = [c[0] for c in rgb]
    df["g"] = [c[1] for c in rgb]
    df["b"] = [c[2] for c in rgb]
else:
    p = pd.to_numeric(df["prob"], errors="coerce").fillna(0).clip(0, 1)
    df["r"] = ((1 -p) * 255).astype(int)
    df["g"] = (p * 255).astype(int)
    df["b"] = 40

layer = pdk.Layer(
            "PolygonLayer",
            id = "cells",
            data=df,
            get_polygon = "polygon",
            get_fill_color = "[r, g, b, 170]",
            get_line_color = [60, 60, 60, 160],
            line_width_min_pixels = 1,
            stroked = True,
            filled = True,
            pickable = True,
            auto_highlight = True,
        )

all_coords = np.array([pt for poly in df["polygon"] for pt in poly])
view = pdk.ViewState(
        latitude = float(all_coords[: ,1].mean()),
        longitude = float(all_coords[:, 0].mean()),
        zoom = 9,
)

event = st.pydeck_chart(
        pdk.Deck(
            layers = [layer],
            initial_view_state = view,
            map_provider= "carto",
            map_style = "light",
            tooltip = 
            {
                "html": "<b>{clasificacion}</b><br/>Prob: {prob_pct}",
                "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white"},
            },
        ),
        on_select = "rerun",
        selection_mode = "single-object",
    )


# Click Detail: simple view + "ver más"
picked = None
sel = event.selection if event else None
if sel:
    hits = (sel.get("objects") or {}).get("cells") or []
    if hits:
        picked = hits[0]
if picked is None:
    st.info("Haz clic en una celda del mapa para ver su detalle")
else:
    st.markdown(f"## Celda '{picked['s2_token']}")

    a, b = st.columns([1, 3])
    a.metric("Clasificación", picked.get("clasificacion", "-"))
    a.metric("Probabilidad", picked.get("prob_pct", "-"))
    with b:
        if picked.get("frase_porque"):
            st.markdown(f"**¿Por qué?** {picked['frase_porque']}")
        if picked.get("frase_frecuencia"):
            st.markdown(f"**Frecuencia** {picked['frase_frecuencia']}")
        if picked.get("frase_percentil"):
            st.markdown(f"**Percentil** {picked['frase_percentil']}")
        if picked.get("frase_confiabilidad"):
            st.caption(picked['frase_confiabilidad'])

    with st.expander("Ver más: detalle técnico (SHAP)"):
        shap_col = f"shap_top_{target}"
        raw = scoring.loc[int(picked["row_ix"]), shap_col] if shap_col in scoring.columns else None
        shap_rows = parse_shap(raw)
        if shap_rows:
            shap_df = pd.DataFrame(shap_rows)
            rename = {"variable": "Variable", "valor": "Valor", "shap": "Peso (SHAP)"}
            shap_df = shap_df.rename(columns={k: v for k, v in rename.items() if k in shap_df.columns})
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
            st.caption(
                "El peso SHAP indica cuánto empuja cada variable la predicción "
                "hacia arriba (positivo) o hacia abajo (nehativo)"
            )
        else:
            st.caption("No hay detalle SHAP disponible para esta celda")
