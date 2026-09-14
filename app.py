
# @title
"""
TOPO-VCE V3 — Interfaz Streamlit (app.py)
Se apoya exclusivamente en topo_core.py (núcleo intocable, ya validado).
Esta capa NO contiene matemática: solo carga, muestra y explica resultados.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import topo_core

st.set_page_config(page_title="TOPO-VCE V3", layout="wide")

st.title("🏔️ TOPO-VCE V3")
st.write("Plataforma de Análisis Topográfico y Detección de Variables Clave Emergentes (VCE)")
st.divider()


def construir_narrativa(df_scores: pd.DataFrame) -> list[str]:
    """
    Arma el texto explicativo para cada variable marcada como VCE (Sección 12
    del manual). Se apoya únicamente en los valores ya calculados por
    topo_core.py: no interpreta nada por su cuenta ni afirma causalidad.

    Nota para futuras versiones: esta función es el punto de enganche natural
    para una capa opcional de interpretación por LLM (funcionalidad de pago
    planeada a futuro). Por ahora arma el texto con reglas fijas y explícitas,
    sin depender de ningún modelo externo, para que el MVP no tenga esa
    dependencia. El día que se agregue el LLM, debería reemplazar solo el
    armado del texto final, tomando como entrada la misma fila de evidencia
    que ya se arma acá — no debería tocar topo_core.py para nada.
    """
    narrativas = []

    # Corrección del bug anterior: la columna se llama 'VCE', no 'vce_signal'.
    if "VCE" not in df_scores.columns:
        return narrativas

    vce_vars = df_scores[df_scores["VCE"] == True]  # noqa: E712 (comparación explícita a propósito, más clara de leer)

    for _, row in vce_vars.iterrows():
        var_name = row.get("variable", "variable sin nombre")
        stab = row.get("stability", 0.0)

        canales = []
        if row.get("dep_signal", False):
            canales.append("dependencia monotónica (Spearman)")
        if row.get("mi_signal", False):
            canales.append("dependencia general (información mutua)")
        if row.get("bic_signal", False):
            canales.append("estructura marginal adicional (BIC)")

        str_canales = ", ".join(canales) if canales else "más de un criterio combinado"

        texto = (
            f"**`{var_name}`** aparece como Variable Clave Emergente porque su evidencia "
            f"supera el umbral estadístico calibrado contra el null sintético en: "
            f"{str_canales}. Esta señal se mantuvo estable en el "
            f"**{stab:.0%}** de las particiones de validación cruzada. "
            f"*(Esto describe estructura geométrica y estadística observada; "
            f"no implica causalidad ni relevancia de negocio por sí solo.)*"
        )
        narrativas.append(texto)

    return narrativas


# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Sube tu dataset (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # .lower() para no fallar con extensiones en mayúsculas (ej. Datos.CSV)
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Archivo {uploaded_file.name} cargado con éxito.")
        st.write(f"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} variables.")

        if st.button("Ejecutar Análisis TOPO-VCE", type="primary"):
            with st.spinner("Procesando datos en el motor geométrico..."):
                resultados = topo_core.analyze(df)

            st.success("¡Análisis completado exitosamente!")

            # ------------------------------------------------------------------
            # 1. Tabla de señales VCE
            # ------------------------------------------------------------------
            st.subheader("📋 Tabla de Señales VCE")
            df_scores = resultados["scores"]
            st.dataframe(df_scores, use_container_width=True)

            st.divider()

            # ------------------------------------------------------------------
            # 2. Narrativa (Sección 12 del manual)
            # ------------------------------------------------------------------
            st.subheader("📝 Narrativa de Hallazgos VCE")
            narrativas = construir_narrativa(df_scores)

            if narrativas:
                for texto in narrativas:
                    st.markdown(f"- {texto}")
            else:
                st.info(
                    "No se detectaron variables con señal VCE estadísticamente "
                    "significativa bajo las restricciones actuales."
                )

            st.divider()

            # ------------------------------------------------------------------
            # 3. Paisaje topográfico 3D — solo observaciones reales
            # ------------------------------------------------------------------
            st.subheader("🌋 Superficie Topográfica 3D (Geometría Local)")

            xy = resultados["xy"]
            geom = resultados["geometry"]
            density = geom["density"]
            slope = geom["slope"]
            curvature = geom["curvature"]
            state = geom["state"]

            hover_text = [
                f"<b>Fila:</b> {i}<br>"
                f"<b>Estado:</b> {state[i]}<br>"
                f"<b>Densidad:</b> {density[i]:.4f}<br>"
                f"<b>Pendiente:</b> {slope[i]:.4f}<br>"
                f"<b>Curvatura:</b> {curvature[i]:.4f}"
                for i in range(len(xy))
            ]

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=xy[:, 0],
                        y=xy[:, 1] if xy.shape[1] > 1 else [0] * len(xy),
                        z=density,
                        mode="markers",
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        marker=dict(
                            size=5,
                            color=density,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Densidad"),
                        ),
                    )
                ]
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="Densidad Local",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=650,
            )

            st.caption(
                "Cada punto corresponde a una fila real del archivo cargado. "
                "No se interpola ninguna superficie ni se inventan observaciones."
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
