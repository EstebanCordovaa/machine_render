# src/app_streamlit.py
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Clasificador de Iris", layout="centered")

# Título de la aplicación
st.title("Clasificador de Flores Iris")
st.markdown("Introduce las medidas de la flor para predecir su especie.")

# Carga el modelo entrenado
# Asegúrate de que el modelo esté en la ruta correcta desde la raíz del proyecto
try:
    model = joblib.load('models/iris_model.pkl')
except FileNotFoundError:
    st.error("Error: Modelo 'iris_model.pkl' no encontrado. Asegúrate de que esté en la carpeta 'models/'.")
    st.stop() # Detiene la ejecución de la app si el modelo no se encuentra

# Crear inputs para las características
st.header("Características de la Flor")
sepal_length = st.slider("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0, 0.1)
sepal_width = st.slider("Ancho del Sépalo (cm)", 0.0, 5.0, 3.0, 0.1)
petal_length = st.slider("Longitud del Pétalo (cm)", 0.0, 7.0, 4.0, 0.1)
petal_width = st.slider("Ancho del Pétalo (cm)", 0.0, 3.0, 1.0, 0.1)

# Botón para predecir
if st.button("Predecir Especie"):
    # Prepara las características para el modelo
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realiza la predicción
    pred_class = model.predict(features)[0]

    # Mapea el resultado numérico a un nombre de clase
    clases = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    prediction_name = clases.get(pred_class, "Desconocida")

    st.subheader("Resultado de la Predicción:")
    st.success(f"La especie de la flor Iris es: **{prediction_name}**")

st.markdown("---")
st.markdown("Desarrollado como parte del proyecto de Ciencia de Datos.")