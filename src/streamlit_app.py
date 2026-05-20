from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

from src.preprocessing import preprocess_face_array


def run_app() -> None:
    """Ejecuta la app visual completa para la fase de deployment."""

    st.set_page_config(page_title="Lab02 ML - Demo visual", layout="centered")
    st.title("Laboratorio 02: Clasificación de género y edad")

    # Cargar ambos modelos
    gender_model_path = Path("artifacts/models/pipeline_genero.pkl")
    age_model_path    = Path("artifacts/models/pipeline_edad.pkl")

    if not gender_model_path.exists() or not age_model_path.exists():
        st.error(
            "No se encontraron los modelos entrenados. "
            "Ejecuta primero main.py para generar los modelos."
        )
        st.stop()

    gender_model = joblib.load(gender_model_path)
    age_model    = joblib.load(age_model_path)

    st.success("Modelos cargados correctamente.")

    # Subir imagen
    uploaded_file = st.file_uploader(
        "Sube una fotografía",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.stop()

    # Leer imagen con OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detector de caras
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.warning("No se detectaron caras en la imagen.")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Imagen cargada sin detecciones",
            use_container_width=True
        )
        st.stop()

    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        # Mismo preprocesamiento que en entrenamiento
        x_vec, _ = preprocess_face_array(face)

        # Predicciones
        pred_gender = gender_model.predict([x_vec])[0]
        pred_age    = age_model.predict([x_vec])[0]

        genero = "Hombre" if pred_gender == 1 else "Mujer"
        label  = f"{genero} | {int(pred_age)} anios"

        # Dibujar rectangulo y etiqueta sobre la imagen
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Mostrar imagen con predicciones
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Resultado con predicciones",
        use_container_width=True
    )

    # Mostrar tabla de resultados
    st.subheader("Resultados por rostro detectado")
    for i, (x, y, w, h) in enumerate(faces):
        face  = image[y:y+h, x:x+w]
        x_vec, _ = preprocess_face_array(face)

        pred_gender = gender_model.predict([x_vec])[0]
        pred_age    = age_model.predict([x_vec])[0]

        genero = "Hombre" if pred_gender == 1 else "Mujer"

        st.write(f"**Rostro {i+1}:** {genero} | {int(pred_age)} años")