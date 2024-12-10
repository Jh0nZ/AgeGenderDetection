import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import sys

# Asegurar UTF-8 en salida estándar
sys.stdout.reconfigure(encoding='utf-8')

# Cargar modelo y escalador seleccionados
def cargar_modelo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    scales_dir = os.path.join(current_dir, '..', 'scales')

    # Mostrar diálogo para seleccionar modelo
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    modelo_path = filedialog.askopenfilename(
        initialdir=models_dir,
        title="Seleccionar Modelo",
        filetypes=(("Archivos Keras", "*.keras"), ("Todos los archivos", "*.*"))
    )
    
    if not modelo_path:
        print("No se seleccionó ningún modelo.")
        sys.exit()

    modelo_nombre = os.path.basename(modelo_path)
    scaler_path = os.path.join(scales_dir, f"{os.path.splitext(modelo_nombre)[0]}.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    print(f"Modelo seleccionado: {modelo_nombre}")
    return load_model(modelo_path), scaler

model, scaler = cargar_modelo()

# Cargar el clasificador de rostros de OpenCV
clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para predecir género y edad
def predecir_genero_edad(rostro):
    rostro = cv2.resize(rostro, (128, 128))
    rostro = img_to_array(rostro) / 255.0
    rostro = np.expand_dims(rostro, axis=0)
    genero_pred, edad_pred = model.predict(rostro)
    genero = "Mujer" if genero_pred[0] > 0.5 else "Hombre"
    edad = scaler.inverse_transform(edad_pred.reshape(-1, 1))[0][0] if scaler is not None else edad_pred[0][0]
    return genero, round(edad, 1)

# Capturar video desde la cámara
captura = cv2.VideoCapture(0)

if not captura.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    ret, frame = captura.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = clasificador_rostros.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro_color = frame[y:y + h, x:x + w]
        genero, edad = predecir_genero_edad(rostro_color)
        texto = f"{genero}, {edad} años"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Detección de Género y Edad', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()
