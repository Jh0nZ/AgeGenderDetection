import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import sys
import joblib

# Asegurarse de que la salida estándar use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Cargar el modelo de Keras
modelo_path = "../models/1000imagenes.keras"
scaler = joblib.load("../scales/1000imagenes.pkl")
model = load_model(modelo_path)
print(f"Modelo cargado desde: {modelo_path}")

# Cargar el clasificador de rostros de OpenCV
clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para predecir género y edad
def predecir_genero_edad(rostro):
    rostro = cv2.resize(rostro, (128, 128))  # Tamaño que espera el modelo
    rostro = img_to_array(rostro) / 255.0  # Normalizar
    rostro = np.expand_dims(rostro, axis=0)
    
    # Predicción
    genero_pred, edad_pred = model.predict(rostro)
    genero = "Mujer" if genero_pred[0] > 0.5 else "Hombre"
    edad = int(edad_pred[0][0])  # Convertir predicción de edad a entero
    return genero, edad

captura = cv2.VideoCapture(0)  

if not captura.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    ret, frame = captura.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises para la detección
    rostros = clasificador_rostros.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro_color = frame[y:y + h, x:x + w]
        
        # Predicción de género y edad
        genero, edad = predecir_genero_edad(rostro_color)
        age_pred = np.array([[edad]])
        edad = scaler.inverse_transform(age_pred.reshape(-1, 1))[0][0] if scaler is not None else age_pred[0][0]

        # Reemplazar caracteres no codificables (si hay) en el texto
        texto = f"{genero}, {edad} anios".encode('ascii', 'ignore').decode('ascii')

        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar resultados de género y edad en el rectángulo
        cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar el video con detecciones
    cv2.imshow('Detección de Género y Edad', frame)

    # Salir con la tecla 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()
