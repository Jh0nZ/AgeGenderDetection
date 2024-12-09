import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from model_loader import get_model, get_scaler

# Cargar modelo y escalador
model = get_model()
scaler = get_scaler()
if model is None:
    print("Modelo no cargado. Asegúrate de seleccionar un modelo válido.")
    exit()

# Cargar el clasificador de rostros de OpenCV
clasificadorRostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predecir_genero_edad(rostro):
    """Preprocesa el rostro y realiza la predicción de género y edad."""
    rostro = cv2.resize(rostro, (128, 128))  # Redimensionar al tamaño que espera el modelo
    rostro = img_to_array(rostro) / 255.0  # Normalizar
    rostro = np.expand_dims(rostro, axis=0)  # Añadir dimensión para batch
    gender_pred, age_pred = model.predict(rostro)

    # Interpretar resultados
    gender = "Mujer" if gender_pred[0] > 0.5 else "Hombre"
    age = scaler.inverse_transform(age_pred.reshape(-1, 1))[0][0] if scaler is not None else age_pred[0][0]
    return gender, int(age)

# Capturar video en tiempo real desde la cámara
captura = cv2.VideoCapture(0)  # Usa 0 para la cámara integrada
if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

print("Presiona 'Esc' para salir.")

while True:
    ret, frame = captura.read()
    if not ret:
        print("No se pudo leer el cuadro de la cámara.")
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = clasificadorRostros.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in rostros:
        rostro = frame[y:y + h, x:x + w]
        gender, age = predecir_genero_edad(rostro)

        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar texto con género y edad
        texto = f"{gender}, {age} años"
        cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar el cuadro con los resultados
    cv2.imshow('Detección de Género y Edad', frame)

    # Salir al presionar 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

captura.release()
cv2.destroyAllWindows()
