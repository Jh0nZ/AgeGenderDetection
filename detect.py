import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tkinter as tk
from tkinter import filedialog

# Cargar el modelo entrenado y el escalador
model = load_model(
    "gender_age_model_v1.h5",
    custom_objects={
        "binary_crossentropy": BinaryCrossentropy(),
        "mse": MeanSquaredError()
    }
)
scaler = joblib.load("scaler_v1.pkl")

# Configuración
img_size = (128, 128)  # el tamaño usado durante el entrenamiento

# Función para predecir género y edad
def predict_new_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    
    # Predecir
    gender_pred, age_pred = model.predict(img_array)
    gender = "Mujer" if gender_pred[0] > 0.5 else "Hombre"
    age = scaler.inverse_transform(age_pred)[0][0]
    print(f"Prediccion: Género: {gender_pred} Edad: {age_pred}")
    
    return gender, round(age)

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")],
        initialdir="./test"
        
    )
    return file_path

# Usar la función para predecir
new_image_path = open_file_dialog()  # Abrir el explorador de archivos
if new_image_path:
    image_name = os.path.basename(new_image_path)
    print(f"Imagen seleccionada: {image_name}")
    gender, age = predict_new_image(new_image_path)
    print(f"Género: {gender}, Edad: {age}")
else:
    print("No se seleccionó ninguna imagen.")