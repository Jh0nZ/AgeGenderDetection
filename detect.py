import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

# Cargar el modelo entrenado y el escalador
model = load_model(
    "gender_age_model.h5",
    custom_objects={
        "binary_crossentropy": BinaryCrossentropy(),
        "mse": MeanSquaredError()
    }
)
scaler = joblib.load("scaler.pkl")

# Configuración
img_size = (128, 128)  # Debe coincidir con el tamaño usado durante el entrenamiento

# Función para predecir género y edad
def predict_new_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    
    # Predecir
    gender_pred, age_pred = model.predict(img_array)
    gender = "Mujer" if gender_pred[0] > 0.5 else "Hombre"
    age = scaler.inverse_transform(age_pred)[0][0]
    
    return gender, round(age)

# Usar la función para predecir
new_image_path = "girl.jpg"  # Cambia esto a la imagen que quieras probar
gender, age = predict_new_image(new_image_path)
print(f"Género: {gender}, Edad: {age}")
