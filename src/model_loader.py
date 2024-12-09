import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

model = None
scaler = None

def load_selected_model(model_name):
    global model, scaler
    model_path = os.path.join("models", model_name)
    scaler_path = os.path.join("scales", model_name.replace('.keras', '.pkl'))

    model = load_model(
        model_path,
        custom_objects={
            "binary_crossentropy": BinaryCrossentropy(),
            "mse": MeanSquaredError()
        }
    )
    print(f"Modelo cargado: {model_name}")
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Escalador cargado: {scaler_path}")
    else:
        print(f"No se encontró el escalador en: {scaler_path}")

def get_model():
    return model

def get_scaler():
    return scaler

def adjust_combobox_width(combobox, values):
    # Determina el ancho máximo necesario para mostrar los valores
    max_length = max(len(str(value)) for value in values)
    # Ajusta el ancho del Combobox (aproximado a caracteres promedio)
    combobox.config(width=max_length)
