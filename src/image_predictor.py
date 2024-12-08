import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model_loader import get_model, get_scaler

def predict_new_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = get_model()
    if model is None:
        return "Seleccione un modelo primero"
    gender_pred, age_pred = model.predict(img_array)
    gender = "Mujer" if gender_pred[0] > 0.5 else "Hombre"
    scaler = get_scaler()
    age = scaler.inverse_transform(age_pred.reshape(-1, 1))[0][0] if scaler is not None else age_pred[0][0]
    print(f"raw predicción gender_pred: {gender_pred} age_pred: {age_pred}")
    print(f"Predicción género {gender}, edad {age}")
    
    return f"Género: {gender}, Edad: {int(age)}"