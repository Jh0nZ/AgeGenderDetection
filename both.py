import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configuración inicial
image_dir = "./images"
img_size = (128, 128)  # Tamaño al que redimensionaremos las imágenes
batch_size = 32

# Leer imágenes y etiquetas
def load_data(image_dir):
    images = []
    genders = []  # 0: Hombre, 1: Mujer
    ages = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            # Parsear edad y género desde el nombre del archivo
            parts = filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])  # 0 o 1
            
            # Cargar y preprocesar la imagen
            img_path = os.path.join(image_dir, filename)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalizar entre 0 y 1
            
            # Agregar a las listas
            images.append(img_array)
            genders.append(gender)
            ages.append(age)
    
    return np.array(images), np.array(genders), np.array(ages)

# Cargar datos
images, genders, ages = load_data(image_dir)

# Normalizar edades
scaler = MinMaxScaler()
ages = scaler.fit_transform(ages.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, gender_train, gender_val, age_train, age_val = train_test_split(
    images, genders, ages, test_size=0.2, random_state=42
)

# Crear el modelo
def create_model():
    input_layer = Input(shape=(img_size[0], img_size[1], 3))
    
    # Capas convolucionales
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Aplanar y capas densas
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Salida para género (clasificación)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    
    # Salida para edad (regresión)
    age_output = Dense(1, activation='linear', name='age_output')(x)
    
    # Modelo
    model = Model(inputs=input_layer, outputs=[gender_output, age_output])
    return model

model = create_model()
model.summary()

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss={
        'gender_output': 'binary_crossentropy',
        'age_output': 'mse'
    },
    metrics={
        'gender_output': 'accuracy',
        'age_output': 'mae'
    }
)

# Entrenar el modelo
history = model.fit(
    X_train, {'gender_output': gender_train, 'age_output': age_train},
    validation_data=(X_val, {'gender_output': gender_val, 'age_output': age_val}),
    batch_size=batch_size,
    epochs=20
)

# Guardar el modelo entrenado
model.save("gender_age_model.h5")

# Predecir con nuevas imágenes
def predict_new_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    
    gender_pred, age_pred = model.predict(img_array)
    gender = "Mujer" if gender_pred[0] > 0.5 else "Hombre"
    age = scaler.inverse_transform(age_pred)[0][0]
    
    return gender, round(age)

# Ejemplo de uso
new_image_path = "ruta_a_nueva_imagen.jpg"  # Cambia esto a la imagen que quieras probar
gender, age = predict_new_image(new_image_path)
print(f"Género: {gender}, Edad: {age}")