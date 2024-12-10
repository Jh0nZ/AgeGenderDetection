import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Función para cargar datos
def load_data(image_dir):
    log_message(f"Cargando imágenes desde: {image_dir}")
    images = []
    genders = []  # 0: Hombre, 1: Mujer
    ages = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    age = int(parts[0])  # Edad
                    gender = int(parts[1])  # Género (0 o 1)
                    if gender not in [0, 1]:
                        log_message(f"Archivo con género inválido: {filename}. Se espera 0 o 1.")
                        continue
                    img_path = os.path.join(image_dir, filename)
                    img = load_img(img_path, target_size=(128, 128))
                    img_array = img_to_array(img) / 255.0  # Normalizar entre 0 y 1
                    
                    images.append(img_array)
                    genders.append(gender)
                    ages.append(age)
                except ValueError:
                    log_message(f"Error en el archivo: {filename}. El formato de edad o género es inválido.")
            else:
                log_message(f"Nombre de archivo incorrecto: {filename}. Debe seguir el formato 'edad_género.jpg'.")
    
    return np.array(images), np.array(genders), np.array(ages)

# Función para registrar mensajes en la consola
def log_message(message):
    console_text.config(state=tk.NORMAL)  # Habilitar el widget de texto
    console_text.insert(tk.END, message + "\n")  # Insertar el mensaje
    console_text.see(tk.END)  # Desplazar hacia abajo
    console_text.config(state=tk.DISABLED)  # Deshabilitar el widget de texto

# Crear el modelo
def create_model():
    input_layer = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    age_output = Dense(1, activation='linear', name='age_output')(x)
    model = Model(inputs=input_layer, outputs=[gender_output, age_output])
    return model

# Función para entrenar el modelo
def train_model():
    image_dir = dir_entry.get().strip()  # Eliminar espacios en blanco
    log_message(f"Directorio ingresado: {image_dir}")  # Para depuración
    if not os.path.exists(image_dir):
        messagebox.showerror("Error", "El directorio de imágenes no existe.")
        return

    images, genders, ages = load_data(image_dir)
    scaler = MinMaxScaler()
    ages = scaler.fit_transform(ages.reshape(-1, 1))

    X_train, X_val, gender_train, gender_val, age_train, age_val = train_test_split(
        images, genders, ages, test_size=0.5, random_state=42
    )

    model = create_model()
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
        batch_size=int(batch_size_entry.get()),
        epochs=int(epochs_entry.get())
    )

    save_trained_model(model, scaler)

# Función para guardar el modelo
def save_trained_model(model, scaler):
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("scales"):
        os.makedirs("scales")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"gender_age_model_{timestamp}.keras" 
    model_path = os.path.join('models', model_name)
    scale_name = f"gender_age_model_{timestamp}.pkl"
    scale_path = os.path.join("scales", scale_name)
    
    joblib.dump(scaler, scale_path)
    model.save(model_path)
    messagebox.showinfo("Éxito", f"Modelo guardado en: {model_path}\nEscalador guardado en: {scale_path}")

# Crear la ventana principal
root = tk.Tk()
root.title("Entrenamiento de Modelo de Género y Edad")
root.geometry("400x400")

# Entrada para el directorio de imágenes
tk.Label(root, text="Directorio de Imágenes:").pack(pady=5)
dir_entry = tk.Entry(root, width=40)
dir_entry.pack(pady=5)
tk.Button(root, text="Seleccionar Directorio", command=lambda: dir_entry.insert(0, filedialog.askdirectory())).pack(pady=5)

# Entrada para el tamaño del batch
tk.Label(root, text="Tamaño del Batch:").pack(pady=5)
batch_size_entry = tk.Entry(root)
batch_size_entry.insert(0, "32")  # Valor por defecto
batch_size_entry.pack(pady=5)

# Entrada para el número de épocas
tk.Label(root, text="Número de Épocas:").pack(pady=5)
epochs_entry = tk.Entry(root)
epochs_entry.insert(0, "16")  # Valor por defecto
epochs_entry.pack(pady=5)

# Botón para iniciar el entrenamiento
train_button = tk.Button(root, text="Entrenar Modelo", command=train_model)
train_button.pack(pady=20)

# Consola desplazable para mostrar mensajes
console_frame = tk.Frame(root)
console_frame.pack(pady=5)

console_text = tk.Text(console_frame, height=10, width=50, state=tk.DISABLED)
console_text.pack(side=tk.LEFT)

scrollbar = tk.Scrollbar(console_frame, command=console_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

console_text.config(yscrollcommand=scrollbar.set)

# Ejecutar la aplicación
root.mainloop()