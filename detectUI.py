import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Configuración
img_size = (128, 128)  # el tamaño usado durante el entrenamiento
model = None
scaler = None  # Inicializar el escalador como None

# Función para cargar el modelo y el escalador
def load_selected_model(model_name):
    global model, scaler
    model_path = os.path.join("models", model_name)
    scaler_path = os.path.join("scales", model_name.replace('.keras', '.pkl'))  # Cambia la extensión a .pkl

    # Cargar el modelo
    model = load_model(
        model_path,
        custom_objects={
            "binary_crossentropy": BinaryCrossentropy(),
            "mse": MeanSquaredError()
        }
    )
    print(f"Modelo cargado: {model_name}")
    
    # Cargar el escalador
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Escalador cargado: {scaler_path}")
    else:
        print(f"No se encontró el escalador en: {scaler_path}")

# Función para predecir género y edad
def predict_new_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    
    # Predecir
    gender_pred, age_pred = model.predict(img_array)
    gender = "Mujer" if gender_pred[0] > 0.5 else "Hombre"
    age = scaler.inverse_transform(age_pred.reshape(-1, 1))[0][0] if scaler is not None else age_pred[0][0]
    print(f"Predicción gender_pred: {gender_pred} age_pred: {age_pred}")
    return gender, round(age)

# Función para mostrar la imagen seleccionada
def show_selected_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.LANCZOS)  # Redimensionar para la vista previa
    img_tk = ImageTk.PhotoImage(img)
    
    preview_label.config(image=img_tk)
    preview_label.image = img_tk  # Mantener una referencia a la imagen

    # Predecir y mostrar el resultado
    gender, age = predict_new_image(image_path)
    result_label.config(text=f"Género: {gender}, Edad: {age}")

# Función para cargar imágenes desde el directorio
def load_images():
    # Limpiar el frame de imágenes
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    # Listar imágenes en el directorio test
    test_images = os.listdir("./test")
    for i in range(0, len(test_images), 3):
        row_frame = tk.Frame(scrollable_frame)
        row_frame.pack(side=tk.TOP, fill=tk.X)
        
        for j in range(3):
            if i + j < len(test_images):
                img_name = test_images[i + j]
                img_path = os.path.join("test", img_name)
                img = Image.open(img_path)
                img = img.resize((100, 100), Image.LANCZOS)  # Redimensionar para la vista previa
                img_tk = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(row_frame, image=img_tk)
                img_label.image = img_tk  # Mantener una referencia a la imagen
                img_label.pack(side=tk.LEFT, padx=5, pady=5)
                img_label.bind("<Button-1>", lambda e, path=img_path: show_selected_image(path))


def adjust_combobox_width(combobox, values):
    # Determina el ancho máximo necesario para mostrar los valores
    max_length = max(len(str(value)) for value in values)
    # Ajusta el ancho del Combobox (aproximado a caracteres promedio)
    combobox.config(width=max_length)


# Crear la ventana principal
root = tk.Tk()
root.title("Predicción de Género y Edad")

# Frame para mostrar imágenes con scroll
frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT, padx=10, pady=10)

# Canvas para el scroll
canvas = tk.Canvas(frame_left)
scrollbar = tk.Scrollbar(frame_left, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

# Configurar el scroll
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Empaquetar el canvas y el scrollbar
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Permitir desplazamiento con la rueda del mouse
def on_mouse_wheel(event):
    canvas.yview_scroll(-1 * (event.delta // 120), "units")  # Ajustar el desplazamiento

canvas.bind_all("<MouseWheel>", on_mouse_wheel)

# Frame para la vista previa y resultados
frame_right = tk.Frame(root)
frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

# Selector de modelos
modelos = os.listdir("models")
model_selector = ttk.Combobox(frame_right, values=os.listdir("models"))
model_selector.pack(pady=10)
adjust_combobox_width(model_selector, modelos)
model_selector.bind("<<ComboboxSelected>>", lambda e: load_selected_model(model_selector.get()))

# Label para la vista previa de la imagen seleccionada
preview_label = tk.Label(frame_right)
preview_label.pack(pady=10)

# Label para mostrar el resultado
result_label = tk.Label(frame_right, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Botón para recargar imágenes
reload_button = tk.Button(frame_right, text="Recargar Imágenes", command=load_images)
reload_button.pack(pady=10)

# Cargar imágenes inicialmente
load_images()

# Iniciar el bucle principal
root.mainloop()