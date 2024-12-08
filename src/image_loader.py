import os
import tkinter as tk
from PIL import Image, ImageTk
from image_predictor import predict_new_image

def load_images(scrollable_frame, preview_label, result_label):
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    test_images = os.listdir("./test")
    for i in range(0, len(test_images), 3):
        row_frame = tk.Frame(scrollable_frame)
        row_frame.pack(side=tk.TOP, fill=tk.X)
        
        for j in range(3):
            if i + j < len(test_images):
                img_name = test_images[i + j]
                img_path = os.path.join("test", img_name)
                img = Image.open(img_path)
                img = img.resize((100, 100), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)

                img_label = tk.Label(row_frame, image=img_tk)
                img_label.image = img_tk
                img_label.pack(side=tk.LEFT, padx=5, pady=5)
                img_label.bind("<Button-1>", lambda e, path=img_path: show_selected_image(path, preview_label, result_label))

def show_selected_image(image_path, preview_label, result_label):
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    
    preview_label.config(image=img_tk)
    preview_label.image = img_tk

    resultado = predict_new_image(image_path)
    result_label.config(text=resultado)