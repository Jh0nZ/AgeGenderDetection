import tkinter as tk
from tkinter import ttk
from image_loader import load_images
from model_loader import load_selected_model, adjust_combobox_width
import os

def create_gui(root):
    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, padx=10, pady=10)

    canvas = tk.Canvas(frame_left)
    scrollbar = tk.Scrollbar(frame_left, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

    # Agregar un label que diga "Modelo"
    model_label = tk.Label(frame_right, text="Modelo:")
    model_label.pack(pady=(10, 0))  # Espaciado superior, sin espaciado inferior

    modelos = os.listdir("models")
    model_selector = ttk.Combobox(frame_right, values=modelos)
    model_selector.pack(pady=10)
    adjust_combobox_width(model_selector, modelos)
    model_selector.bind("<<ComboboxSelected>>", lambda e: load_selected_model(model_selector.get()))

    preview_label = tk.Label(frame_right)
    preview_label.pack(pady=10)

    result_label = tk.Label(frame_right, text="", font=("Arial", 14))
    result_label.pack(pady=10)

    reload_button = tk.Button(frame_right, text="Recargar Imágenes", command=lambda: load_images(scrollable_frame, preview_label, result_label))
    reload_button.pack(pady=10)
    
    # Permitir desplazamiento con la rueda del mouse
    def on_mouse_wheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")  # Ajustar el desplazamiento

    scrollbar.bind_all("<MouseWheel>", on_mouse_wheel)

    load_images(scrollable_frame, preview_label, result_label)