import tkinter as tk
from tkinter import ttk
from gui import create_gui
import os

def seleccionar_modelo(root):
    # Obtener la lista de modelos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    modelos = [archivo for archivo in os.listdir(models_dir) if archivo.endswith('.keras')]
    modelo_seleccionado = tk.StringVar(root)
    modelo_seleccionado.set(modelos[0]) 
    ventana = tk.Toplevel(root)
    ventana.title("Seleccionar Modelo")
    ttk.Label(ventana, text="Selecciona un modelo:").pack(pady=10)
    dropdown = ttk.OptionMenu(ventana, modelo_seleccionado, *modelos)
    dropdown.pack(pady=10)
    def aceptar_seleccion():
        ventana.destroy()
        create_gui(root, modelo_seleccionado.get())

    ttk.Button(ventana, text="Aceptar", command=aceptar_seleccion).pack(pady=10)

def main():
    root = tk.Tk()
    root.title("Predicción de Género y Edad")
    seleccionar_modelo(root)
    root.mainloop()

if __name__ == "__main__":
    main()
