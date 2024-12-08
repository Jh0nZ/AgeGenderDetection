import tkinter as tk
from tkinter import ttk
from gui import create_gui

def main():
    root = tk.Tk()
    root.title("Predicción de Género y Edad")
    create_gui(root)
    root.mainloop()

if __name__ == "__main__":
    main()