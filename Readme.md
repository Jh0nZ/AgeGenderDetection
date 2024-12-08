# Predicción de Género y Edad

Este proyecto utiliza un modelo de aprendizaje profundo para predecir el género y la edad de las personas a partir de imágenes. La aplicación está construida con Python y utiliza TensorFlow para el modelo de predicción.

## Requisitos

- Python 3.12.2
- Git

## Clonar el Repositorio

Para comenzar, clona el repositorio en tu máquina local:

```bash
git clone https://github.com/Jh0nZ/AgeGenderDetection.git
cd AgeGenderDetection
```

## Configuración del Entorno
1. Crear un entorno virtual:

```bash
python -m venv venv
```

2. Activar el entorno virtual:

```bash
venv\Scripts\activate
```
3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```
## Formato de las Imágenes
Las imágenes de entrenamiento deben seguir el siguiente formato de nombre:

```bash
edad_genero_*.jpg
```
- `genero`: 0 para hombre, 1 para mujer.
- `edad`: La edad de la persona en la imagen.

## Entrenamiento del Modelo
Para entrenar el modelo, asegúrate de tener tus imágenes de entrenamiento en el directorio `images/`. Luego, ejecuta el siguiente comando:

```bash
python train.py
```
Esto generará los modelos entrenados y los escaladores en los directorios `models/` y `scales/`, respectivamente.

## Ejecutar la Aplicación
Una vez que hayas entrenado el modelo, puedes ejecutar la aplicación con el siguiente comando:

```bash
python src/app.py
```

## Uso de la Aplicación
1. Selecciona un modelo entrenado desde el menú desplegable.
2. Haz clic en "Recargar Imágenes" para cargar las imágenes de prueba desde el directorio test/.
3. Haz clic en una imagen para predecir el género y la edad.
