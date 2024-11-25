import tensorflow as tf

print("Dispositivos disponibles para TensorFlow:")
print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))
