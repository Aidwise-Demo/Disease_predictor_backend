import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('covidmodel')

# Display the names of all layers
for layer in model.layers:
    print(layer.name)
