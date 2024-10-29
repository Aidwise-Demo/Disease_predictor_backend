import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap(model, img_array):
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Get the last convolutional layer
    last_conv_layer = model.get_layer('conv2d_3')  # Replace with the actual layer name
    # Create a model that maps the input image to the activations of the last conv layer
    heatmap_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    # Get the gradients of the last conv layer with respect to the model output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute the mean intensity of the gradient over each feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is" with regard to the predicted class
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap[0]

async def predict_class_with_heatmap(path):
    img = cv2.imread(path)

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    # plt.imshow(RGBImg)
    
    image = np.array(RGBImg) / 255.0
    new_model = tf.keras.models.load_model("DRAPI/DR-CNN.model")
    
    # Generate heatmap
    heatmap = generate_heatmap(new_model, image)
    
    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (RGBImg.shape[1], RGBImg.shape[0]))

    # Superimpose the heatmap on the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image with a specified intensity
    superimposed_img = cv2.addWeighted(RGBImg, 0.6, heatmap, 0.4, 0)
    
    # Display the original image, heatmap, and superimposed image
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(RGBImg)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.subplot(133)
    plt.imshow(superimposed_img)
    plt.title('Superimposed Image')
    plt.savefig('DRAPI/superimposed_image.png')
    # plt.show()

    # Make a prediction
    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)
    if per == 1:
        return 'Diabetic Retinopathy Not Detected'
    else:
        return 'Diabetic Retinopathy Detected'
