from PIL import Image
import base64

# Read the image file
image_path = "superimposed_image.png"
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Encode the image data in Base64
encoded_string = base64.b64encode(image_data).decode("utf-8")


# Decode the Base64 string back to image data (optional)
decoded_data = base64.b64decode(encoded_string)
with open("decoded_image.jpg", "wb") as image_file:
    image_file.write(decoded_data)