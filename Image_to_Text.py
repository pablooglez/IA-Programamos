from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load the model and processor
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Load an example image
image_url = "https://cdn.pixabay.com/photo/2023/04/19/09/25/dog-7937282_1280.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Preprocess the image
text_prompt = "Describe image:"
inputs = processor(images=image, text=text_prompt, return_tensors="pt")

# Generate caption
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption in Spanish:", caption)