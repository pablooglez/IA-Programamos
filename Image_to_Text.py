from transformers import pipeline
from PIL import Image
import requests

def obtener_descripcion_imagen(url_imagen, token):
    try:
        # Inicializar el pipeline usando la API de Hugging Face
        captioner = pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-large",
            token=token,  # Tu token de Hugging Face
            device=-1,    # Usar API en lugar de GPU/CPU local
            use_auth_token=True
        )
        
        imagen = Image.open(requests.get(url_imagen, stream=True).raw)
        resultado = captioner(imagen)
        return resultado[0]['generated_text']
    
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

if __name__ == "__main__":
    # Configura tu token de Hugging Face
    HF_TOKEN = "tu_token_aquí"
    
    url_imagen = "https://cdn.pixabay.com/photo/2023/04/19/09/25/dog-7937282_1280.jpg"
    descripcion = obtener_descripcion_imagen(url_imagen, HF_TOKEN)
    print("Descripción de la imagen:", descripcion)