# Loading the image method
# Embedding image method
# Embedding text method
# Load model
# Embedding loop
# Ensure the code includes proper comments.

# the .py file should contains code with this order:
# model_img = ...
# embeddig_image function
# embedding loop

# then, similar for text:
# model_text = ...
# embeddig_text function
# embedding loop for text
# and nothing else


from torchvision import models, transforms
import numpy as np
from PIL import Image
import requests
import torch

model_img = models.resnet50(pretrained=True)
model_img.eval()

# Define preprocessing for images
# Normalizing according documentation 
# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

def load_image_from_url(url: str) -> np.ndarray:
    """Loads an image from a given URL and converts it to an RGB NumPy array.

    Args:
        url: The URL of the image to load.

    Returns:
        A NumPy array representing the image in RGB format.
    """

    response = requests.get(url, stream=True)
    image = Image.open(response.raw).convert('RGB')

    return np.array(image)

def encode_image(image: str) -> np.ndarray:
    """Encodes a given image URL into an embedding vector.

    Args:
        image:  The URL of the image to encode.

    Returns:
         A NumPy array representing the image embedding.
    """

    # Load and preprocess the image
    raw_image = load_image_from_url(image)
    processed_image = image_transform(Image.fromarray(raw_image)).unsqueeze(0)
    
    # Generate image embedding
    with torch.no_grad():
        embedding = model_img(processed_image)

    return embedding.squeeze().numpy()

import tqdm
from sklearn.decomposition import PCA

image_data = []


# Assuming 'urls' is a list contains url for a single image
for url in tqdm(urls, desc="Encoding images"):
    try:
        embedding = encode_image(url)
        image_data.append(embedding)
    except Exception as e:
        print(f"Error encoding image at {url}: {e}")

image_data_array = np.array(image_data)

# Reducing dimensionality using sklern PCA 
pca = PCA(n_components=384)
image_data = pca.fit_transform(image_data_array)


# -------------------Text embeddings-------------------------
from sentence_transformers import SentenceTransformer
from typing import List

model_text = SentenceTransformer('all-MiniLM-L6-v2')

def encode_text(captions: List[List[str]]) -> List[np.ndarray]:
    """
    Encodes a list of captions (one list per image) into a list of text embeddings.

    Args:
      captions: A list of lists, where each inner list contains captions for a single image.

    Returns:
      A list of NumPy arrays, where each array represents embedding for an image's captions.
    """

    text_embeddings = []
    
    for caption_list in tqdm(captions, desc="Encoding captions"):

        caption_embeddings = model_text.encode(caption_list)
        aggregated_embedding = np.mean(caption_embeddings, axis=0)
        text_embeddings.append(aggregated_embedding)
    
    return text_embeddings

# Assuming 'sentences' is a list of lists where each inner list contains captions for a single image
text_data = encode_text(sentences)
