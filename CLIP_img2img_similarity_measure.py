from PIL import Image
import requests
from transformers import CLIPProcessor, FlaxCLIPModel

import jax.numpy as jnp
from jax import vmap, pmap, jit


# loading the preetrained model for HF
model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# loading CLIPProcessor to handle Tokineization and feature extraction of the images
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# loading two images for example purpose
ground_truth_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
ground_truth = Image.open(requests.get(ground_truth_url, stream=True).raw)     # in our project it'll come from dataset

predicton_url = "https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2020/02/322868_1100-800x825.jpg"
predicton = Image.open(requests.get(predicton_url, stream=True).raw)      # in our project it'll come from NeRF model's prediction

# processing the 2 images
processed_images = processor(images=[ground_truth, predicton], return_tensors="np")


# for optimation purpose (dunno if it makes difference?)
phi = jit(jnp.dot)

# Geting the 2 image's embedding values from Clip's ViT
img_embeddings = model.get_image_features(pixel_values=processed_images['pixel_values'])
img_embeddings = img_embeddings / jnp.linalg.norm(img_embeddings, axis=-1, keepdims=True)     # turn 'em into unit vectors

I, I_hat = img_embeddings   # destructuring 2 embeddings

# Finally, measure similarity with dot product
print(phi(I, I_hat))