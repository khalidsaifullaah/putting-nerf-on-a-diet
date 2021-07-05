import jax
import jax.numpy as np
from jax import jit, random



from transformers import FlaxCLIPModel

N_samples = 128
test_inner_steps = 64

CLIP_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
K = 16
l = 1 # loss weight lambda, but the exact value is not indicated in the paper.
hwf = None # defined in dataset

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype = np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype = np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype = np.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def random_poses(rng, bds):
    radius = random.uniform(rng, minval = bds[0], maxval = bds[1])
    theta = random.uniform(rng, minval = 0, maxval = 2*np.pi)
    phi = random.uniform(rng, minval = 0, maxval = np.pi/2)
    return np.stack([pose_spherical(r,t,p) for r, t, p in zip(radius, theta, phi)])

# def random_poses(rng, bds):
#     radius = random.uniform(rng, shape = [batch_size], minval = bds[0], maxval = bds[1])
#     theta = random.uniform(rng, shape = [batch_size], minval = 0, maxval = 2*np.pi)
#     phi = random.uniform(rng, shape = [batch_size], minval = 0, maxval = np.pi/2)
#     return np.stack([pose_spherical(r,t,p) for r, t, p in zip(radius, theta, phi)])

def CLIPProcessor(image):
    '''
        jax-based preprocessing for CLIP

        image  [B, 3, H, W]: batch image
        return [B, 3, 224, 224]: pre-processed image for CLIP
    '''
    B,D,H,W = image.shape
    image = jax.image.resize(image, (B,D,224,224), 'bicubic') # assume that images have rectangle shape. 
    image = (image - np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,3,1,1)) / np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,3,1,1)
    return image

@jit
def SC_loss(target_emb, source_img, l):
    '''
        target_emb [B, D]: pre-computed target embedding vector \phi(I)
        source_img [B, 3, H, W]: source image \hat{I}
        l: loss weight lambda
        return: SC_loss
    '''
    source_img = CLIPProcessor(source_img)
    source_emb = CLIP_model.get_image_features(pixel_values=source_img)
    source_emb /= np.linalg.norm(source_emb, axis=-1, keepdims=True)
    return l * (target_emb * source_emb).sum()/source_emb.shape[0]

@jit
def single_step(rng, image, rays, params, bds, step, target_emb):
    '''
        step [] : current step
        target_emb [B, D]: pre-computed target embedding vector \phi(I)

        TODO:
        implement random pose
    '''
    def sgd(param, update):
        return param - inner_step_size * update

    rng, rng_inputs = jax.random.split(rng)

    def loss_model(params):
        g = render_rays(rng_inputs, model, params, None, rays, bds[0], bds[1], N_samples, rand=True)
        L = mse_fn(g, image)

        if step%K == 0:
            # these lines are not confirmed
            rays = get_ray(hwf[0], hwf[1], hwf[2], random_pose(rng_inputs))
            rays = np.reshape(rays, (2,-1,3))
            source_img = np.clip(render_rays(rng_inputs, model, params, None, rays, bds[0], bds[1], N_samples, rand=False), 0, 1)
            L += SC_loss(target_emb, source_img, l) # add Semantic Consistency loss
        return L

    model_loss, grad = jax.value_and_grad(loss_model)(params)
    new_params = jax.tree_multimap(sgd, params, grad)
    return rng, new_params, model_loss

'''
# libraries for checking SC_loss
from PIL import Image
import requests

# in our project target_emb'll be pre-computed and fed by dataloader
target_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
target_img = np.array(Image.open(requests.get(target_url, stream=True).raw))
target_img = np.expand_dims(target_img.transpose(2,0,1), 0)/255
target_img = CLIPProcessor(target_img)
target_emb = model.get_image_features(pixel_values=target_img)
target_emb /= np.linalg.norm(target_emb, axis=-1, keepdims=True)

# in our project source image'll come from NeRF model's prediction
source_img_url = "https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2020/02/322868_1100-800x825.jpg"
source_img = np.array(Image.open(requests.get(source_img_url, stream=True).raw))/255
source_img = np.expand_dims(source_img.transpose(2,1,0), 0)

print(SC_loss(target_emb, source_img, 1))
'''