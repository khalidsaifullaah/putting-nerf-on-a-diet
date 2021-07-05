import jax
import jax.numpy as np
from jax import jit, random


from transformers import FlaxCLIPModel

batch_size = 256
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

def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, step = step), np.arange(H, step = step), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(
        c2w[:3,-1], rays_d.shape)
    return np.stack([rays_o, rays_d], 0)

def random_pose(rng, bds):
    radius = random.uniform(rng, minval = bds[0], maxval = bds[1])
    theta = random.uniform(rng, minval = 0, maxval = 2*np.pi)
    phi = random.uniform(rng, minval = 0, maxval = np.pi/2)
    return pose_spherical(radius,theta,phi)

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

def SC_loss(rng_inputs, model, params, target_img, target_emb, rays, l):
    '''
        target_emb [B, D]: pre-computed target embedding vector \phi(I)
        source_img [B, 3, H, W]: source image \hat{I}
        l: loss weight lambda
        return: SC_loss
    '''
    source_img = np.clip(render_rays(rng_inputs[1], model, params, rays, N_samples, rand=False), 0, 1)
    source_img = np.reshape(source_img, [1,64,64,3]).transpose(0,3,1,2)
    source_img = CLIPProcessor(source_img)
    source_emb = CLIP_model.get_image_features(pixel_values=source_img)
    source_emb /= np.linalg.norm(source_emb, axis=-1, keepdims=True)
    return l/2 * ((target_emb - source_emb)**2).sum()/source_emb.shape[0]

# update inner model network weights with once step of sgd
@jit
def single_step(rng, images, rays, params, target_emb, step):
    def sgd(param, update):
        return param - inner_step_size * update

    def loss_fn(params, rng_inputs):
        pixels = np.reshape(images, [-1,3])
        pixel_rays = np.reshape(rays, (2,-1,3))
        idx = random.randint(rng_inputs[0], shape=(batch_size,), minval=0, maxval=pixels.shape[0])
        pixel_sub = pixels[idx,:]
        pixel_ray_sub = pixel_rays[:,idx,:]

        idx = random.randint(rng_inputs[0], shape=(1,), minval=0, maxval=images.shape[0])
        image_sub = images[idx]
        target_emb_sub = target_emb[idx,:] 
        ray_sub = np.reshape(rays[:,idx][:,:,::2,::2], [2,-1,3])

        
        g = render_rays(rng_inputs[1], model, params, pixel_ray_sub, N_samples, rand=True)
        L = mse_fn(g, pixel_sub)

        L = jax.lax.cond(step%K == 0,
            lambda _: L + SC_loss(rng_inputs, model, params, image_sub, target_emb_sub, ray_sub, 1),
            lambda _: L, 
            operand=None
        )
        return L

    rng, *rng_inputs = jax.random.split(rng, 3)
    loss, grad = jax.value_and_grad(loss_fn)(params, rng_inputs)
    params = jax.tree_multimap(sgd, params, grad)
    return rng, params, loss

# update inner model network weights inner_update_steps number of times
def update_network_weights(rng, images, rays, params, target_emb, step):
    for _ in range(inner_update_steps):
        rng, params, loss = single_step(rng, images, rays, params, target_emb, step)
    return rng, params, loss

# update meta model weights based on trained inner model weights
def update_model(rng, params, images, rays, target_emb, step):
    rng, new_params, model_loss = update_network_weights(rng, images, rays, params, target_emb, step)

'''
rng = jax.random.PRNGKey(0)
num_test = 5
rng, rng_input = jax.random.split(rng)
train_images, test_images = np.split(images, [images.shape[0]-num_test], axis=0)
train_poses, test_poses = np.split(poses, [images.shape[0]-num_test], axis=0)

target_images = CLIPProcessor(train_images.transpose(0,3,1,2))
target_emb = CLIP_model.get_image_features(pixel_values=target_images)
target_emb /= np.linalg.norm(target_emb, axis=-1, keepdims=True)

for step in tqdm(range(2)):
    rays = get_ray_batch(hwf[0], hwf[1], hwf[2], train_poses)
    update_model(rng, params, train_images, rays, target_emb, step)
'''