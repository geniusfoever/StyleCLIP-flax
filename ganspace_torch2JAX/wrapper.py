import jax
from jax import random
from pathlib import Path
from types import SimpleNamespace
from utils import download_ckpt
from config import Config
import flax
import dill as pickle
import builtins
from jax._src.lib import xla_client
import tensorflow as tf
import argparse
import math
import os
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.example_libraries.optimizers import adam
from functools import partial
from flax.training.common_utils import shard
from tqdm import tqdm
from PIL import Image
from functools import partial
from flax.training.common_utils import shard
import transformers
# from . import biggan
# from . import stylegan
# from . import stylegan2_jax as stylegan2_jax
from abc import abstractmethod, ABC as AbstractBaseClass
from functools import singledispatch
import numpy as onp

class BaseModel(AbstractBaseClass):

    # Set parameters for identifying model from instance
    def __init__(self, model_name, class_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.outclass = class_name

    # Stop model evaluation as soon as possible after
    # given layer has been executed, used to speed up
    # netdissect.InstrumentedModel::retain_layer().
    # Validate with tests/partial_forward_test.py
    # Can use forward() as fallback at the cost of performance.
    @abstractmethod
    def partial_forward(self, x, layer_name):
        pass

    # Generate batch of latent vectors
    @abstractmethod
    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        pass

    # Maximum number of latents that can be provided
    # Typically one for each layer
    def get_max_latents(self):
        return 1

    # Name of primary latent space
    # E.g. StyleGAN can alternatively use W
    def latent_space_name(self):
        return 'Z'

    def get_latent_shape(self):
        return tuple(self.sample_latent(1).shape)

    def get_latent_dims(self):
        return jnp.prod(self.get_latent_shape())

    def set_output_class(self, new_class):
        self.outclass = new_class

    # Map from typical range [-1, 1] to [0, 1]
    def forward(self, x):
        out = self.model.forward(x)
        return 0.5*(out+1)

    # Generate images and convert to numpy
    def sample_np(self, z=None, n_samples=1, seed=None):
        if z is None:
            z = self.sample_latent(n_samples, seed=seed)
        elif isinstance(z, list):
            z = [jnp.array(l) if not isinstance(l, jnp.ndarray) else l for l in z]
        elif not isinstance(z, jnp.ndarray):
            z = jnp.array(z)
        img = self.forward(z)
        img_np = jnp.transpose(img, (0, 2, 3, 1))
        return jnp.clip(img_np, 0.0, 1.0).squeeze()

    # For models that use part of latent as conditioning
    def get_conditional_state(self, z):
        return None

    # For models that use part of latent as conditioning
    def set_conditional_state(self, z, c):
        return z

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

# JAX port of StyleGAN 2
class StyleGAN2(BaseModel):
    def __init__(self, device, class_name, seed,truncation=1.0, use_w=False):
        super(StyleGAN2, self).__init__('StyleGAN2', class_name or 'ffhq')
        self.device = device
        self.truncation = truncation
        self.truncation_psi=0.7
        self.latent_avg = None
        self.set_noise_seed(0)
        seed=100 if seed is None else seed
        self.seed=jax.random.PRNGKey(seed)
        self.w_primary= use_w # use W as primary latent space?
            # Image widths
        configs = {
            # Converted NVIDIA official
            'ffhq': 1024,
            'car': 512,
            'cat': 256,
            'church': 256,
            'horse': 256,
            # Tuomas
            'bedrooms': 256,
            'kitchen': 256,
            'places': 256,
        }

        assert self.outclass in configs, \
            f'Invalid StyleGAN2 class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = configs[self.outclass]
        self.name = f'StyleGAN2-{self.outclass}'
        self.has_latent_residual = True
        self.load_model()

    def latent_space_name(self):
        return 'W' if self.w_primary else 'Z'

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def download_checkpoint(self, outfile):
        pass
    def load_model(self):
        # Similar to the PyTorch version, but load the JAX StyleGAN2 model instead
        builtins.bfloat16 = xla_client.bfloat16
        CheckPointsFolder="/home/beich/majesty/project/stylegan2-flax-tpu/checkpoints/ckpt_392000_best.pickle"
        if not os.path.isfile(CheckPointsFolder):
            raise FileNotFoundError(f'Could not find StyleGAN2 checkpoint at {CheckPointsFolder}')

        def pickle_load(filename):
            """ Wrapper to load an object from a file."""
            with tf.io.gfile.GFile(filename, 'rb') as f:
                pickled = pickle.loads(f.read())
            return pickled
        # model=pickle_load("/home/beich/majesty/project/models/food-512.pkl")
        # model=pickle_load("/home/beich/majesty/project/WISE1-40000_best.pickle")
        model=pickle_load(CheckPointsFolder)

        G=model['state_G']
        self.model=G
        self.latent_avg = self.mean_w_values=self.generator_mean_latent(10000,self.seed)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = jnp.random.randint(jnp.iinfo(jnp.int32).max) # use (reproducible) global rand state

        rng = jnp.random.RandomState(self.seed)

        z = rng.standard_normal(512 * n_samples).reshape(n_samples, 512)
        if self.w_primary:
            z =  self.generator_mean_latent(10000,self.seed)
        return onp.array(z)
    def generator_mean_latent(self, n_latent, rng):
        generator=G=self.model
        latent_in = random.normal(rng, (n_latent, 512))

        latent = generator.apply_mapping({'params': G.params['mapping'], 'moving_stats': G.moving_stats},latent_in,c=jnp.array([[0,0,0,0,1],[0,0,0,0,1]]),train=False).mean(0, keepdims=True)
        return latent
    def get_max_latents(self):
        return 15

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError('StyleGAN2: cannot change output class without reloading')

    def forward(self, x):
        if self.w_primary:
            w=x
        else:
            w = self.model.apply_mapping({'params': self.model.params['mapping'], 'moving_stats': self.model.moving_stats},x,train=False)
        out=self.model.apply_synthesis({'params': self.model.params['synthesis'], 'moving_stats': self.model.moving_stats,'noise_consts': self.model.noise_consts},self.truncation_psi * w + (1 - self.truncation_psi) * self.latent_avg , noise_mode='none')
        # x = x if isinstance(x, list) else [x]
        # out, _ = self.model(x, noise=self.noise,
        #     truncation=self.truncation, truncation_latent=self.latent_avg, input_is_w=self.w_primary)
        return 0.5*(out+1)
    def partial_forward(self, x, layer_name):
        if layer_name=='style':
            return self.model.apply_mapping({'params': self.model.params['mapping'], 'moving_stats': self.model.moving_stats},x,train=False)
        self.forward(x)
        # styles = x if isinstance(x, list) else [x]
        # inject_index = None
        # noise = self.noise

        # if not self.w_primary:
        #     styles = [self.model.style(s) for s in styles]

        # if len(styles) == 1:
        #     # One global latent
        #     inject_index = self.model.n_latent
        #     latent = self.model.strided_style(styles[0].unsqueeze(1).repeat(1, inject_index, 1)) # [N, 18, 512]
        # elif len(styles) == 2:
        #     # Latent mixing with two latents
        #     if inject_index is None:
        #         inject_index = random.randint(1, self.model.n_latent - 1)

        #     latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
        #     latent2 = styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)

        #     latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        # else:
        #     # One latent per layer
        #     assert len(styles) == self.model.n_latent, f'Expected {self.model.n_latents} latents, got {len(styles)}'
        #     styles = torch.stack(styles, dim=1) # [N, 18, 512]
        #     latent = self.model.strided_style(styles)

        # if 'style' in layer_name:
        #     return

        # out = self.model.input(latent)
        # if 'input' == layer_name:
        #     return

        # out = self.model.conv1(out, latent[:, 0], noise=noise[0])
        # if 'conv1' in layer_name:
        #     return

        # skip = self.model.to_rgb1(out, latent[:, 1])
        # if 'to_rgb1' in layer_name:
        #     return

        # i = 1
        # noise_i = 1

        # for conv1, conv2, to_rgb in zip(
        #     self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs
        # ):
        #     out = conv1(out, latent[:, i], noise=noise[noise_i])
        #     if f'convs.{i-1}' in layer_name:
        #         return

        #     out = conv2(out, latent[:, i + 1], noise=noise[noise_i + 1])
        #     if f'convs.{i}' in layer_name:
        #         return
            
        #     skip = to_rgb(out, latent[:, i + 2], skip)
        #     if f'to_rgbs.{i//2}' in layer_name:
        #         return

        #     i += 2
        #     noise_i += 2

        # image = skip

        # raise RuntimeError(f'Layer {layer_name} not encountered in partial_forward')
    def set_noise_seed(self, seed):
        random.seed(seed)  # Use JAX random instead of torch
        self.noise = [random.normal(key, (1, 1, 2 ** 2, 2 ** 2)) for key in random.split(random.PRNGKey(seed), self.model.log_size * 2)]

        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(random.normal(random.PRNGKey(seed), (1, 1, 2 ** i, 2 ** i)))
    def set_noise_seed(self, seed):
        random.seed(seed) # Use JAX random instead of torch
        self.noise = [random.normal(key, (1, 1, 2 ** 2, 2 ** 2)) for key in random.split(random.PRNGKey(seed), self.model.log_size * 2)]
        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(random.normal(random.PRNGKey(seed), (1, 1, 2 ** i, 2 ** i)))


# Version 1: separate parameters
@singledispatch
def get_model(name, output_class, device, **kwargs):
    # Check if optionally provided existing model can be reused
    inst = kwargs.get('inst', None)
    model = kwargs.get('model', None)
    
    if inst or model:
        cached = model or inst.model
        
        network_same = (cached.model_name == name)
        outclass_same = (cached.outclass == output_class)
        can_change_class = ('BigGAN' in name)
        
        if network_same and (outclass_same or can_change_class):
            cached.set_output_class(output_class)
            return cached
    
    if name == 'DCGAN':
        import warnings
        warnings.filterwarnings("ignore", message="nn.functional.tanh is deprecated")
        model = GANZooModel(device, 'DCGAN')
    elif name == 'ProGAN':
        model = ProGAN(device, output_class)
    elif 'BigGAN' in name:
        assert '-' in name, 'Please specify BigGAN resolution, e.g. BigGAN-512'
        model = BigGAN(device, name.split('-')[-1], class_name=output_class)
    elif name == 'StyleGAN':
        model = StyleGAN(device, class_name=output_class)
    elif name == 'StyleGAN2':
        model = StyleGAN2(device, class_name=output_class)
    else:
        raise RuntimeError(f'Unknown model {name}')

    return model

# Version 2: Config object
@get_model.register(Config)
def _(cfg, device, **kwargs):
    kwargs['use_w'] = kwargs.get('use_w', cfg.use_w) # explicit arg can override cfg
    return get_model(cfg.model, cfg.output_class, device, **kwargs)

# Version 1: separate parameters
@singledispatch
def get_instrumented_model(name, output_class, layers, device, **kwargs):
    model = get_model(name, output_class, device, **kwargs)
    model.eval()

    inst = kwargs.get('inst', None)
    if inst:
        inst.close()

    if not isinstance(layers, list):
        layers = [layers]

    # Verify given layer names
    module_names = [name for (name, _) in model.named_modules()]
    for layer_name in layers:
        if not layer_name in module_names:
            print(f"Layer '{layer_name}' not found in model!")
            print("Available layers:", '\n'.join(module_names))
            raise RuntimeError(f"Unknown layer '{layer_name}''")
    
    # Reset StyleGANs to z mode for shape annotation
    if hasattr(model, 'use_z'):
        model.use_z()

    from netdissect.modelconfig import create_instrumented_model
    inst = create_instrumented_model(SimpleNamespace(
        model = model,
        layers = layers,
        cuda = device.type == 'cuda',
        gen = True,
        latent_shape = model.get_latent_shape()
    ))

    if kwargs.get('use_w', False):
        model.use_w()

    return inst

# Version 2: Config object
@get_instrumented_model.register(Config)
def _(cfg, device, **kwargs):
    kwargs['use_w'] = kwargs.get('use_w', cfg.use_w) # explicit arg can override cfg
    return get_instrumented_model(cfg.model, cfg.output_class, cfg.layer, device, **kwargs)