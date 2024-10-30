# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper functions for PSNR computations."""

import jax
import jax.numpy as jnp
import h5py
import flax
import flax.linen as nn
import os
import pickle
import h5py

# =================================================
# discrete gauss pmf

# @jax.jit
# def get_size_list(x):
#   print(jnp.shape(x))
#   H,W,Ch = jnp.shape(x)
#   size_list = [[int(H),int(W)]]
#   assert H >= 16
#   assert W >= 16
#   for i in range(4):
#     size_list.append([int(H/(2**(i+1))),int(W/(2**(i+1)))])
#   return size_list

if os.path.exists('pmf.p'):

  pmf_list = pickle.load(open('pmf.p','rb'))

else:

  size_list = [[512,512],[256,256],[128,128],[64,64],[32,32]]

  def get_pmf(size_list): # x not needed now as input
    sigma = 4000
    # size_list = get_size_list(x)
    pmf_list = {}
    for [H,W] in size_list:
      H_half = int(H/2)
      W_half = int(W/2)
      pmf = jnp.array([[(1/(2*sigma**2))*jnp.exp(-((i_ref_p-0)**2+(j_ref_p-0)**2)/2*sigma**2)\
              for j_ref_p in range(-W_half,W_half)]\
              for i_ref_p in range(-H_half,H_half)])
      pmf = jnp.expand_dims(pmf/jnp.sum(pmf),axis=0)
      pmf_list[H] = pmf
    return pmf_list

  pmf_list = get_pmf(size_list)
  pickle.dump(pmf_list,open('pmf.p','wb'))

# =================================================
# VGG structure

# VGG_SHIFT = (1.0 + jnp.array([-.030, -.088, -.188])) / 2.0
# VGG_SCALE = jnp.array([.458, .448, .450]) / 2.0

# class VggBlock(flax.linen.Module):
#   """One block within the VGG network."""
#   num_features: int
#   num_layers: int

#   @flax.linen.compact
#   def __call__(self, x):
#     for _ in range(self.num_layers):
#       x = flax.linen.Conv(
#           features=self.num_features,
#           kernel_size=(3, 3),
#           padding="same",
#       )(x)
#       x = flax.linen.relu(x)
#     return x

# class VggNet(flax.linen.Module):
#   """VGG network which returns intermediate activations."""

#   @flax.linen.compact
#   def __call__(self, x):
#     assert x.shape[-2] >= 16, str(x.shape)
#     assert x.shape[-3] >= 16, str(x.shape)
#     outputs = [x]
#     for params in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
#       x = VggBlock(*params)(x)
#       outputs.append(x)
#       x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     return outputs

# def read_params(filename=None):
#   params = VggNet().init(jax.random.PRNGKey(0), jnp.zeros([1, 224, 224, 3]))
#   if filename is None:
#     filename = "/home/chiu/Desktop/vgg19_norm_weights.bin"
#   with gfile.Open(filename, "rb") as f:
#     return flax.serialization.from_bytes(params, f.read())
# # weights = load_weights.load_weights(h5py.File('vgg19_norm_weights.h5','r'))

class VGG(nn.Module):

  def setup(self):
    self.param_dict = h5py.File('vgg19_norm_weights.h5', 'r')
    self.dtype = 'float32'

  @nn.compact
  def __call__(self, x, train=False):

    mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(x.dtype)
    std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(x.dtype)
    x = (x - mean) / std

    act = [x]

    x,act = self._conv_block(x, features=64, num_layers=2, block_num=1, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=128, num_layers=2, block_num=2, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=256, num_layers=4, block_num=3, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=512, num_layers=4, block_num=4, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=512, num_layers=4, block_num=5, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    return act

  def _conv_block(self, x, features, num_layers, block_num, act, dtype='float32'):
    for l in range(num_layers):
      layer_name = f'block{block_num}_conv{l + 1}'
      w = lambda *_ : jnp.array(self.param_dict[layer_name][layer_name]['kernel:0']) 
      b = lambda *_ : jnp.array(self.param_dict[layer_name][layer_name]['bias:0']) 
      x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=w, bias_init=b,\
        padding='same', name=layer_name, dtype=dtype)(x)
      act.append(x)
      x = nn.relu(x)
    return x,act

# =================================================
# wasserstein distortion

def compute_features(image):
  # size_list = get_size_list(image)
  # image = jnp.transpose(image, (1, 2, 0))[None]
  image = image[None]
  # pmf_list = get_pmf(size_list)
  # image = (image - VGG_SHIFT) / VGG_SCALE
  vgg19 = VGG()
  init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
  params = vgg19.init(init_rngs, image)
  features = vgg19.apply(params, image)
  return [jnp.squeeze(f, 0).transpose((2, 0, 1)) for f in features] #,pmf_list

def compute_stats(features,pmf_list):
  means = []
  variances = []
  for f in features:
    H = jnp.shape(f)[1]
    pmf = pmf_list[H]
    squared = jnp.square(f)
    m = jnp.sum(f*pmf,axis=(1,2))
    p = jnp.sum(squared*pmf,axis=(1,2))
    v = p - jnp.square(m)
    means.append(m)
    variances.append(v)
  return means,variances

def wasserstein_distortion(features_a, features_b,pmf_list):
  means_a, variances_a = compute_stats(features_a, pmf_list)
  means_b, variances_b = compute_stats(features_b, pmf_list)
  wd_maps = []
  # wd_maps = [jnp.square(features_a - features_b)]
  assert len(means_a) == len(means_b) == len(variances_a) == len(variances_b)
  for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
    sa = jnp.sqrt(va + 1e-4)
    sb = jnp.sqrt(vb + 1e-4)
    wd_maps.append(jnp.square(ma - mb) + jnp.square(sa - sb))
  # assert len(wd_maps) == levels + 1
  dist = 0.
  # stuff = defaultdict(list, wd_maps=wd_maps)
  for i, wd_map in enumerate(wd_maps):
    # weight = jax.nn.relu(1 - abs(log_sigma - i))
    weight = 10**(2 - int(i/5))
    # stuff["weights"].append(weight)
    dist += jnp.sum(weight * wd_map)
  return dist #, stuff

def loss_fn(a,b):
  features_a = compute_features(a)
  features_b = compute_features(b)
  l = wasserstein_distortion(features_a,features_b,pmf_list)
  return l

# =================================================
# psnr

psnr_fn = lambda mse: -10 * jnp.log10(mse)
psnr_fn_jitted = jax.jit(psnr_fn)