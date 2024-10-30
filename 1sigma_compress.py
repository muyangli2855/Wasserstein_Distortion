'''
    @author: Yang QIU

    this code is developed based on @Md Sarfarazul Haque's impelementation
    https://github.com/mdsarfarazulh/deep-texture-synthesis-cnn-keras
    of the paper Texture Synthesis Using Convolutional Neural Networks by
    Gatys et al.
'''


import numpy as np 
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
import keras.utils as image
import math
import sys
import os
import time

import pickle

from vgg19_n import VGG19  # The customed VGG19 network is based on fchollet implementation of VGG19
                    # from https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py
                    # and the normalized weights, as suggested in Gaty et al.'s paper, is translated
                    # from https://github.com/paulu/deepfeatinterp/blob/master/models/VGG_CNN_19/vgg_normalised.caffemodel
from keras import backend as K 
import tensorflow as tf
from tensorflow.python.ops import math_ops
tf.compat.v1.disable_eager_execution()
# tf.random.set_seed(7)
# np.random.seed(7)



class DeepTexture(object):

    def __init__(self, tex_path, base_img_path=None):
        '''
            tex_path: Path to the source image
            gen_prefix: Prefix associated with the output image's names
            base_img_path: Either None or path to an image; if None, the reconstruction is
                initiated as a random noise; otherwise initiated using the given image
        '''
        # initial parameters
        self.width, self.height = image.load_img(tex_path).size
        self.loss_value = None
        self.grad_values = None
        self.channels = 3 # 3 for rgb, 1 for grayscale
        if base_img_path == None:
            x = np.random.rand(self.width, self.height, self.channels)
            x = np.expand_dims(x, axis=0)
            self.base_img = x[:,:,:,::-1].astype(np.float32) # 'RGB' -> 'BGR'
        else:
            self.base_img = self.preprocess_image(base_img_path) # If base_img_path is not `None`
                                                                 # then use that image as base image
        self.tex_path = tex_path
        if K.image_data_format() == 'channels_last':
            self.input_shape = (1, self.height, self.width, self.channels)
        else:
            self.input_shape = (1, self.channels, self.height, self.width)
        filename = os.path.basename(tex_path)
        filename = os.path.splitext(filename)[0]
        self.filename = filename


    def preprocess_image(self, img_path):
        '''
            pre-process image: 'RGB' -> 'BGR'
        '''
        img = image.load_img(img_path, target_size=(self.width, self.height))
        img = image.img_to_array(img)
        img = img[:,:,::-1] # RGB to BGR
        img = np.expand_dims(img, axis=0)
        return img


    def deprocess_image(self, x):
        '''
            de-process: 'BGR' -> 'RGB', then map floats to integers between 0 and 255.
        '''
        if K.image_data_format() == 'channels_first':
            x = x.reshape((self.channels, self.height, self.width))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.height, self.width, self.channels))
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    def get_tsg(self,size_list):
        '''
            generates the a 3-D array for two-sided geoemetric distribution, first two dimension corresponds to 
            the probability mass for the corresponding pixel for corresp. spatial dimension, and last dimension
            corresponds to one particular pixel of interest
        '''
        sigma=4000
        tsg_list_sizes = {}
        for size in size_list:
            current_height = size[0]
            current_width = size[1]
            current_height_half = int(current_height/2)
            current_width_half = int(current_width/2)
            tsg_current = np.array([[np.exp(-((i_ref_p-0)**2+(j_ref_p-0)**2)/(2*sigma**2))\
                                    for j_ref_p in range(-current_width_half,current_width_half)]\
                                    for i_ref_p in range(-current_height_half,current_height_half)])
            tsg_current = tsg_current/np.sum(tsg_current)
            tsg_current = np.expand_dims(tsg_current,axis=2)
            tsg_current_tf = tf.convert_to_tensor(tsg_current,dtype=tf.float32)
            tsg_list_sizes[current_height] = tsg_current_tf
            print('current size is ' + str(size))
        return tsg_list_sizes

    
    def gauss_wass_dist(self,tex,gen,tsg_list_sizes):
        '''
            calculate the distortion for output of each layer
        '''
        current_height = tex.shape[0]
        current_width = tex.shape[1]
        tsg_current = tsg_list_sizes[current_height]
        mean_tex = tf.reduce_sum(tsg_current*tex,axis=[0,1])
        mean_gen = tf.reduce_sum(tsg_current*gen,axis=[0,1])
        std_tex = tf.math.sqrt(tf.nn.relu(tf.reduce_sum((tex**2)*tsg_current,axis=[0,1]) -\
                                        tf.math.square(mean_tex))+1e-7)
        std_gen = tf.math.sqrt(tf.nn.relu(tf.reduce_sum((gen**2)*tsg_current,axis=[0,1]) -\
                                        tf.math.square(mean_gen))+1e-7)
        return tf.reduce_sum((mean_tex-mean_gen)**2 + (std_tex - std_gen)**2)

    def gauss_wass_dist_feature(self,tex,tsg_list_sizes):
        current_height = tex.shape[0]
        current_width = tex.shape[1]
        tsg_current = tsg_list_sizes[current_height]
        mean_tex = tf.reduce_sum(tsg_current*tex,axis=[0,1])
        std_tex = tf.math.sqrt(tf.nn.relu(tf.reduce_sum((tex**2)*tsg_current,axis=[0,1]) -\
                                        tf.math.square(mean_tex))+1e-7)
        return mean_tex,std_tex
        

    def eval_loss_and_grads(self, x):
        '''
            calculates the total loss and the total gradient of total loss with respect to the 
            synthesised image
        '''
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.height, self.width))
        else:
            x = x.reshape((1, self.height, self.width, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


    def get_loss(self, x):
        '''
            helper function to help optimizer to get loss function
        '''
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return loss_value


    def get_grads(self, x):
        '''
            helper function to help optimizer to get gradient values
        '''
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def quantize(self,x,level):
        maxi = tf.reduce_max(x)
        mini = tf.reduce_min(x)
        # x_quantized = tf.quantization.fake_quant_with_min_max_vars(x,min=mini,max=maxi,\
        #     num_bits=level)
        x_quantized = tf.quantization.quantize_and_dequantize_v2(x,input_min=mini,input_max=maxi,\
            num_bits=level,range_given=False)
        return x_quantized
    
    def buildTexture(self, features, level, iterations):
        '''
            main function

            features: list of layers of interest; presets to choose from or name own list
            iterations: iterations of the program
        '''
        tex_img = K.variable(self.preprocess_image(self.tex_path))
        gen_img = K.placeholder(shape=self.input_shape)
        model_tex = VGG19(include_top=False, input_tensor=tex_img, weights='normalized')
        model_gen = VGG19(include_top=False, input_tensor=gen_img, weights='normalized')
        outputs_dict_tex = dict([(layer.name, layer.output) for layer in model_tex.layers])
        outputs_dict_gen = dict([(layer.name, layer.output) for layer in model_gen.layers])
        loss = K.variable(0.)
        flag = True
        all_layers = [layer.name for layer in model_tex.layers[1:]]

        if features == 'all':
            feature_layers = all_layers
        elif features == 'pool':
            feature_layers = ['block1_pool', 'block2_pool','block3_pool','block4_pool','block5_pool']
        elif features == 'eachlayer':
            feature_layers = ['block1_conv2','block2_conv2','block3_conv4','block4_conv4',\
                                'block5_conv4']
        elif features == 'test':
            feature_layers = ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',\
                                'block3_conv1','block3_conv4','block4_conv1','block4_conv4',\
                                'block5_conv1','block5_conv4']
        elif features == '4nopool':
            feature_layers = ['block1_conv1','block1_conv2','block1_pool','block2_conv1',\
                                'block2_conv2','block2_pool','block3_conv1','block3_conv2',\
                                'block3_conv3','block3_conv4','block3_pool','block4_conv1',\
                                'block4_conv2','block4_conv3','block4_conv4']
        elif isinstance(features, (list, tuple)):
            for f in features:
                if f not in all_layers:
                    flag = False
            if flag:
                feature_layers = features
            else:
                raise ValueError('`features` should be either `all` or `pool` or a set of names from\
                 layer.name from model.layers')
        else:
            raise ValueError('`features` should be either `all` or `pool` or a set of names from\
             layer.name from model.layers')
            
        total_shape = [self.input_shape]
        for layer in feature_layers:
            shape = model_tex.get_layer(layer).output_shape
            total_shape.append(shape)
        total_shape = np.array(total_shape)
        size_list = np.unique(total_shape[:,1:3],axis=0)
        total_layer_no = len(feature_layers)
        
        start = time.time()
        tsg_list_sizes = self.get_tsg(size_list)
        end = time.time()
        print('get two sided geometric list doen with %.2f secs' % (end - start))

        tex_pixels = tf.squeeze(tex_img)/255.
        mean_tex,std_tex = self.gauss_wass_dist_feature(tex_pixels,tsg_list_sizes)
        features_tex = [[mean_tex,std_tex]]
        for i_layer,layer_name in enumerate(feature_layers):
            layer_features = outputs_dict_tex[layer_name]
            tex_features = layer_features[0,:,:,:]
            mean_tex,std_tex = self.gauss_wass_dist_feature(tex_features,tsg_list_sizes)
            features_tex.append([mean_tex,std_tex])
        features_tex = tf.transpose(tf.concat(features_tex,axis=1))
        features_tex_quantized = self.quantize(features_tex,level)

        gen_pixels = tf.squeeze(gen_img)/255.
        mean_gen,std_gen = self.gauss_wass_dist_feature(gen_pixels,tsg_list_sizes)
        features_gen = [[mean_gen,std_gen]]
        for i_layer,layer_name in enumerate(feature_layers):
            layer_features = outputs_dict_gen[layer_name]
            gen_features = layer_features[0, :, :, :]
            mean_gen,std_gen = self.gauss_wass_dist_feature(gen_features,tsg_list_sizes)
            features_gen.append([mean_gen,std_gen])
        features_gen = tf.transpose(tf.concat(features_gen,axis=1))

        total_features = features_gen.shape[0]
        layer_weights = tf.concat([tf.ones([1,3])*200.,
            tf.ones([1,int(total_features/3)])*50.,
            tf.ones([1,int(total_features/3)])*10.,
            tf.ones([1,total_features-2*int(total_features/3)-3])*1.],axis=1)
        
        loss = tf.reduce_sum(tf.reduce_sum((features_gen - features_tex_quantized)**2,\
            axis=1)*layer_weights)

        grads = tf.gradients(loss, gen_img)
        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)
        self.f_outputs = K.function([gen_img], outputs)
        x = self.base_img

        if not os.path.exists('output'):
            os.makedirs('output')

        total_start_time = time.time()
        min_val_list = []
        for i in range(iterations):
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(func=self.get_loss, x0=x.flatten(),\
                                                fprime=self.get_grads, maxfun=20)
            min_val_list.append(min_val)

            if math.isnan(min_val):
                print('Loss is NaN!!')
                sys.exit()
            elif i <= 30:
                continue
            elif min_val >= np.min(min_val_list[-31:-1]):
                print('loss didn\'t drop after 30 iterations')
                print('Current loss value:', min_val)
                img = self.deprocess_image(x.copy())
                np.save('./output/reconstruction.npy',img)
                fname = './output/' + self.filename + '_level_{}_iter_{}.png'.format(level,(i+1))
                im = Image.fromarray(img)
                im.save(fname)
                end_time = time.time()
                print('Image saved as', fname)
                print('Iteration %d completed in %.2f secs' % ((i+1), end_time - start_time))
                break
            elif (i+1)%50 == 0:
                print('Current loss value:', min_val)
                img = self.deprocess_image(x.copy())
                np.save('./output/reconstruction.npy',img)
                fname = './output/' + self.filename + '_level_{}_iter_{}.png'.format(level,(i+1))
                im = Image.fromarray(img)
                im.save(fname)
                end_time = time.time()
                end_time = time.time()
                print('Image saved as', fname)
                print('Iteration %d completed in %.2f secs' % ((i+1), end_time - start_time))
            else:
                continue
        
        np.save('./output/loss.npy',np.array(min_val_list))
        total_end_time = time.time()
        print('All iterations completed in %.2f secs' % (total_end_time - total_start_time))


tex = DeepTexture('./textures/'+sys.argv[1]+'.jpg')
tex.buildTexture(features='test',level=int(sys.argv[2]),iterations=4000)
