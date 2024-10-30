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

from vgg19_n import VGG19 # The customed VGG19 network is based on fchollet implementation of VGG19
                    # from https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py
                    # and the normalized weights, as suggested in Gaty et al.'s paper, is translated
                    # from https://github.com/paulu/deepfeatinterp/blob/master/models/VGG_CNN_19/vgg_normalised.caffemodel    
from keras import backend as K 
import tensorflow as tf
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
        self.height, self.width = image.load_img(tex_path).size
        self.loss_value = None
        self.grad_values = None
        self.channels = 3 # 3 for rgb, 1 for grayscale
        if base_img_path == None:
            x = np.random.rand(self.height, self.width, self.channels)
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
        img = image.load_img(img_path, target_size=(self.height, self.width))
        img = image.img_to_array(img)
        img = img[:,:,::-1]
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
        tsg_list_sizes = {}
        sigma0_list = {}
        for size in size_list:
            current_height = size[0]
            current_width = size[1]
            height_half = int(current_height/2)
            width_half = int(current_width/2)
            height_tenth = int(current_height/8)
            width_tenth = int(current_width/8)
            print('current size is ' + str(size))
            sigma_map = np.ones([current_height,current_width])
            sigma_map[height_half-height_tenth:height_half+height_tenth+1,\
                        width_half-width_tenth:width_half+width_tenth+1] = 0
            sigma0_pixels = np.squeeze(np.transpose(np.expand_dims(np.where(sigma_map == 0),axis=1)))
            sigma0_list[current_height] = sigma0_pixels
            rest = np.squeeze(np.transpose(np.expand_dims(np.where(sigma_map == 1),axis=1)))
            lmbda = 1.5 # proportionality for linear growth of sigma
            for pixel in rest:
                dist = np.min(np.sum(np.abs(pixel-sigma0_pixels),axis=1))
                sigma_map[pixel[0],pixel[1]] = dist*lmbda
            tsg_list = []
            for i in range(20):
                '''
                    generate 20 sets of random pixels of interest; in distortion calculation, 
                    use one randomly for each layer/each iteration
                '''
                nonzero_eval_pixels_indx = np.random.choice(rest.shape[0],25,replace=False) # 25 random pixels of interest
                pixels_of_interest = rest[nonzero_eval_pixels_indx]
                tsg_list_temp = []
                for pixel in pixels_of_interest:
                    [i_height,i_width] = pixel
                    sigma = sigma_map[i_height,i_width]
                    if sigma == 0:
                        raise Exception('Sigma here should be strictly positive')
                    else:
                        tsg_temp = np.array([[((np.exp(1/sigma)-1)/(np.exp(1/sigma)+1))*\
                                    np.exp(-(np.abs(i_ref_p)+np.abs(j_ref_p))/sigma)\
                                    for j_ref_p in range(-i_width+1,current_width-i_width+1)]\
                                    for i_ref_p in range(-i_height+1,current_height-i_height+1)])
                        tsg_temp = tsg_temp/np.sum(tsg_temp)
                        tsg_temp = np.expand_dims(tsg_temp,axis=2)
                        tsg_list_temp.append(tsg_temp)
                tsg_list_temp = np.expand_dims(np.concatenate(tsg_list_temp,axis=2),axis=2)
                tsg_list_temp_tf = tf.convert_to_tensor(tsg_list_temp,dtype=tf.float32)
                tsg_list.append(tsg_list_temp)
            tsg_list_sizes[current_height] = tsg_list
        return tsg_list_sizes,sigma0_list

    
    def gauss_wass_dist(self,tex,gen,tsg_list_sizes,sigma0_list):
        '''
            calculate the distortion for output of each layer
        '''
        rdm_indx = np.random.randint(low=0,high=20) # choose which random set of pixels of interest to use
        current_height = tex.shape[0]
        current_width = tex.shape[1]
        tsg_list = tsg_list_sizes[current_height]
        tsg = tsg_list[rdm_indx]
        sigma0_pixels = sigma0_list[current_height]
        mean_tex = tf.reduce_sum(tsg*tex,axis=[0,1])
        mean_gen = tf.reduce_sum(tsg*gen,axis=[0,1])
        std_tex = tf.math.sqrt(tf.nn.relu(tf.reduce_sum((tex**2)*tsg,axis=[0,1]) -\
                                        tf.math.square(mean_tex))+1e-7)
        std_gen = tf.math.sqrt(tf.nn.relu(tf.reduce_sum((gen**2)*tsg,axis=[0,1]) -\
                                        tf.math.square(mean_gen))+1e-7)
        tex_sigma0 = tf.gather_nd(tex,sigma0_pixels)
        gen_sigma0 = tf.gather_nd(gen,sigma0_pixels)
        dist_sigma0 = tf.reduce_sum((tex_sigma0-gen_sigma0)**2)
        return tf.reduce_sum((mean_tex-mean_gen)**2 + (std_tex-std_gen)**2)*200 + dist_sigma0
        

    def eval_loss_and_grads(self, x):
        '''
            calculates the total loss and the total gradient of total loss with respect to the 
            synthesised image
        '''
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, self.channels, self.height, self.width))
        else:
            x = x.reshape((1, self.height, self.width, self.channels))
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
        # Getting loss and grad values.
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
        

    def buildTexture(self, features, iterations):
        '''
            main function

            features: list of layers of interest; presets to choose from or name own list
            iterations: iterations of the program
        '''
        tex_img = K.variable(self.preprocess_image(self.tex_path))
        gen_img = K.placeholder(shape=self.input_shape)
        input_tensor = K.concatenate([tex_img, gen_img], axis=0)
        model = VGG19(include_top=False, input_tensor=input_tensor, weights='normalized')
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        loss = K.variable(0.)
        flag = True
        all_layers = [layer.name for layer in model.layers[1:]]
        if features == 'all':
            feature_layers = all_layers
        elif features == 'pool':
            feature_layers = ['block1_pool', 'block2_pool','block3_pool','block4_pool','block5_pool']
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
        
        total_shape = []
        for layer in feature_layers:
            shape = model.get_layer(layer).output_shape
            total_shape.append(shape)
        total_shape = np.array(total_shape[1:])
        size_list = np.unique(total_shape[:,1:3],axis=0)
        total_layer_no = len(feature_layers)
        
        start = time.time()
        tsg_list_sizes,sigma0_list = self.get_tsg(size_list)
        end = time.time()
        print('get two sided geometric list doen with %.2f secs' % (end - start))

        for i_layer,layer_name in enumerate(feature_layers):
            layer_features = outputs_dict[layer_name]
            tex_features = layer_features[0, :, :, :]
            gen_features = layer_features[1, :, :, :]
            tex_features = tf.expand_dims(tex_features, axis=3)
            gen_features = tf.expand_dims(gen_features, axis=3)

            layer_loss = self.gauss_wass_dist(tex_features, gen_features,tsg_list_sizes,sigma0_list)

            if (i_layer+1) <= int(total_layer_no/3):
                layer_weight = 10
            elif (i_layer+1) <= int(2*total_layer_no/3):
                layer_weight = 5
            else:
                layer_weight = 1

            loss = loss + layer_loss*layer_weight
            
        tex_pixels = tf.expand_dims(tf.squeeze(tex_img)/255.,axis=3)
        gen_pixels = tf.expand_dims(tf.squeeze(gen_img)/255.,axis=3)
        loss = loss + self.gauss_wass_dist(tex_pixels,gen_pixels,tsg_list_sizes,sigma0_list)*100

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
                fname = './output/' + self.filename + '_reproduction_at_iteration_%d.png' % (i+1)
                im = Image.fromarray(img)
                im.save(fname)
                end_time = time.time()
                print('Image saved as', fname)
                print('Iteration %d completed in %.2f secs' % ((i+1), end_time - start_time))
                break
            elif (i+1)%50 == 0:
                print('Current loss value:', min_val)
                img = self.deprocess_image(x.copy())
                fname = './output/' + self.filename + '_reproduction_at_iteration_%d.png' % (i+1)
                im = Image.fromarray(img)
                im.save(fname)
                end_time = time.time()
                print('Image saved as', fname)
                print('Iteration %d completed in %.2f secs' % ((i+1), end_time - start_time))
            else:
                continue
        
        np.save('./output/loss.npy',np.array(min_val_list))
        total_end_time = time.time()
        print('All iterations completed in %.2f secs' % (total_end_time - total_start_time))


tex = DeepTexture(tex_path='./textures/bark.jpg')
tex.buildTexture(features='4nopool',iterations=4000)