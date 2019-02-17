import tensorflow as tf
import numpy as np


initializer = tf.initializers.random_normal(0, 0.001)    

def encoder(net, latent_size, use_tanh):
    for j in range(9):
        curr_channels = min(latent_size, 32 * 2**(j//2))
        for i in range(2):
            net = tf.layers.conv2d(net, curr_channels, 3, padding='same',
                                   kernel_initializer = initializer,
                                   activation = tf.nn.leaky_relu)
        if j == 8 and use_tanh:
            net = tf.layers.conv2d(net, curr_channels, 3, strides=(2,2), 
                                   kernel_initializer = initializer, padding='same',
                                   activation = tf.nn.tanh)
        else:
            net = tf.layers.conv2d(net, curr_channels, 3, strides=(2,2), 
                                   kernel_initializer = initializer, padding='same',
                                   activation = tf.nn.leaky_relu)

    return net

def fix_dims(t):
    return tf.expand_dims(tf.expand_dims(t, 1), 1)

def adain(x, y, c, b):
    #xb, xw, xh, xc = tf.shape(x)
    y = tf.reshape(tf.layers.dense(y, c*2, tf.nn.leaky_relu), (b, c, 2))
    mu_x, var_x = tf.nn.moments(x, axes=[1,2], keep_dims=True)
    std_x = tf.sqrt(var_x)
    mu_y = fix_dims(y[:,:,0])
    std_y = fix_dims(y[:,:,1])
    return std_y*(x - mu_x)/std_x + mu_y


    


def decoder(latent_space, batch_size, latent_size):
    init_channels = 512
    conv_blocks = 8
    latent_space = tf.reshape(latent_space, (-1, latent_size))

    init_val = np.random.normal(size=(4, 4, init_channels)).astype(np.float32)
    base_image = tf.Variable(init_val, dtype=tf.float32)
    dec = tf.stack([base_image] * batch_size)

    for i in range(conv_blocks):
        curr_channels = int(init_channels / 2 ** i)
        if i != 0:
            dec = tf.layers.conv2d_transpose(dec, curr_channels, 3, strides = (2,2), 
                                             kernel_initializer = initializer,
                                             activation = tf.nn.leaky_relu, padding='same')
            dec = tf.layers.conv2d(dec, curr_channels, 3, kernel_initializer = initializer, 
                                   activation = tf.nn.leaky_relu, padding='same')
            
        dec = dec + tf.random.normal(tf.shape(dec), 0.0, 0.01)
        dec = adain(dec, latent_space, curr_channels, batch_size)
        dec = tf.layers.conv2d(dec, curr_channels, 3, kernel_initializer = initializer, 
                               activation = tf.nn.leaky_relu, padding='same')
        dec = dec + tf.random.normal(tf.shape(dec), 0.0, 0.01)
        dec = adain(dec, latent_space, curr_channels, batch_size)
            
    output = tf.layers.conv2d(dec, 3, 1, kernel_initializer = initializer, 
                              activation = tf.nn.tanh, padding='valid')

    return output
    

def get_loss(x, y):
    return tf.reduce_sum(tf.pow((x - y), 2))

def get_vgg_loss(x, y):
    from tensorflow.contrib.slim.nets import vgg as model_module
    combined_images = tf.concat([x, y], axis=0)
    input_img = (combined_images + 1.0) / 2.0
    VGG_MEANS = np.array([[[[0.485, 0.456, 0.406]]]]).astype('float32')
    VGG_MEANS = tf.constant(VGG_MEANS, shape=[1,1,1,3])
    vgg_input = (input_img - VGG_MEANS) * 255.0
    bgr_input = tf.stack([vgg_input[:,:,:,2], 
                          vgg_input[:,:,:,1], 
                          vgg_input[:,:,:,0]], axis=-1)
        
    slim = tf.contrib.slim
    with slim.arg_scope(model_module.vgg_arg_scope()):
        _, end_points = model_module.vgg_19(
        bgr_input, num_classes=1000, spatial_squeeze = False, is_training=False)

    loss = 0
    for layer in ['vgg_19/conv2/conv2_1', 'vgg_19/conv3/conv3_1']:
        layer_shape = tf.shape(end_points[layer])
        x_vals = end_points[layer][:layer_shape[0]//2]
        y_vals = end_points[layer][layer_shape[0]//2:]
        loss += tf.reduce_mean(tf.pow(x_vals - y_vals, 2))

    return loss