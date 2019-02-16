import tensorflow as tf
import numpy as np


initializer = tf.initializers.random_normal(0, 0.001)    

def encoder(net, latent_size):
    for j in range(9):
        curr_channels = min(latent_size, 32 * 2**(j//2))
        for i in range(2):
            net = tf.layers.conv2d(net, curr_channels, 3, padding='same',
                                   kernel_initializer = initializer,
                                   activation = tf.nn.leaky_relu)
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