import utils.load_data as load_data
import numpy as np
import tensorflow as tf

epochs = 1

file_list = load_data.get_image_file_list('./lag-dataset/')
total_number_of_images = len(file_list)

X = tf.placeholder(tf.float32, shape=(None, 200, 200, 3))

net = X
initializer = tf.initializers.random_normal(0, 0.001)
for j in range(5):
    for i in range(2):
        net = tf.layers.conv2d(net, 32 * 2**(j//2), 3, 
                               kernel_initializer = initializer,
                               activation = tf.nn.leaky_relu)
    net = tf.layers.conv2d(net, 32 * 2**(1 + j//2), 3, strides=(2,2), 
                           kernel_initializer = initializer,
                           activation = tf.nn.leaky_relu)

latent_space = net



dataset = tf.data.Dataset.from_tensor_slices(file_list)
dataset = dataset.map(lambda filename: tf.py_func(load_data.load_image, [filename], [tf.float32]))
dataset = dataset.shuffle(buffer_size = 1024)
dataset = dataset.batch(8)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


iterations = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                a = sess.run(next_element)
                print(j, iterations, sess.run(net, feed_dict = {X:a[0]}).shape)
                iterations += 1
            except tf.errors.OutOfRangeError:
                print("Epoch complete")
                break
            