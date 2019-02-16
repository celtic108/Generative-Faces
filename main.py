import utils.load_data as load_data
import numpy as np
import tensorflow as tf
import network_utils
import misc_utils
from random import shuffle
import time

epochs = 1
batch_size = 8
size = (512, 512)
z_space = 256
learning_rate = 0.001
loss_smoothing = 1000
model_path = '/tmp/model.ckpt'


file_list = load_data.get_image_file_list('./lag-dataset/')
shuffle(file_list)
total_number_of_images = len(file_list)

X = tf.placeholder(tf.float32, shape=(None, *size, 3))


latent_space = network_utils.encoder(X, z_space)

output = network_utils.decoder(latent_space, batch_size, z_space)

dataset = tf.data.Dataset.from_tensor_slices(file_list)
dataset = dataset.map(lambda filename: tf.py_func(load_data.load_image, [filename, size], [tf.float32]))
dataset = dataset.shuffle(buffer_size = 64)
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

loss = network_utils.get_loss(X, output)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

start_time = time.time()
iterations = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, model_path)
        print("Restored model from ", model_path)
    except:
        print("No model found. Using random initialization.")
    for j in range(epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                a = sess.run(next_element)
                if a[0].shape[0] == batch_size:
                    l, _ = sess.run([loss, optimizer], feed_dict = {X:a[0]})
                    if iterations == 0:
                        running_loss = l
                    else:
                        running_loss = misc_utils.smooth_loss(l, loss_smoothing, iterations, running_loss)
                    print(j, iterations, l, running_loss)
                    iterations += 1
            except tf.errors.OutOfRangeError:
                print("Epoch complete. Average epoch time: ", (time.time()-start_time)/(j+1))
                saver.save(sess, model_path)
                sess.run(iterator.initializer)
                a = sess.run(next_element)
                temp_image = sess.run(output, feed_dict={X:a[0]})
                load_data.display_image(load_data.unpreprocess_image(a[0][0]))
                load_data.display_image(load_data.unpreprocess_image(temp_image[0]))
                break
            