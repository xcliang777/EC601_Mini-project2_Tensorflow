#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import numpy as np
import model
import os


# Select an image from the training set
def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0, n)
    img_dir = os.path.join(train, files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    train = '/Users/xcliang/PycharmProjects/cats_vs_dogs/data/test/'

    # Get image path set and label set
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1  # Only read one image, therefore set batch as 1
        N_CLASSES = 2  # Probability of 2 output neurons, [1,0] or [0,1] cats and dogs
        # convert image format
        image = tf.cast(image_array, tf.float32)
        # Picture standardization
        image = tf.image.per_image_standardization(image)
        # The image was originally three-dimensional [208, 208, 3] redefining the image shape to a 4D four-dimensional tensor
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        # Since the inference returns without an activation function, the result is activated with softmax here.
        logit = tf.nn.softmax(logit)

        # Enter data into the model using the most primitive input data. placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # The path to store the model
        logs_train_dir = '/Users/xcliang/PycharmProjects/cats_vs_dogs/data/saveNet/'
        # define saver
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Load the model from the specified path. . . ")
            # Load the model into sess
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('The model is loaded successfully, and the number of steps in the training is %s' % global_step)
            else:
                print('Model failed to load,,, file not found')
                # Import images into model calculations
            prediction = sess.run(logit, feed_dict={x: image_array})
            # Get the index of the maximum probability in the output
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('Cat probability %.6f' % prediction[:, 0])
            else:
                print('Dog probability %.6f' % prediction[:, 1])
            # test


evaluate_one_image()
