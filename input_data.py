#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np


def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(".")
        if 'cat' in name[0]:
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            if 'dog' in name[0]:
                dogs.append(file_dir + file)
                label_dogs.append(1)
        image_list = np.hstack((cats, dogs))
        label_list = np.hstack((label_cats, label_dogs))
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    # Put the label and the picture in temp and then shuffle the order, then take it out
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    # mess up the order
    np.random.shuffle(temp)

    # take the first element as image. take the second element as label
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list


# test get_files
imgs , label = get_files('/Users/xcliang/PycharmProjects/cats_vs_dogs/data/train/')
for i in imgs:
	print("img:",i)

for i in label:
	print('label:',i)
# end test get_files


# image_W ,image_H: Specify image size，batch_size: Number of reads per batch ， capacity: maximum number of elements in queue
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # Convert data to a format that ts can recognize
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # Put image and label in the queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    # Read all the information of the picture
    image_contents = tf.read_file(input_queue[0])
    # Decode the picture，channels ＝3 means colorful pirture, black and white picture is 1
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # Crop or expand the image to the specified image_W, image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # Standardize data: which means subtract its mean, divide by his variance
    image = tf.image.per_image_standardization(image)

    # Generate batch,  tf.train.shuffle_batch mess up the order
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)

    # Redefine the shape of label_batch
    label_batch = tf.reshape(label_batch, [batch_size])
    # convert picture
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

# test get_batch
import matplotlib.pyplot as plt
BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = '/Users/xcliang/PycharmProjects/cats_vs_dogs/data/train/'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
   i = 0
   #  Coordinator and start_queue_runners monitor the status of queue
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   # coord.should_stop() return true(data is finished), then call coord.request_stop()
   try:
       while not coord.should_stop() and i<1:
           # Test one step
           img, label = sess.run([image_batch, label_batch])

           for j in np.arange(BATCH_SIZE):
               print('label: %d' %label[j])
               plt.imshow(img[j,:,:,:])
               plt.show()
           i+=1
   # No data in the queue
   except tf.errors.OutOfRangeError:
       print('done!')
   finally:
       coord.request_stop()
   coord.join(threads)
sess.close()
