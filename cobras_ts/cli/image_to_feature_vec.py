import os
import re

import tensorflow.compat.v1 as tf
# from tensorflow.python.platform.gfile import GFile
from tensorflow.python.platform import gfile
import numpy as np
import json


def create_graph(network_path):
    if not network_path:
        network_path = 'classify_image_graph_def.pb'

    # with gfile.FastGFile(network_path, 'rb') as f:
    with gfile.GFile(network_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images, network_path):
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    create_graph(network_path)

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.GFile(image, 'rb').read()

            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)

    return features



def convert_img_to_feature_vec(dir, network_path=None):
    list_images = [dir + '/' + f for f in sorted(os.listdir(dir)) if re.search('jpg|JPG|', f)]
    features = extract_features(list_images,network_path)
    return list_images, features


