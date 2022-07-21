import os
import numpy as np

import tensorflow as tf
from bs4 import BeautifulSoup


#  flow form directory
class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df,
                 image_ids,
                 batch_size,
                 folder="dataset",
                 input_size=(224, 224, 3),
                 shuffle=False):
        # self.ids = valid_ids
        self.df = df
        self.image_ids = image_ids
        self.folder = folder
        # self.path = os.path.join(dir, folder)
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)

    def __get_input(self, id):
        path = self.df.loc[df['image'] == id]['path']
        image = tf.keras.preprocessing.image.load_img(os.path.join(self.folder, path))
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        return image_arr / 255.0

    def __get_output(self, id):
        temp = self.df.loc[df['image'] == id]

        img_shape = temp['shape']

        xmin = temp['xmin']
        xmax = temp['xmax']
        ymin = temp['ymin']
        ymax = temp['ymax']

        h, w, c = image_shape.split(',')

        return [xmin * 256 / w, ymin * 256 / h, xmax * 256 / w, ymax * 256 / h]

    #     xml_path = os.path.join(self.path, id + ".xml")
    #
    #     with open(xml_path) as f:
    #         soup = BeautifulSoup(f, 'xml')
    #         boxes = []
    #         objects = soup.find_all('object')
    #
    #         for obj in objects:
    #             xmin = int(obj.bndbox.xmin.text)
    #             ymin = int(obj.bndbox.ymin.text)
    #             xmax = int(obj.bndbox.xmax.text)
    #             ymax = int(obj.bndbox.ymax.text)
    #
    #             box = [xmin, ymin, xmax, ymax]
    #             boxes.append(box)
    #         return boxes

    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(x) for x in ids])
        y_batch = np.asarray([self.__get_output(x) for x in ids])

        # y_batch = np.asarray([ [xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in zip(batches['xmin'], batches['ymin'], batches['xmax'], batches['ymax'])])

        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size
