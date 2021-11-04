"""
data loading and provision for keras/tensorflow model training, prediction, etc.
"""
import tensorflow as tf
import pandas as pd
import itertools
import cv2
import os
import numpy as np
import base64
from PIL import Image
import io
import json
from skimage import transform
from core.aug import Augmentor
from core.utils import get_datapoint_list
import math
keras = tf.keras


class DataManager(object):
    random_nr = 0

    def __init__(self,
                 datapoint_list,
                 data_general_params,
                 image_on_full_res=True,
                 load_data_in_ram=False,
                 augmentation_params=None,
                 min_value=None,
                 max_value=None,
                 init_tf_dataset_and_tf_iterator=True,
                 shuffle_buffer=64,
                 prefetch_size=64,
                 batch_size=16,
                 replay_buffer=1200,
                 replay_probability=0.1,
                 num_classes=3
                 ):

        """
        provides data for keras/tensorflow training, etc.
        optionally augmentation can be conducted.
        :param datapoint_list: list of paths
        :param data_general_params: see scripts/params.json
        :param load_data_in_ram: bool
        :param augmentation_params: see scripts/params.json
        :param min_value: float or int that is used to normalize the images
        :param max_value: float or int that is used for normalization of images
        :param init_tf_dataset_and_tf_iterator: bool
        :param shuffle_buffer: bool
        :param prefetch_size: int
        :param batch_size: int
        """
        self.data_general_params = data_general_params
        self.image_on_full_res = image_on_full_res
        self.full_res = None
        self.datapoint_list = get_datapoint_list(datapoint_list)
        self.image_min_value = data_general_params["min_value"] if min_value is None else min_value
        self.image_max_value = data_general_params["max_value"] if max_value is None else max_value
        self.augmentation_params = augmentation_params
        self.replayer = Replayer(buffer_size=replay_buffer)
        self.replay_probability = replay_probability
        self.num_classes = num_classes
        tf.compat.v1.disable_eager_execution()
        # init
        self._init_datapoint_lst(datapoint_list=datapoint_list,
                                 load_data_in_ram=load_data_in_ram)
        if augmentation_params is not None:
            self._init_augmentor(augmentation_params=augmentation_params)
        else:
            self.augmentor = None

        if init_tf_dataset_and_tf_iterator:
            self._init_tf_dataset(shuffle_buffer=shuffle_buffer,
                                  batch_size=batch_size,
                                  prefetch_size=prefetch_size
                                  )
            self._init_tf_iterator()
        else:
            self.tf_dataset = None
            self.tf_iterator = None

    def _init_datapoint_lst(self,
                            datapoint_list,
                            load_data_in_ram):
        self.datapoint_list = datapoint_list
        if load_data_in_ram:
            self.data_in_ram = []
            '''
            get a list of json/lif-file data and load it into ram
            '''
            for path in datapoint_list:
                loaded_datapoint = self._load_datapoint(path)
                # optionally rescale
                if self.data_general_params['general_rescale_factor'] is not None:
                    rescale_factors = (float(self.data_general_params['general_rescale_factor']['y']),
                                       float(self.data_general_params['general_rescale_factor']['x']))
                    if not self.image_on_full_res:
                        loaded_datapoint['image'] = transform.rescale(loaded_datapoint['image'],
                                                                      order=1,
                                                                      scale=rescale_factors,
                                                                      multichannel=True,
                                                                      mode='constant',
                                                                      anti_aliasing=False,
                                                                      preserve_range=True,
                                                                      clip=True)
                self.data_in_ram.append(loaded_datapoint)
            self.num_data = len(self.data_in_ram)
            self.indices_data_in_ram = range(self.num_data)
        else:
            self.data_in_ram = None
            self.num_data = len(datapoint_list)
            self.indices_data_in_ram = range(self.num_data)
        self.idx_counter = 0
        self.counter = 0

    def _init_augmentor(self, augmentation_params):
        self.augmentor = Augmentor(params=augmentation_params)

    def _init_tf_dataset(self,
                         shuffle_buffer=1024,
                         batch_size=16,
                         prefetch_size=64):

        datapoint_generator = self.get_datapoint_generator()
        output_types = (tf.float32, tf.float32)
        output_shapes = (self.data_general_params['image_shape'],[3])
        data = tf.data.Dataset.from_generator(
            datapoint_generator,output_types,output_shapes)
        if shuffle_buffer is not None:
            data = data.shuffle(shuffle_buffer)
        data = data.batch(batch_size)
        data = data.prefetch(prefetch_size)
        self.tf_dataset = data
        
    def _init_tf_iterator(self):

        self.tf_iterator = tf.compat.v1.data.make_initializable_iterator(self.tf_dataset)
        sess = tf.compat.v1.Session()
        sess.run(self.tf_iterator.initializer)
        #with tf.Session() as sess:
        #  #Initialize variables
        #  sess.run(tf.global_variables_initializer())
        #  sess.run(self.tf_iterator)

    def get_datapoint_generator(self):
        def generator():
            for i in itertools.count(1):
                datapoint_dict = self._get_next_datapoint()
                if datapoint_dict['image'] is not None and datapoint_dict['lab'] is not None: 
                  yield datapoint_dict['image'], datapoint_dict['lab']
                else:
                  continue

        return generator

    def _get_next_datapoint (self):
        """
        get next datapoint from ram, if filled else load from disc.
        then process cropping, augmentation etc.
        """
        #if self.counter==0:
        # replay dtp or generate new one
        if np.random.uniform(0.0, 1.0) < self.replay_probability and self.replayer.is_filled():
            datapoint_dict = self.replayer.get_item()
            image, lab = datapoint_dict['image'], datapoint_dict['lab']
        else:
            if self.replayer.is_filled():
                self.replayer.remove_item()
            if self.data_in_ram is None:
                datapoint_dict = self._load_datapoint(self.datapoint_list[self.idx_counter])
            else:
                image = self.data_in_ram[self.indices_data_in_ram[self.idx_counter]]['image']
                lab = self.data_in_ram[self.indices_data_in_ram[self.idx_counter]]['lab']
                datapoint_dict = {'image': image.copy(),
                                  'lab': lab.copy() if lab is not None else None}
            # next datapoint
            self.idx_counter += 1
            if self.idx_counter >= self.num_data:
                # reset counter
                self.idx_counter = 0

            # unpack
            image = datapoint_dict['image']
            lab = datapoint_dict['lab']

            #num_classes = 3
            #lab = np.eye(num_classes)[lab]

            # optionally augment 
            if self.augmentor is not None:
                image= self.augmentor.augment(image=image)

            # general rescale
            if self.data_general_params['general_rescale_factor'] is not None and not self.data_in_ram:
                # if self.data_in_ram -> rescaling already conducted when loading data in ram
                rescale_factors = (float(self.data_general_params['general_rescale_factor']['y']),
                                   float(self.data_general_params['general_rescale_factor']['x']))
                if not self.image_on_full_res:
                    image = transform.rescale(image,
                                              order=1,
                                              scale=rescale_factors,
                                              multichannel=True,
                                              mode='constant',
                                              preserve_range=True,
                                              clip=True)

            # ensure value range
            if image is not None:
                image = np.clip(image, 0.0, 1.0, out=image)

            # add to replayer
            if self.replayer is not None:
                self.replayer.add_item({'image': image, 'lab': lab})   
            
        return {'image': image, 'lab': lab}       

    def _load_datapoint(self, path):
        """
        loads a datapoint from file of diverse type:
        .bmp (uint16), .png formats are provided
        :param path: str
        :return: datapoint containing image and labels
        """
        # get path
        file_type = path.split('.')[-1]
        class_type = os.path.split(path)
        lab = None
        if self.num_classes>3:
            if "class_1" in path:
                lab=np.array([1,0,0,0])
            if "class_2" in path:
                lab=np.array([0,1,0,0])
            if "class_3" in path:
                lab=np.array([0,0,1,0])
            if "class_0" in path:   #if bg is added as a particular class
                lab=np.array([0,0,0,1])

        if self.num_classes==3:
            if "class_1" in path:
                lab=np.array([1,0,0])
            if "class_2" in path:
                lab=np.array([0,1,0])
            if "class_3" in path:
                lab=np.array([0,0,1])

        if file_type == 'bmp':
            # load original data and replace data from above
            img = cv2.imread(path)
            if img.dtype=='uint16':
                #for prediction
                imgres=cv2.imread(path,cv2.COLOR_BGR2GRAY)
            else:
                img=imgres
        elif file_type == "png":
            #load data for training
            imgres = cv2.imread(path)
            imgres = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)

        else:
            raise ValueError('bad data to read. file_type for {} is {}'.format(path, file_type))
        
        if len(imgres.shape) == 2:
            imgres = np.expand_dims(imgres, 2)
        # ensure float and value range
        # standardization
        imgres = np.clip(imgres, self.image_min_value, self.image_max_value, out=imgres)
        imgres -= self.image_min_value
        imgres = imgres / float(self.image_max_value - self.image_min_value)
        #from IPython import embed;embed()
        return {'image': imgres, 'lab': lab}

    def get_tf_dataset(self):
        return self.tf_dataset

    def get_tf_iterator(self):
        return self.tf_iterator

    def save_datapoint_list_as_df(self, outpath):
        pd.DataFrame({'path': self.datapoint_list}).to_csv(outpath,index=False)
    
    
class Replayer(list):
    def __init__(self, lst=[], buffer_size=None):
        super(Replayer, self).__init__(lst)
        self.buffer_size = buffer_size
        self.offset = 32

    def is_filled(self):
        if len(self) > max(self.offset, (0.5 * self.buffer_size)):
            return True
        else:
            return False

    def add_item(self, item):
        if len(self) < self.buffer_size:
            self.append(item)
        else:
            self[-1] = item

    def get_item(self, idx=None):
        if self:
            if idx is None:
                return self[np.random.randint(self.offset, len(self))]
            else:
                if idx >= 0 and idx < len(self):
                    return self[idx]

    def remove_item(self, idx=0):
        if self:
            if idx >= 0 and idx < len(self):
                del self[idx]