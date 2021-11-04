"""
Define augmentation functions in this file
"""
import cv2
import numpy as np
from skimage import transform
import time
import logging

np.random.seed(1713)

LOG_FUNCTION_DURATION = False

if LOG_FUNCTION_DURATION:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='AUGMENTATION.log',
                        filemode='w')
    time_logger = logging.getLogger('augmentation time logger')


def timing(f):
    def wrap(*args):
        if LOG_FUNCTION_DURATION:
            time1 = time.time()
            ret = f(*args)
            time2 = time.time()
            time_logger.info('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        else:
            ret = f(*args)
        return ret

    return wrap


class Augmentor(object):
    def __init__(self,
                 params):
        """
        provides several augmentations for 2D images
        :param params: see scripts/params.json
        """
        self.params = params

    @staticmethod
    def _gaussian(x, mu, sig):
        return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)

    @staticmethod
    @timing
    def _add_gaussian_noise(image, noise_variance):
        if noise_variance is not None:
            random_noise_variance = np.abs(np.random.normal(0.1) * noise_variance)
            image += np.random.normal(0.0, random_noise_variance ** 0.5, image.shape)
        return image


    @staticmethod
    @timing
    def _median(image, params):
        if params is not None:
            do_it = True
            if params['propability'] is not None:
                propability = np.clip( params['propability'], 0.0, 1.0)
                do_it = np.random.choice([True,False],p=[propability, 1.0-propability])
            if do_it:
                if params['size'] is not None:
                    kernel_size = np.random.randint(3, np.clip(params['size'],3,21)+1)
                    if kernel_size % 2 == 0:
                        kernel_size = kernel_size - 1
        return image


    @staticmethod
    @timing
    def _flipping(image, params):
        if params is not None:
            if np.random.binomial(1, 0.5) and params['lr']:
                image = np.fliplr(image)
            if np.random.binomial(1, 0.5) and params['ud']:
                image = np.flipud(image)

        return image

    @staticmethod
    @timing
    def _rotation(image, params):
        """
        only works with float
        counter-clockwise rotation
        """
        if params is not None:
            if params['angle_range'] is not None:
                # random_rot_angle = np.random.randint(params['angle_range'][0],
                #                                      params['angle_range'][1], 1)
                random_rot_angle=(params['angle_range'][1]-params['angle_range'][0])*0.25*np.random.randn(1)
                image = transform.rotate(image,
                                         angle=float(random_rot_angle),
                                         mode='constant',
                                         clip=True,
                                         resize=False,
                                         preserve_range=True)

        return image

    @staticmethod
    @timing
    def _tranlation(image,params):
        "only left and right"
        if params is not None:
            if np.random.uniform(0.0, 1.0)<0.5:
                x= transform.AffineTransform(translation=(20,0))
                image=transform.warp(image,x)

                return image
            if np.random.uniform(0.0,1.0)>0.5:
                x= transform.AffineTransform(translation=(-20,0))
                image=transform.warp(image,x)
                
                return image

    @staticmethod
    @timing
    def _rescaling(image, rescale_factors):
        if rescale_factors is not None:
            rescale_factor_y = np.random.uniform(rescale_factors['y'][0], rescale_factors['y'][1] + 0.00001)
            if rescale_factors['keep_aspect_ratio']:
                rescale_factor_x = rescale_factor_y
            else:
                rescale_factor_x = np.random.uniform(rescale_factors['x'][0], rescale_factors['x'][1] + 0.00001)
            image = transform.rescale(image, scale=[rescale_factor_y, rescale_factor_x], multichannel=True,
                                      preserve_range=True, anti_aliasing=False, mode='constant')
    
        return image

    def augment(self, image):
        """
        Augmentation params can be set to None when no augmentation should be conducted.

        :param image: np.array((y,x,1), dtype=np.float32)
        :param params: dict - see scripts/params.json
        :return: image
        """
        # some checks
        if self.params is None:
            return image
        # check image shape on channel dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        # conduct augmentations
        image= self._rescaling(image, self.params['random_rescaling_factors'])
        image= self._rotation(image, self.params['random_rotation'])
        image= self._flipping(image, self.params['flip'])
        image = self._add_gaussian_noise(
            image,
            self.params['noise_variance'])
        image = self._median(image, self.params['median'])

        if LOG_FUNCTION_DURATION:
            time_logger.info('-----------------')
        return image
