import tensorflow as tf
import json
import gc
import os
import numpy as np
import pandas as pd
from core.data import DataManager
from core.utils import get_datapoint_list, ensure_dir
from core.customnet import CustomNetV2 
keras = tf.keras

def classify(params,
          num_classes = 3,
          log_dir='log/',
          epochs=None):
    """
    train Inception model for classification (hand gestures)
    :param params: dict - holding parameter for training (see script/params.json)
    :param log_dir: str - path to new directory where trained h5 is stored
    :param epochs: int
    """
    # prepare process
    if epochs is None:
        epochs = params['training']['epochs']
    ensure_dir(log_dir + '/')
    json.dump(params, open(log_dir + '/params.json', 'w'), indent=2, sort_keys=True)
    #from IPython import embed; embed()
    # specify gpu to use for training
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(params['training']['gpu'])
    # Get datasets
    batch_size = int(params['training']['batch_size'])
    tr_data_manager = DataManager(
        datapoint_list=get_datapoint_list(params['data']['general']['tr_datapoint_list']),
        data_general_params=params['data']['general'],
        augmentation_params= params['data']['augmentation']['tr'],
        load_data_in_ram=params['training']['load_data_in_ram'],
        shuffle_buffer=batch_size * 10,
        prefetch_size=batch_size * 4,
        batch_size=batch_size,
        replay_buffer=params['training']['replay']['buffer_size'],
        replay_probability=params['training']['replay']['probability'],
        max_value=params['data']['general']['max_value'],
        min_value=params['data']['general']['min_value'],
        num_classes=num_classes)
    tr_dataset = tr_data_manager.get_tf_dataset()
    tr_data_iterator = tr_data_manager.get_tf_iterator()
    tr_data_manager.save_datapoint_list_as_df(outpath=log_dir + '/tr.csv')

    va_data_manager = DataManager(
        datapoint_list=get_datapoint_list(params['data']['general']['va_datapoint_list']),
        data_general_params=params['data']['general'],
        augmentation_params=None,  # params['data']['augmentation']['va'],
        load_data_in_ram=params['training']['load_data_in_ram'],
        shuffle_buffer=batch_size * 10,
        prefetch_size=batch_size * 4,
        batch_size=batch_size,
        max_value=params['data']['general']['max_value'],
        min_value=params['data']['general']['min_value'],
        num_classes=num_classes)
    va_dataset = va_data_manager.get_tf_dataset()
    va_data_iterator = va_data_manager.get_tf_iterator()
    va_data_manager.save_datapoint_list_as_df(outpath=log_dir + '/va.csv')

    # Create model
    model_dir_to_load = params['model']['model_dir_to_load']
    if model_dir_to_load is None:
        netv2=CustomNetV2(image_shape=[380,240,1],
                                out_channel=num_classes)
        model = netv2.get_model()

    #from IPython import embed; embed()
    model.compile(optimizer=tf.keras.optimizers.Adam(params['training']['learning_rate']),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # training procedure using callbacks for procedure observation

    reduce_lr=keras.callbacks.ReduceLROnPlateau(
            factor=0.1,
            patience=4,
            min_lr=0.000000001,
            cooldown=1,
            verbose=1)
    callbacks=[reduce_lr]
    history = model.fit(
        tr_dataset,
        epochs=epochs,
        steps_per_epoch=params['training']['steps_per_epoch'],
        validation_data=va_dataset,
        validation_steps=params['training']['va_steps'],
        callbacks=callbacks)
    model.save(log_dir+'/model.h5')
    #tf.contrib.saved_model.save_keras_model(model,log_dir + '/model')
    import pickle
    with open(log_dir+"train-hist-dict", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    # clear memory
    del model
    del tr_data_manager
    del tr_dataset
    del tr_data_iterator
    del va_data_manager
    del va_dataset
    del va_data_iterator
    del callbacks
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    gc.collect()


if __name__ == "__main__":
    training_params = json.load(open('/home/ram/rl/params.json'))

    classify(training_params,
          log_dir='/home/ram/rl/log',
          num_classes = 3,
          epochs = None)