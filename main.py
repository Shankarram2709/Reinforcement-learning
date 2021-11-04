#!/usr/bin/python3
import json
import pandas as pd
import os
import glob
import argparse
from core.utils import  get_datapoint_list
from tensorflow.keras import models
from core.control import Control


if __name__ == '__main__':
    """
    1. Set step_parameter of the step you want to run.
    2. Run this python script with the step's name as argument.
    """

    parser                  = argparse.ArgumentParser()
    subparsers              = parser.add_subparsers(help='select a step', dest="step")
    parser_train            = subparsers.add_parser('train',                            help='train a net')
    parser_predict          = subparsers.add_parser('predict',                          help='produce a plain prediction from input image')
    parser_control          = subparsers.add_parser('control',                          help='produce a pygame window and webcam to control env with gestures')

    parser_train.add_argument('-p','--param-files',     dest='param_files',     type=str, help='path to *.json files containing training parameters.',  required=True)
    parser_train.add_argument('-c','--checkpoints-dir', dest='h5_dir', type=str, help='path to location where checkpoints are stored.',        required=True)
    parser_train.add_argument('-n','--number-of-classes', dest='num_classes', type=int,  help='decide number of clases 3 or 4-based on bg', nargs=1,    default= 3)

    parser_predict.add_argument('-m','--model',         dest='model_dir',   type=str,   help='path to the model directory containing params.json and weights.hdf5',     required=True)

    parser_control.add_argument('-m','--model',         dest='model_dir',   type=str,   help='path to the model directory containing params.json and weights.hdf5',     required=True)
    parser_control.add_argument('-o','--outpath',       dest='output_dir',   type=str,   help='path to output folder to save actions and rewards as df',     required=True)

    
    args = parser.parse_args()


    if args.step == 'train':
        '''step_params'''
        TRAIN_PARAMS_PATHS = glob.glob(args.param_files, recursive=False) 
        TRAIN_BASE_DIR = args.h5_dir
        NUM_CLASSES = args.num_classes
        ''''''

        if not os.path.isdir(TRAIN_BASE_DIR) or not os.access(TRAIN_BASE_DIR, os.R_OK):
            print("checkpoint dir does not exist or is not readable: {}. Creating".format(TRAIN_BASE_DIR))
            os.makedirs(TRAIN_BASE_DIR)
        if not os.path.isdir(TRAIN_BASE_DIR) or not os.access(TRAIN_BASE_DIR, os.R_OK):
            print("failed to create checkpoint dsplit_folderir: {}.".format(TRAIN_BASE_DIR))
            exit(2)

        if len(TRAIN_PARAMS_PATHS) < 1:
            print('warning: no parameters for training are given')

        from train import classify

        for train_params_path in TRAIN_PARAMS_PATHS:
            if not os.path.isfile(train_params_path) or not os.access(train_params_path, os.R_OK):
                print("param file does not exist or is not readable: {}. skip".format(train_params_path))
                continue
            train_params = json.load(open(train_params_path))
            model_name = train_params['name']
            print('start training for model: {} ---->'.format(model_name))
            classify(train_params, num_classes = NUM_CLASSES, log_dir=os.path.join(TRAIN_BASE_DIR,model_name, ''))
            print('---> finished training.')
            #split_folder(os.path.join(TRAIN_BASE_DIR, model_name))
            os._exit(1)
    
    if args.step =='control':
        model_path = args.model_dir
        outpath = args.output_dir
        model = models.load_model(model_path)

        from core.control import Control
        # used only model with 3 classes
        print('starting real time inference using webcam and env ')
        control = Control(outpath, model, width= 640, height = 480)
        control.control_car()

    if args.step =='predict':
        model_path = args.model_dir
        model = models.load_model(model_path)

        from core.predict import predict
        print('starting real time inference using webcam')
        predict(model)
    '''
    if args.step == 'heatmap':
        model_path = args.model_dir
        model = models.load_model(model_path)

        from core.heatmap import introspect
        print('Introspect Heatmaps using guided backprop')'''

        

        



    