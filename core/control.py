import gym
#import os
#from gym.utils.play import *
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras import models
from random import randrange
#from pygame.locals import VIDEORESIZE
import pygame
import pyautogui


class Control(object):

    def __init__(self, outpath, model, width= 640, height = 480):
        self.model = model
        self.width = width
        self.height = height

    @staticmethod
    def display_arr(screen, arr,video_size, transpose):
        """
        function invoked from https://github.com/openai/gym/blob/103b7633f564a60062a25cc640ed1e189e99ddb7/gym/utils/play.py
        conversion of changes in mountain car movements into pixel arrays and transposing it back to screen 
        """
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        screen.blit(pyg_img, (0, 0))

    def control_car(self):    
        # specifygame
        env = gym.make('MountainCar-v0')

        cap = cv2.VideoCapture(0)
        cap.set(3,self.width)
        cap.set(4,self.height)
        
        #init game
        '''
        pygame.init()
        #video_size = render_display(env)
        #init display
        screen = pygame.display.set_mode((800,600))
        #clock = pygame.time.Clock()
        #init surface
        #instead rendering the initial obs and frame we create surface
        env.reset()
        pyg_img = pygame.surfarray.make_surface(env.render(mode='rgb_array'))
        pyg_img = pygame.transform.scale(pyg_img,(800,600))
        screen.blit(pyg_img,(0,0))
        '''

        env.reset()
        rendered = env.render(mode="rgb_array")
        #from IPython import embed;embed()
        #relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

        video_size = [rendered.shape[1], rendered.shape[0]]
        zoom = None
        if zoom is not None:
            video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

        resolution =(video_size[0],video_size[1])
        #codec = cv2.VideoWriter_fourcc(*"XVID")
        #filename = "Record1.avi"
        #fps = 60
        #video_out = cv2.VideoWriter(filename,0,1,resolution)
        running = True
        env_done = False
        screen = pygame.display.set_mode(video_size)
        clock = pygame.time.Clock()
        action_list = []
        reward_list = []
        while running:
            ret,frame = cap.read()
            x = frame.shape[0]
            y = frame.shape[1]
            roi = frame[100:(100+512),400:(400+x)]
            gray =cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            im = Image.fromarray(gray,'L')
            cv2.imshow("roi",gray)
            #cv2.imshow("frame",frame)
            #im = Image.fromarray(roi,'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array,axis = 0)
            img_array = np.expand_dims(img_array,axis= -1)
            prediction = self.model.predict(img_array)
            #from IPython import embed;embed()
            #if prediction.argmax(axis = -1)==3:
            #    continue
            if prediction.argmax(axis=-1) is None:
                gest = 1
                action = gest
            else:
                gest = prediction.argmax(axis=-1)
                action = gest[0]
            print(action)
            action_list.append(action)
            obs, rew, env_done, info = env.step(action)
            reward_list.append(rew)
            prev_obs = obs
            #if callback is not None:
            #    callback(prev_obs, obs, action, rew, env_done, info)
            if obs is not None:
                rendered = env.render(mode="rgb_array")
                self.display_arr(screen, rendered, transpose=True, video_size=video_size)
            #rec = pyautogui.screenshot()
            #rec_frame = np.array(rec)
            #video_out.write(rec_frame)
            pygame.display.flip()
            clock.tick(30)
            #clock.tick(60)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        df = pd.DataFrame(np.column_stack([action_list, reward_list]), 
                               columns=['actions', 'reward'])

        df.to_csv(outpath+'/'+'act_rew.lst',index=False)
