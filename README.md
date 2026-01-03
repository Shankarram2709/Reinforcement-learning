# Control of OpenAI Gym environment using human-in-loop through real time hand gesture recognition

This repo demonstates controlling an openAI Reinforcement environment with hand gestures captured via webcam trained in Tensorflow-keras
Mountain Car V0 control using hand gestures

## Environment

![Mountain_Car](https://github.com/user-attachments/assets/9664ad74-6b11-42fe-adec-7b4ceab6d8f9)

## Action Mapping
Class-0(Considered as background)- Action space =1

Class-1(backward movement)- Action space = 0

class-2(Neutral) - Action space = 1

class-3(forward movement) - Action space = 2

| ![Class1](https://user-images.githubusercontent.com/50954616/146073007-92031bb6-0a50-4428-9de2-6bb392fb7f93.png) ![Class2](https://user-images.githubusercontent.com/50954616/146073389-68288a85-55a9-4c6b-941e-455f3eb10c9b.png) ![Class3](https://user-images.githubusercontent.com/50954616/146073551-630a7602-3a9d-4cae-86c0-e6517b804ae0.png) |
|:--:| 
|*Class1 (Backward)*   *Class2 (Neutral)*    *Class3 (Forward)*|


Create list of images for training
```
python3 generate_tr_va.py path to train and val images
````

To train hand gesture images
```
python3 main.py  train -p path to params.json -c path to where weights should be stored
```

To control Mountain car with gestures
```
python3 main.py control -m path to trained model -o output path for action and reward df
```

To perform real time inference on webcam
```
python3 main.py predict -m path to trained model
```

## Project Structure
```
Reinforcement-learning/
├── core/                   # Core modules (model architecture, utilities)
├── main.py                # CLI entry point (train/predict/control modes)
├── train.py               # Training logic and model compilation
├── generate_tr_va.py      # Data preparation and list generation
├── params.json            # Training hyperparameters and config
├── bg_3class.h5           # Trained model (3 gesture classes)
├── bg_4class.h5           # Trained model (4 classes with background)
├── act_rew.lst            # Action-reward logs from RL episodes
└── README.md              # This file
```

# Note
Empty frames without any gestures are considered as null (or) neutral
