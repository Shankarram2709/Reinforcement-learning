# Control of RL env

Control openai Reinforcement environment with hand gestures using tensorflow keras 
Mountain Car V0 control using hand gestures

Create list of images for training

Class-0(Considered as background)
Class-1(backward movement)
class-2(Neutral)
class-3(forward movement)

![Alt text](/home/ram/rl/gestures/class_2/Class2_10.png "Class-1")
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

# Note

Empty frames without any gestures are considered as null (or) neutral