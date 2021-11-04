# Control of RL env

Control openai Reinforcement environment with hand gestures using tensorflow keras 
Mountain Car V0 control using hand gestures

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