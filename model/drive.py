import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import torch
import torch.nn as nn

sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 30

class Nvidia(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*18, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        #print(x.shape)
        x = self.linear_layers(x)
       # print(x.shape)
        return x


def img_preprocess(img):
    img = img[60:135, :, :]
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    img = cv2.resize(img, (200,66))
    img = img/255
    return img

def to_tensor(img):
    a = img

    a = torch.tensor(img)

    a = torch.unsqueeze(a, dim = 0)

    a = a.permute(0,3,1,2)

    a = a.to(torch.float32)
    return a


def predict_steering(img, model):
    model.eval()

    with torch.no_grad():
        pred = model(img)
    return pred


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = to_tensor(image)
    steering_angle = float(predict_steering(image, model))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    model = Nvidia()
    model.load_state_dict(torch.load("model/23_model_1.pth", map_location=torch.device('cpu')))
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)