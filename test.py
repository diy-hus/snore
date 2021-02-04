import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pyaudio
import librosa
import librosa.display

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

#-----------------------
import RPi.GPIO as GPIO
from folder_nrf.nrf24l01 import NRF24L01
from folder_nrf.config import *
#init nrf24
rf24 = NRF24L01()
#---------------------------

MODEL_NAME = 'mobilenet'
GRAPH_NAME = 'mobilenet_quant_edgetpu.tflite'
LABELMAP_NAME = 'labelmap.txt'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
IMG_SIZE = (224, 224)
INPUT_MEAN = 127.5
INPUT_STD = 127.5
count = 0

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = self.RATE * 2
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global interpreter, input_data, output_details, labels, count, floating_model
        N_FFT = 1024         # 
        HOP_SIZE = 1024      #  
        N_MELS = 128          # Higher   
        WIN_SIZE = 1024      # 
        WINDOW_TYPE = 'hann' # 
        FEATURE = 'mel'      # 
        FMIN = 1400
        
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        # numpy_array = nr.reduce_noise(audio_clip=numpy_array, noise_clip=numpy_array, prop_decrease = 1, verbose=False)
        S = librosa.feature.melspectrogram(y=numpy_array, sr=self.RATE,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE, 
                                        n_mels=N_MELS, 
                                        htk=True, 
                                        fmin=FMIN, # higher limit ##high-pass filter freq.
                                        fmax=self.RATE / 4) # AMPLITUDE
                                        
        fig = plt.figure(1,frameon=False)
        fig.set_size_inches(2.24,2.24)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(librosa.power_to_db(S**2,ref=np.max), fmin=FMIN) #power = S**2
        file_name = 'tmp.png'
        fig.savefig(file_name)
        count += 1
                            
        
        image = cv2.imread(file_name)
        cv2.imshow('Melspec', image)
        cv2.waitKey(1)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(image, axis=0)
        
        scale, zero_point = input_details[0]['quantization']
        input_data[:, :] = np.uint8(input_data / scale + zero_point)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        Y_pred = interpreter.get_tensor(output_details[0]['index'])[0]
        y_pred = np.argmax(Y_pred, axis=0)
        print(labels[y_pred])
        
        # tranform result
        print(y_pred)
        #print(type(y_pred))
        rf24.write(b'%d  ' % y_pred)

        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(1.0)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = (input_details[0]['dtype'] == np.float32)

audio = AudioHandler()
audio.start()
audio.mainloop()
audio.stop()

