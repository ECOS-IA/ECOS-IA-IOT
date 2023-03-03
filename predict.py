#from memory_profiler import memory_usage
#print('Memory (Before): ' + str(memory_usage()) + 'MB' )

import tensorflow as tf
import numpy as np
import csv
import struct
import scipy
import requests
import json



# Load the model.
def load_model():
  interpreter = tf.lite.Interpreter('model.tflite')

  input_details = interpreter.get_input_details()
  waveform_input_index = input_details[0]['index']
  output_details = interpreter.get_output_details()
  scores_output_index = output_details[0]['index']
  #embeddings_output_index = output_details[1]['index']
  #spectrogram_output_index = output_details[2]['index']

  return interpreter, input_details, waveform_input_index, output_details, scores_output_index


model, input_details, waveform_input_index, output_details, scores_output_index = load_model()
my_classes = ["chainsaw", "fireworks", "crackling_fire", "engine","thunderstorm","wind","dog"]
map_class_to_id = {"chainsaw":0, "fireworks":1, "crackling_fire":2, "engine":3, "thunderstorm":4, "wind":5, "dog":6}


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


def predict(frames, sample_rate=16000):
  wav_data = np.array([struct.unpack('1024h', frame) for frame in frames]).flatten()

  # Show some basic information about the audio.
  duration = len(wav_data)/sample_rate
  print(f'Sample rate: {sample_rate} Hz')
  print(f'Total duration: {duration:.2f}s')
  print(f'Size of the input: {len(wav_data)}')


  # data scaling
  waveform = wav_data / tf.int16.max
  waveform = waveform.astype(np.float32)

  # prediction
  model.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
  model.allocate_tensors()
  model.set_tensor(waveform_input_index, waveform)
  model.invoke()
  scores=model.get_tensor(scores_output_index)

  class_probabilities = tf.nn.softmax(scores, axis=-1).numpy()
  args = tf.argsort(class_probabilities)[::-1].numpy()
  label, probability = my_classes[args[0]], class_probabilities[args[0]]
  print(f'The main sound is: {label} ({round(probability*100,2)}%)')
  
  return label, probability


def send_to_server(data):
    danger_list = ["chainsaw", "fireworks", "crackling_fire", "engine"]

    if data["label"] in danger_list and data["probability"] > 0.9:
      print("send to server.....")
      res = requests.post(f"http://{data['server_ip']}:4000/alert", data=json.dumps(data))
      print(res.text)

#predict('test.wav')
#print('Memory (After) : ' + str(memory_usage()) + 'MB')