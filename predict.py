#from memory_profiler import memory_usage
#print('Memory (Before): ' + str(memory_usage()) + 'MB' )

import tensorflow as tf
import numpy as np
import csv
import struct
import scipy


# Load the model.
def load_model():
  interpreter = tf.lite.Interpreter('yamnet_1.tflite')

  input_details = interpreter.get_input_details()
  waveform_input_index = input_details[0]['index']
  output_details = interpreter.get_output_details()
  scores_output_index = output_details[0]['index']
  embeddings_output_index = output_details[1]['index']
  spectrogram_output_index = output_details[2]['index']

  return interpreter, input_details, waveform_input_index, output_details, scores_output_index, embeddings_output_index, spectrogram_output_index


model, input_details, waveform_input_index, output_details, scores_output_index, embeddings_output_index, spectrogram_output_index = load_model()


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with open(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_names = class_names_from_csv('yamnet_class_map.csv')


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
  scores, embeddings, spectrogram = (
      model.get_tensor(scores_output_index),
      model.get_tensor(embeddings_output_index),
      model.get_tensor(spectrogram_output_index))


  infered_class = class_names[scores.mean(axis=0).argmax()]
  print(f'The main sound is: {infered_class}')


#predict('test.wav')
#print('Memory (After) : ' + str(memory_usage()) + 'MB')