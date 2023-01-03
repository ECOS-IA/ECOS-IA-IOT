from memory_profiler import memory_usage
print('Memory (Before): ' + str(memory_usage()) + 'MB' )

import pyaudio
import wave
import time
from predict import predict

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

try:
    val=0
    while True:
        print("\n",f"---------------- Frames sequence N°{val}")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording ")
        predict(frames)
        val=val+1

except KeyboardInterrupt:
    print('Memory (After) : ' + str(memory_usage()) + 'MB')
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


stream.stop_stream()
stream.close()
p.terminate()