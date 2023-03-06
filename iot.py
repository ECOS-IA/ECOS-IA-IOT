from memory_profiler import memory_usage
print('Memory (Before): ' + str(memory_usage()) + 'MB' )

import pyaudio
import wave
from predict import predict, send_to_server
import threading
from datetime import datetime

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 6 
WAVE_OUTPUT_FILENAME = "output.wav"

SERVER_IP = "192.168.1.172"
RASP_ID = 2

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
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        print("* done recording ")

        label,probability = predict(frames)
        thread = threading.Thread(target=send_to_server, args=[ {
            "label": label,
            "probability": probability,
            "timestamp": datetime.now().strftime('%d-%m-%Y %H:%M:%S'), 
            "id_raspberry": RASP_ID, 
            "server_ip": SERVER_IP
            } ])
        thread.start()
        
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