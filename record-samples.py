#!/usr/bin/python3

import pyaudio, wave, config

def main():
    buffer_size = config.BUFFER_SIZE
    sample_rate = config.SAMPLE_RATE
    sample_format = pyaudio.paFloat32
    channels = 1
    seconds = 2
    filename = 'samples/sample-03.wav'

    p = pyaudio.PyAudio()

    print('Recording')

    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=sample_rate,
        frames_per_buffer=buffer_size,
        input=True
    )

    frames = []
    for i in range(0, int(sample_rate / buffer_size * seconds)):
        data = stream.read(buffer_size)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    print('Saving file')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print('File saved')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
