import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

MODEL = whisper.load_model("base")

def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Speak now...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    wav.write(filename, fs, recording)
    print("Recording complete")

    return filename


def transcribe_audio(filename):
    result = MODEL.transcribe(filename)
    return result["text"]


def listen():
    audio_file = record_audio()
    text = transcribe_audio(audio_file)
    print("You said:", text)
    return text