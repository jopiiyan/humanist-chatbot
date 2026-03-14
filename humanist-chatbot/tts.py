import pyttsx3

def speak(text):
    print("Assistant:", text)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()