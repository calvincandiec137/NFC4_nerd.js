import speech_recognition as sr
import pyttsx3
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import os
from io import BytesIO
from pydub.playback import play

load_dotenv()

api_key=os.getenv('elevenlabs_api_key')

def fallback_tts(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def test(text):
    try:
        VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Rachel voice

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
            audio.save('C:\New folder\codes\NFC4\NFC4_nerd.js\audio_files')
            play(audio)
        elif response.status_code == 401:
            print("Error 401: Quota exceeded or invalid API key.")
            print(response.json()["detail"]["message"])
            fallback_tts(text)
        else:
            print(f"Error {response.status_code}:", response.text)
            fallback_tts(text)

    except Exception as e:
        print("Exception occurred:", e)
        fallback_tts(text)

if __name__=='__main__':
    test("Hello world")