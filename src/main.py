from pathlib import Path
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

files = ["lost_debit_card.wav","sample-1.mp3","my_audio.wav"]

for input_filename in files:
  print(input_filename)
  with open(Path.cwd()/input_filename,"rb") as audio_file:
    transcript = openai.Audio.transcribe(
      model="whisper-1",
      file=audio_file
    )
    print(transcript)
     

  print("- "*40)