from pathlib import Path
import os
import semantic_kernel as sk 
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

input_filename = "sample-1.mp3"

with open(Path.cwd()/input_filename,"rb") as audio_file:
  transcript = openai.Audio.transcribe(
    model="whisper-1",
    file=audio_file
  )
  print(transcript)

kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

prompt_prefix = """You are an librarian assistant, tasked with providing book recommendations. 
You will be prompted with a text, and based on that you will come with a book recommendation
that fits this text. Explain how you got to this recommendation, and provide at least 2 alternative
recommendations, also explain why you chose those.
"""

prompt = f"{prompt_prefix}" + "{{$input}}" + "Output:\n"
recommend = kernel.create_semantic_function(prompt, max_tokens=2048, temperature=1.2)

recommendation = recommend(transcript.text)

print(recommendation)

