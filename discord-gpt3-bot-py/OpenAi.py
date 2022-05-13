import os
import openai
from dotenv import load_dotenv 
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nAlphaText:"
restart_sequence = "\n\nUser:"

# Type in prompt on the screen if you want 
session_prompt="",

def ask(input):
    inputs = input.split("Question:")
    context = inputs[0]
    question = inputs[1]
    prompt_text = f"{context}\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
      model="ada:ft-personal-2022-05-07-22-50-48",
      prompt=prompt_text,
      temperature=0,
      max_tokens=30,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n"],
    )
    reply = response['choices'][0]['text']
    return str(reply)

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
