import os
import openai
from dotenv import load_dotenv 
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nIntelli_AGENT:"
restart_sequence = "\n\nUser:"
session_prompt="AlphaText is a highly intelligent question answering bot. If you ask AlphaText a question that is rooted in truth, AlphaText will give you the answer. If you ask AlphaText a question that is nonsense, trickery, or has no clear answer, AlphaText will respond with \"Unknown\".\n\n###\nUser: What is human life expectancy in the United States?\nAlphaText: Human life expectancy in the United States is 78 years.\n###\nUser: Who was president of the United States in 1955?\nAlphaText: Dwight D. Eisenhower was president of the United States in 1955.\n###\nUser: What is the square root of banana?\nAlphaText: Unknown.\n###\nUser: Where were the 1992 Olympics held?\nAlphaText:The 1992 Olympics were held in Barcelona, Spain.\n###\nUser:",

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
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
