import os
import base64
import requests
from typing import List, Optional, Union
import json
import tempfile
from PIL import Image
from enum import Enum

class UserMessageTypes(str, Enum):
    Text = "text"
    ImagePathHigh = "img_path_high"
    UrlHigh = "url_high"
    ImagePathLow = "img_path_low"
    UrlLow = "url_low"


class UserMessage:
  def __init__(self):
    self._content:List[str] = []
    self._types:List[str]= []

  def encode_image(self, image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  def add_text(self, string:str):
    self._content.append(string)
    self._types.append(UserMessageTypes.Text)

  def add_img_path(self, path:str, high_detail=True):
    self._content.append(self.encode_image(path))
    self._types.append(UserMessageTypes.ImagePathHigh if high_detail else UserMessageTypes.ImagePathLow)

  # saves a temp file
  def add_img_array(self, array, high_detail=True):

    # Create a temporary file with a random name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file: # need delete = False to avoid permission issues..
      image = Image.fromarray(array)
      temp_file_path = temp_file.name
      image.save(temp_file_path)
      self.add_img_path(temp_file_path, high_detail=high_detail)
    os.remove(temp_file_path)

  def add_img_url(self, url:str, high_detail=True):
    self._content.append(url)
    self._types.append(UserMessageTypes.UrlHigh if high_detail else UserMessageTypes.UrlLow)

  def get_all_text_str(self):
    strr = ""
    for s, t in zip(self._content, self._types):
      if t == UserMessageTypes.Text:
        strr = strr + s

  def get_content_openai(self):
    to_return = []
    for type, string in zip(self._types, self._content):
      if type == UserMessageTypes.Text:
        to_return.append({
          "type": "text",
          "text": string,
        })
      elif type in [UserMessageTypes.UrlLow, UserMessageTypes.UrlHigh]:
        if type == UserMessageTypes.UrlHigh:
          detail = "high"
        else:
          detail = "low"
        to_return.append({
          "type": "image_url",
          "image_url": {
            "url": string,
            "detail": detail
          }
        })
      elif type in [UserMessageTypes.ImagePathLow, UserMessageTypes.ImagePathHigh]:
        if type == UserMessageTypes.ImagePathHigh:
          detail = "high"
        else:
          detail = "low"
        to_return.append( {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{string}",
            "detail": detail
          }})

    return to_return


class ImageChatHistory:
  def __init__(self):
    self.messages = []

  def clear(self):
    self.messages = []

  def add_system_msg(self, content: str):
    assert len(self.messages) == 0, "System should come first (I'm pretty sure, maybe it doesn't matter)"
    self.messages.append({"role": "system", "content": "<|ENDOFTEXT|>" + content})

  def add_user_msg(self, user_message:Union[str,UserMessage]):
    if type(user_message) == str:
      user_message_new = UserMessage()
      user_message_new.add_text(user_message)
      user_message = user_message_new
    self.messages.append({"role": "user", "content": user_message.get_content_openai()})

  def add_assistant_msg(self, content_str: str):
    self.messages.append({"role": "assistant", "content": [{"type": "text",
                                                         "text": content_str
    }]})

  def __repr__(self):
    strr = "::: Conversation History :::\n"
    for message in self.messages:
      #print(message)
      role = message['role']
      if role == "system":
        strr = strr + role + ":\n" + message['content'] + "\n"
      else:
        content = ""
        for c in message['content']:
          print(c['type'])
          if c['type'] == "text":
            print("ADDED")
            content = content + c['text'] + "\n"
          elif c['type'] == "image_url":
            content = content + c['image_url']['url'][0:100] + "..." + "\n"
        strr = strr + role + ":\n" + content + "\n"
    return strr

  def __str__(self):
    return self.__repr__()


def get_streaming_responses(streaming_response):
  buffer = ""  # Initialize a buffer to accumulate chunks
  for byte_chunk in streaming_response:
    buffer += byte_chunk.decode('utf-8')  # Decode bytes to string and accumulate
    while '\n' in buffer:
      line, buffer = buffer.split('\n', 1)  # Split at the first newline
      line = line.strip()  # Strip whitespace
      if line.startswith('data:'):  # Check if the line contains JSON data
        line = line[5:].strip()  # Remove the 'data:' prefix
        try:
          chunk_dict = json.loads(line)  # Attempt to decode the JSON
          if 'choices' in chunk_dict and chunk_dict['choices']:
            message_content = chunk_dict['choices'][0].get('delta', {}).get('content', {})
            if message_content:
              yield message_content
        except json.JSONDecodeError:
          continue  # Continue if JSON is incomplete or malformed


#TODO there are more options to expose here. Tool use is especially interesting https://platform.openai.com/docs/api-reference/chat/create
def call_model(history:ImageChatHistory,
               stream=False,
               temperature:float=1.0, # 1 is openai default, value between 0-2 see also top_p in https://platform.openai.com/docs/api-reference/chat/create
               model:str="gpt-4o", # this is the vision model
               max_response_tokens:int=4096,#this is max for gpt-4-turbo as of 10/05/2024
               api_key:Optional[str]=None, # if not provided will use env variable OPENAI_API_KEY
               ogranization_id:Optional[str]=None # if not provided will use env variable 
               ):
  api_key = api_key if api_key is not None else os.getenv('OPENAI_API_KEY')
  ogranization_id = ogranization_id if ogranization_id is not None else os.getenv('OPENAI_ORGANIZATION_ID')

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }
  if ogranization_id is not None:
    headers["OpenAI-Organization"] = ogranization_id

  payload = {
    "model": model,
    "messages": history.messages,
    "max_tokens": max_response_tokens,
    "temperature": temperature,
    "stream": stream,
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, stream=stream)

  if not stream:
    if 'choices' in response.json():
      text = response.json()['choices'][0]['message']['content']
    else:
      text = response.json()['error']
    return text
  else:
    return get_streaming_responses(response)



if __name__ == "__main__":
  ########
  # assuming  OPENAI_API_KEY is set in .env can also manually set it in env or pass it into call_model
  from dotenv import load_dotenv
  load_dotenv(override=True)
  #########


  example_history = ImageChatHistory()

  # always should add a system message
  example_history.add_system_msg("Do what the users says")

  # Then add a user msg
  usr_msg = UserMessage()
  # options add_text, add_img_path, add_img_url. One message can have multiple!
  usr_msg.add_text("hello, if I give you an image tell me what's in it")


  example_history.add_user_msg(usr_msg)
  # you can even add assistant messages (not needed for the basic demo)
  example_history.add_assistant_msg("Certainly!")

  new_usr_msg = UserMessage()
  new_usr_msg.add_text("here's an image, answer in the style of shakespeare")
  # can also do add_img_path for local path and add_image_array
  new_usr_msg.add_img_url("https://upload.wikimedia.org/wikipedia/commons/7/77/Big_Nature_%28155420955%29.jpeg", high_detail=True)
  
  # there is also an option add_img_path
  example_history.add_user_msg(new_usr_msg)

  #call the model like so (see args above!)
  model_message = call_model(
    example_history,
    stream=False,

  )

  print(model_message)
  #NOTE if you are doing a chat, make sure to add the message to history
  #example_history.add_assistant_msg(model_message)


  # can also do streaming (I should add something that captures the entire text to send to example_history, also need to do async call
  stream_response = call_model(
    example_history,
    stream=True,

  )

  for chunk in stream_response:
    print(chunk)

