# app.py
import sys
sys.path.append('../redpajama_lora_finetune')
from flask import Flask, render_template, redirect, url_for, jsonify
import inference_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import requests
from uuid import uuid4
import json
import time
import os

app = Flask(
  __name__,
  static_url_path='',
  static_folder='static'
)

checkpoint = "../redpajama_lora_finetune/outputs/checkpoint-300"

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1", load_in_8bit=True,  device_map={"":0})

# Load the Lora model
model = PeftModel.from_pretrained(model, checkpoint, device_map={"":0})
model.eval()
session = requests.Session()

def authenticate():
    ljson={"username_or_email":os.environ['FAKEYOUUSERNAME'],"password":os.environ['FAKEYOUPASSWD']}
    lrjson = session.post("https://api.fakeyou.com/"+"login",json=ljson)
    print(lrjson.json())

def text2biden(text):
    biden = 'TM:wsvak9gwrdqf'
#     biden = 'TM:dt5b38hv3yjf' # angry
    base_url = "https://storage.googleapis.com/vocodes-public"
    
    uuid = str(uuid4())
    inference_job = session.post("https://api.fakeyou.com/tts/inference",
                    headers={'Accept': 'application/json', 'Content-Type': 'application/json'},
                    data=json.dumps({"uuid_idempotency_token": uuid,"tts_model_token": biden,"inference_text":text}))
    inference_job = inference_job.json()
    print(inference_job)
    while True:
        status = requests.get(f"https://api.fakeyou.com/tts/job/{inference_job['inference_job_token']}",
                        headers={'Accept': 'application/json'})
        status = status.json()
        if status['state']['status'] != "complete_success":
            print(status)
            time.sleep(1)
            print("retrying")
            continue
        else:
            break
    return f"{base_url}{status['state']['maybe_public_bucket_wav_audio_path']}"


@app.route('/')
def home():
    # Render the 'home.html' template
    return render_template('index.html')

@app.route('/handle_button_click', methods=['POST'])
def handle_button_click():
    # Handle the button click and return a response
    #response_text = "You clicked the button!"
    input_ids = inference_utils.get_input_sample(tokenizer).unsqueeze(0)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=True, top_p=0.9)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(ans)
    ans = ans.replace('Answer the following question: write me a good dad joke', '')
    url = "whatever"
    authenticate()
    url = text2biden(ans)
    print(url)
    return jsonify({'message': ans, 'url':url})

if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')

