# app.py
import sys
sys.path.append('../Efficient_RedPajama_Finetuning')
from flask import Flask, render_template, redirect, url_for, jsonify
import inference_example
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

checkpoint = "../Efficient_RedPajama_Finetuning/outputs/checkpoint-300"

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1", load_in_8bit=True,  device_map={"":0})

# Load the Lora model
model = PeftModel.from_pretrained(model, checkpoint, device_map={"":0})
model.eval()


@app.route('/')
def home():
    # Render the 'home.html' template
    return render_template('index.html')

@app.route('/handle_button_click', methods=['POST'])
def handle_button_click():
    # Handle the button click and return a response
    #response_text = "You clicked the button!"
    input_ids = inference_example.get_input_sample(tokenizer).unsqueeze(0)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=True, top_p=0.9)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(ans)
    ans = ans.replace('Answer the following question: write me a good dad joke', '')
    return jsonify({'message': ans})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

