from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json


def get_input_sample(tokenizer, eval_path = "data/eval.jsonl"):
    # with open(eval_path, "r") as f:
    #     line = json.loads(f.readline())
    query = "Answer the following question: write me a good dad joke"
    #query = "write me a good dad joke"
    tokenized_query = tokenizer(query, return_tensors="pt").input_ids[0].to("cuda:0")
    return tokenized_query

# if __name__ == "__main__":
#     run_inference("./outputs/checkpoint-5900")


