# Dad Joke Generator

This is code + flask app for finetuning RedPajama 3B model on a Kaggle dataset of 200000 dadjokes from Reddit. The app runs inference on a checkpoint to generate a joke and then renders it in Joe Biden's voice.

## Download the dataset
Download the dataset [here](https://www.kaggle.com/datasets/oktayozturk010/reddit-dad-jokes)
Then run `create_dataset.py` in redpajama_lora_finetune/data/ folder to create `train.jsonl` and `eval.jsonl`

## Install Requirements

`pip install -r requirements.txt`


## Finetune Redpajama
To run finetuning `CUDA_SET_VISIBLE_DEVICES=0 python train.py`

## Inference 
Access the app [here](http://papajokes.keerthanapg.com/) 

This was built at [Any Thing But Wrappers Hackathon](https://www.anythingbutwrappers.com/). 
