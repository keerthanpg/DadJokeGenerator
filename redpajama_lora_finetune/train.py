

import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import fire
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft import prepare_model_for_int8_training
from transformers import Trainer
import utils
from utils import smart_tokenizer_and_embedding_resize, make_supervised_data_module
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

### FEEL FREE TO CHANGE THIS OR ADD MORE PARAMETERS AS YOU SEE FIT!
### This dataclass is used to store the configuration for the trainer.
@dataclass
class TrainerConfig:
    batch_size: int = 12
    micro_batch_size: int = 3
    lora_r: int = 8  #  This controls the rank of the matrices used in LoRA
    lora_alpha: int = 16 # Hyperparameter for LoRA training
    lora_dropout: float = 0.05
    warmup_steps: int = 5
    learning_rate: float = 3e-4
    epochs: int = 1
    steps_per_checkpoint: int = 50
    steps_per_eval: int = 20
    steps_per_log: int = 4
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    output_dir = "outputs/"
    pretrained_model_path ="togethercomputer/RedPajama-INCITE-Base-3B-v1"
    use_wandb : bool = True
    wandb_project : str = 'AnythingButWrappersExamples'
    wandb_name : str = 'RedPajama-LoRA-Training'


class CustomCheckpointTrainer(Trainer):
    """
    We use this custom trainer to save checkpoints to a new directory every time
    """
    def __init__(self, *args, checkpoint_root_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_root_dir = checkpoint_root_dir or self.args.output_dir
        self.checkpoint_counter = 0

    def save_model(self, output_dir=None, **kwargs):
        if output_dir is None:
            # Create a new checkpoint directory
            output_dir = os.path.join(self.checkpoint_root_dir, str(self.checkpoint_counter))
            self.checkpoint_counter += 1

        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Call model.save_pretrained to save the checkpoint
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


def train(config):

    # Configuration Stuff
    cfg = config
    gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if cfg.use_wandb:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        os.environ["WANDB_NAME"] = cfg.wandb_name

    if ddp:
        print(f"Using DDP with {world_size} GPUs")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_path, load_in_8bit=True, device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)
    print(f"tokenizer.pad_token: {tokenizer.pad_token}")
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=utils.DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    train_dataset, eval_dataset, data_collator = make_supervised_data_module(tokenizer=tokenizer, train_data_path=cfg.train_data_path, eval_data_path=cfg.eval_data_path)


    ## Prepare the model for int8 training and get the PEFT model
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create the trainer    

    tokenizer.pad_token_id = 0  
    trainer = CustomCheckpointTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            fp16=True,
            logging_steps=cfg.steps_per_log,
            evaluation_strategy="steps",
            eval_steps=cfg.steps_per_eval,
            save_steps=cfg.steps_per_checkpoint,
            output_dir=cfg.output_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            report_to="wandb" if cfg.use_wandb else None,
        ),
        checkpoint_root_dir=cfg.output_dir,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model.config.use_cache = False


    # Begin training
    trainer.train()
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)



if __name__ == "__main__":
    cfg = TrainerConfig()
    train(cfg)