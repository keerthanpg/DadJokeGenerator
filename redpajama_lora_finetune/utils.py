import json
from typing import Sequence, Dict, List, Any
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
import transformers
import torch

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def _tokenize_fn(strings: Sequence[str],
                 tokenizer) -> List[torch.Tensor]:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
        ) for text in strings
    ]
    labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    return labels

def preprocess_paths(sources: Sequence[Dict[str, Any]],
                        tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
    """Preprocess paths for training."""
    input_ids, targets = [], []

    prefix = "Answer the following question: "
    tokenized_prefix = tokenizer(prefix, return_tensors="pt").input_ids[0]
    query_strings =  [source["input"] for source in sources]
    answer_strings = [source["output"] for source in sources]

    tokenized_query_strings = _tokenize_fn(query_strings, tokenizer)
    tokenized_answer_strings = _tokenize_fn(answer_strings, tokenizer)

    for i in range(len(tokenized_query_strings)):
        tokenized_intro = torch.cat([tokenized_prefix, tokenized_query_strings[i]])
        masked_intro = torch.tensor([IGNORE_INDEX] * len(tokenized_intro)).to(tokenized_intro.device)

        eos_token = torch.tensor([tokenizer.eos_token_id]).to(tokenized_prefix.device)
        tokenized_answer = torch.cat([tokenized_answer_strings[i], eos_token])

        curr_input = torch.cat([tokenized_intro, tokenized_answer])
        curr_target = torch.cat([masked_intro, tokenized_answer])

        if curr_input.shape[0] > 1024:
            continue

        input_ids.append(curr_input)
        targets.append(curr_target)
    
    return dict(input_ids=input_ids, labels=targets)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 debug: bool = False):
        super(SupervisedDataset, self).__init__()
        self.debug = debug

        sources = self.get_sources(data_path)
        data_dict = preprocess_paths(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def get_sources(self, data_path: str):
        if data_path.endswith(".jsonl"):
            list_data_dict = [json.loads(line) for line in open(data_path, "r")]
        else:
            list_data_dict = json.load(open(data_path, "r"))
        if self.debug:
            list_data_dict = list_data_dict[:400]
        return list_data_dict

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def select(self, indices: List[int]):
        self.input_ids = [self.input_ids[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                train_data_path,
                                eval_data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_cls = (LazySupervisedDataset
    #                if lazy_preprocess else SupervisedDataset)
    dataset_cls = SupervisedDataset
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=train_data_path)
    
    eval_dataset = dataset_cls(tokenizer=tokenizer,
                                 data_path=eval_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, eval_dataset, data_collator


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# def main():
#     tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
#     pdb.set_trace()
#     data_path = '/workspace/AnythingButWrappers/Efficient_RedPajama_Finetuning/data.jsonl'
#     SupervisedDataset(data_path, tokenizer, debug=True)

# if __name__ == "__main__":
#     main()


