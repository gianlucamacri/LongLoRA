# Written by Yukang Chen
# Some code based on https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Union
from string import Formatter

import torch
import torch.nn as nn
import transformers
import evaluate
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from llama_attn_replace_sft import replace_llama_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


# unused (?)
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# args classes
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    hf_dataset: str = field(default=None, metadata={"help": "Name of the huggingface dataset to be used, overwrites data_path and eval_data_path."})
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data (validation set)."})
    prompts: List[str] = field(default_factory=lambda : ["{instruction}"], metadata={"nargs":"+", "help" : "Prompt(s) to be used for the data. It may include some placeholders that need to have the same name of the dataset columns they intend to replace. When multiple prompts are given a column named prompt_idx containing the index of the prompt to be used (integer) is required in the data."})
    prompts_are_fn: bool = field(default=False, metadata={"help" :"Whether to interpret the prompts as filenames conataining the actual prompts."})
    target_column: str = field(default='output', metadata={"help" : "column to be used as the target for the prediction"})
    max_prompt_token_count: int = field(default=None, metadata={"help" : "Max number of token allowed by the dataset, preforms a sanity check informing the user"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            pad_to_multiple_of=4
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def get_prompts(prompts:List[str], prompts_are_fn:bool):
    if prompts_are_fn:
            actual_prompts = []
            for prompt_fn in prompts:
                with open(prompt_fn, 'r') as f:
                    actual_prompts.append(f.read())
            prompts = actual_prompts
    prompts = [prompt.replace('\\n', '\n') for prompt in prompts]
    return prompts

def get_prompts_max_token_len(prompts, tokenizer):
    prompt_token_lengths = [len(tokenizer.tokenize(prompt)) for prompt in prompts]
    return max(prompt_token_lengths)

class SupervisedDatasetWithPrompts(Dataset):
    """Dataset for supervised fine-tuning with prompts."""

    def __init__(self, data: Union[str, List[Dict[str,str]]], tokenizer: transformers.PreTrainedTokenizer, prompts: List[str], target_column: str):
        super(SupervisedDatasetWithPrompts, self).__init__()
        
        logging.info("Loading data...")
        if type(data) is str:
            assert data.split('.')[-1] == 'json', "data path must have json extension"
            list_data_dict = jload(data)
        elif type(data) is list:
            list_data_dict = data
        else:
            raise Exception(f"unsupported data type: {type(data)}")

        logging.info("Formatting inputs...")
        single_prompt = len(prompts) == 1
        assert single_prompt or 'prompt_idx' in list_data_dict[0].keys(), 'prompt_idx must be a column in the data when multiple prompts are used'
        field_names_for_prompt = [[v[1] for v in Formatter().parse(prompt) if v[1]] for prompt in prompts]
        logging.info(f"found the following field names in prompts: {field_names_for_prompt}")

        def format_example(example):
            """ format a prompt according to the right prompt index if needed"""
            prompt_idx = 0
            if not single_prompt:
                prompt_idx = example['prompt_idx']
            prompt = prompts[prompt_idx]
            prompt_field_names = field_names_for_prompt[prompt_idx]
            format_dict = {field_name:example[field_name] for field_name in prompt_field_names}
            return prompt.format(**format_dict)

        sources = [format_example(example) for example in list_data_dict]
        targets = [f"{example[target_column]}{tokenizer.eos_token}" for example in list_data_dict]
        
        logging.info("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = None
    eval_data = None

    prompts = get_prompts(data_args.prompts, data_args.prompts_are_fn)
    max_prompt_len = get_prompts_max_token_len(prompts, tokenizer)

    if data_args.max_prompt_token_count:
        if max_prompt_len <= data_args.max_prompt_token_count:
            logging.info('prompt max length under the given threshold')
        else:
            logging.warning(f'Prompt max length grater than threshold ({max_prompt_len} > {data_args.max_prompt_token_count}) !')

    # data/datapath creation
    if data_args.hf_dataset:
        dataset_dict = load_dataset(data_args.hf_dataset)
        train_data = dataset_dict['train'].to_list()
        if 'validation' in dataset_dict.keys():
            logging.info('validation split found')
            eval_data = dataset_dict['validation'].to_list()
        elif 'test' in dataset_dict.keys():
            logging.warning('using test split for the evaluation since no eval split was found')
            eval_data = dataset_dict['test'].to_list()
    else:
        train_data = data_args.data_path
        if data_args.eval_data_path:
            eval_data = data_args.eval_data_path

    
    # dataset and collator creation
    train_dataset = SupervisedDatasetWithPrompts(tokenizer=tokenizer, data=train_data, prompts=prompts, target_column=data_args.target_column)
    if eval_data:
        eval_dataset = SupervisedDatasetWithPrompts(tokenizer=tokenizer, data=eval_data, prompts=prompts, target_column=data_args.target_column)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    replace_llama_attn(training_args.use_flash_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        #device_map="auto", # use only if model does not fint on a single gpu and lauch with python, less time efficient 
        torch_dtype=torch.float16, # float16 is the default, but float32 seem to behave a little better and avoid mismatch with lora float32
        quantization_config=BitsAndBytesConfig(
        #    load_in_8bit=True,
        #    llm_int8_threshold=6.0,
        #    llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        trust_remote_code=True
    )

   
    for name, module in model.named_modules():
        module.requires_grad = False  # freeze the model
        if "norm" in name or "embed" in name: # added embed and norm since they both gets trained  
            module = module.to(torch.float32)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
            #return super().forward(x).to(torch.float16)
        

    model.lm_head = CastOutputToFloat(model.lm_head)

    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
    

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # here compute_metrics function will be None, enablin just the loss evaluation, using other functions
    # caused memory overflow for some reason, I thought it may be due to eval_accumulation_steps not being set
    # but in that case there seem to be another issue that couses the crash
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print(f'training args: {trainer.args}')
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
