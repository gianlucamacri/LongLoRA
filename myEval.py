import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Union
from llama_attn_replace import replace_llama_attn
import math
import logging
import torch
import sys
import datasets
import evaluate
import numpy as np


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
)

PROMPT_DICT = {
    "prompt_no_input_llama2":(
        "<s>[INST] <<SYS>>\n"
        "{system_prompt}"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
}

@dataclass
class EvalArguments():
    model_name_or_path: str = field(metadata={"required":True, "help": "Name of the hf model name or path for the model, tokenizer and config"})
    model_context_size: int = field(default=None,metadata={"help": "Maximum model context size used during training"})
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for evaluation (full attention will still be used)."},
    )
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    
    temperature: float = field(default=0.6, metadata={"help": "Temperature parameter for generation (higher values for more randomness, lower for more determinism)"})
    max_length: int = field(default=None, metadata={"help": "Maximum number of charachter to be generated, defaults to the model original context size"})
    top_p: int = field(default=0.9, metadata={"help": "Top-p sampling parameter (higher values for more randomness)"})

    eval_data_path: str = field(metadata={"required":True, "help": "Path to the evaluation data (validation set)."})
    split_name: str = field(default='validation', metadata={"help": "Dataset split name to be used for the evaluation."})

    prompt_config: str = field(default=None,metadata={"help": "Filename containing the prompt information."})
    prompts: List[str] = field(default_factory=lambda : ["{instruction}"], metadata={"nargs":"+", "help" : "Prompt(s) to be used for the data. It may include some placeholders that need to have the same name of the dataset columns they intend to replace. When multiple prompts are given a column named prompt_idx containing the index of the prompt to be used (integer) is required in the data."})
    prompts_are_fn: bool = field(default=False, metadata={"help" :"Whether to interpret the prompts as filenames conataining the actual prompts."})
    target_column: str = field(default='output', metadata={"help" : "column to be used as the target for the prediction."})
    system_prompt: str = field(default=SYSTEM_PROMPT, metadata={"help" : "Prompt to be used as a system prompt to define the model behaviour."})

    load_in_4bit: bool = field(default=False, metadata={"help" : "Whether to load the model in 4 bit to reduce memory usage (this should not affect inference performance)"})

def load_metrics_and_get_eval_function(tokenizer):
    # this cell may take long sometimes for some reason 
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    rouge = evaluate.load("rouge")

    def compute_metrics_hf(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {}
        metrics.update(accuracy.compute(predictions=predictions, references=labels))
        #metrics.update(precision.compute(predictions=predictions, references=labels))
        #metrics.update(recall.compute(predictions=predictions, references=labels))
        #metrics.update(f1.compute(predictions=predictions, references=labels))
        predictions = tokenizer.decode(predictions, skip_special_tokens=True)
        metrics.update(rouge.compute(predictions=predictions, references=labels))
        return metrics
    
    return compute_metrics_hf


def main():
    parser = transformers.HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()

    if args.flash_attn:
        replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path
    )

    orig_rope_scaling = getattr(config, "rope_scaling",  None)
    if not orig_rope_scaling:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor

        if args.model_context_size and args.model_context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(args.model_context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

            # not sure wether this behavior makes sense or not
            logging.warning(f'changing rope scaling factor due to context size larger than trining')

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=True,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # todo get data, format dataset

    eval_ds = datasets.load_dataset(args.eval_data_path, split=args.split_name)

    for i in range(0, len(formatted_data), args.batch_size):
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p
        )

        out = tokenizer.decode(output[0], skip_special_tokens=True)

        ## remove prompt (I think)

        # evaluate metrics (perplexity, loss, accuracy, rouge,...)
    








if __name__ == "__main__":
    main()