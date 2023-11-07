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
import os
from collections import defaultdict
import json
from tqdm.auto import tqdm
from promptUtils import *

@dataclass
class EvalArguments():
    model_name_or_path: str = field(metadata={"required":True, "help": "Name of the hf model name or path for the model, tokenizer and config"})
    model_context_size: int = field(default=None,metadata={"help": "Maximum model context size used during training"})
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for evaluation (full attention will still be used)."},
    )

    temperature: float = field(default=0.6, metadata={"help": "Temperature parameter for generation (higher values for more randomness, lower for more determinism)"})
    max_length: int = field(default=None, metadata={"help": "Maximum number of charachters that the model should handle (prompt+answer), defaults to the model original context size"})
    top_p: int = field(default=0.9, metadata={"help": "Top-p sampling parameter (higher values for more randomness)"})

    generate_repetition_number: int = field(default=1, metadata={"help": "How many times to repeat the generation for each example (may be usefull in case of higher temperatures due to non-determinism)."})

    output_save_path: str = field(default = None, metadata={"help": "File name to be used for saving output data json."})

    eval_data_path: str = field(default = None,metadata={"required":True, "help": "Path to the evaluation data (validation set)."})
    split_name: str = field(default='validation', metadata={"help": "Dataset split name to be used for the evaluation."})

    prompt_config: str = field(default=None,metadata={"help": "Filename containing the prompt information, will superseed following prompt commands. This will allow to use also different prompt templates as long as they keep the standard placeholders system_prompt and instruction."})
    instruction: str = field(default=None, metadata={"required":True, "help" : "(limited for now) Prompt to be used for the data/user instruction. It may include some placeholders that need to have the same name of the dataset columns they intend to replace. When multiple prompts are given a column named prompt_idx containing the index of the prompt to be used (integer) is required in the data."})
    system_prompt: str = field(default=LLAMA_SYSTEM_PROMPT, metadata={"help" : "Prompt to be used as a system prompt to define the model behaviour."})

    target_column: str = field(default='output', metadata={"help" : "column to be used as the target for the prediction."})

    load_in_4bit: bool = field(default=False, metadata={"help" : "Whether to load the model in 4 bit to reduce memory usage (this should not affect inference performance)"})


def load_metrics_and_get_eval_function(tokenizer):
    # this cell may take long sometimes for some reason 
    rouge = evaluate.load("rouge")

    def compute_metrics_and_get_summary(model_output, input_char_count, expected_output):

        # convert tensor to string
        predictions = tokenizer.decode(model_output.to('cpu')[0], skip_special_tokens=True)
        predictions = predictions[input_char_count:] # keep only answer tokens
        
        metrics = {}
        metrics.update(rouge.compute(predictions=[predictions], references=[expected_output]))
        return metrics, predictions
    
    return compute_metrics_and_get_summary


def prepare_dataset(dataset, prompt_template, system_prompt, user_prompt, target_column):
    
    assert type(user_prompt) == str, "single prompt only sopported for now" # possible todo, but not used for now
    user_prompt_placeholders = extract_placeholer_names(user_prompt)
    def format_example(ex):
        instruction = user_prompt.format_map({ph:ex[ph] for ph in user_prompt_placeholders})
        ex["input"] = prompt_template.format(system_prompt=system_prompt, instruction=instruction)
        ex["output"] = ex[target_column]

        return ex
    
    dataset = dataset.map(format_example)
    dataset = dataset.select_columns(['input', 'output', 'is_camera', 'id', 'reference', 'summary'])

    return dataset


def get_average_metrics(metrics, exclude_set = set()):
    average_metrics = {}
    for metric_name, metric_value in metrics.items():
        if metric_name in exclude_set:
            continue
        average_metrics[metric_name] = sum(metric_value)/len(metric_value)
    return average_metrics

def main():
    parser  = transformers.HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclasses()[0]

    logging.debug(args)

    if not args.output_save_path:
        logging.warning("no output file was indecated, output results won't be stored")
        store_results = False
    else:
        assert not os.path.exists(args.output_save_path), "save path already exists, please specify another name"
        store_results = True

    if not args.max_length:
        args.max_length = args.model_context_size # infer the number of new tokens from the model context size

    if args.prompt_config:
        prompt_template, system_prompt, user_prompts = read_prompt_config(args.prompt_config)

        assert len(user_prompts) == 1, 'only single instruction mode is supported at the moment'
        instruction  = user_prompts[0]
    else:
        prompt_template = LLAMA_CHAT_PROMPT
        system_prompt = args.system_prompt
        instruction = args.instruction
    
    assert set(extract_placeholer_names(prompt_template)) == set('system_prompt', 'instruction'), 'prompt template must include all and only the following placeholders: {system_prompt}, {instruction}'


    if args.use_flash_attn:
        replace_llama_attn(inference=True)

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

            logging.warning(f'changing rope scaling factor due to the model being ')

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
        model_max_length=args.model_context_size if args.model_context_size >= orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=True,
    )

    model.eval()

    # perplexity and loss would not be evaluated since they would require the model to
    # be used as in training but this would intrinsically limit the model generation
    # as (I think) it will be a sort of a teacher forcing appoach
    compute_metrics_and_get_summary = load_metrics_and_get_eval_function(tokenizer)

    metrics = defaultdict(lambda : [])
    generated_texts = {}

    eval_ds = datasets.load_dataset(args.eval_data_path, split=args.split_name)
    eval_ds = prepare_dataset(eval_ds, prompt_template, system_prompt, instruction, args.target_column)

    for i in tqdm(range(0, len(eval_ds))):

        example = eval_ds[i]
        example_id = example['id']

        formatted_input = example['input']
        expected_output = example['output']
        
        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
        
        if store_results:
            generated_texts[example_id] = {
                'is_camera': example['is_camera'],
                'reference': example['reference'],
                'original_summary': example['summary'],
                'generated_summaries': [],
            }

        for _ in range(args.generate_repetition_number):
            output = model.generate(
                inputs['input_ids'],
                temperature = args.temperature,
                max_length = args.max_length,
                top_p = args.top_p)

            iteration_metrics, summary = compute_metrics_and_get_summary(output, len(formatted_input), expected_output)
            
            if store_results:
                generated_texts[example_id]['generated_summaries'].append(summary)            

            for m_name, m_value in iteration_metrics.items():
                metrics[m_name].append(m_value)
        
        del output # free memory up

    average_metrics = get_average_metrics(metrics)
    metrics['average_metrics'] = average_metrics

    if store_results:
        output_data={
            'model_name': args.model_name_or_path,
            'metrics': metrics,
            'prompting':
                {
                    'prompt_template': prompt_template,
                    'system_prompt': system_prompt,
                    'user_prompt': instruction,
                },
            'data': generated_texts,
        }

        with open(args.output_save_path, 'w') as f:
            logging.info(f'writing results to {args.output_save_path}')
            json.dump(output_data, f)

    print(f'average metrics:\n{average_metrics}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()