import json
from string import Formatter
import logging

LLAMA_CHAT_PROMPT =(
        "[INST] <<SYS>>\n"
        "{system_prompt}\n"
        "<</SYS>> \n\n {instruction} [/INST]"
)

LLAMA_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

def read_prompt_config(prompt_config_fn):

    with open(prompt_config_fn, 'r') as f:
        prompt_config = json.load(f)

    try:
        prompt_template = prompt_config['prompt_template']
    except KeyError:
        logging.warning(f'"prompt_template" key missing, using llama default value as a fall back: {LLAMA_CHAT_PROMPT}')
        prompt_template = LLAMA_CHAT_PROMPT

    try:
        system_prompt = prompt_config['system_prompt']
    except KeyError:
        logging.warning(f'"system_prompt" key missing, using llama default value as a fall back: {LLAMA_SYSTEM_PROMPT}')
        prompt_template = LLAMA_SYSTEM_PROMPT

    user_prompts = prompt_config['user_prompst']

    return prompt_template, system_prompt, user_prompts


def write_prompt_config(out_fn, prompt_template, system_prompt, user_prompts):

    prompt_config = {
        'prompt_template':prompt_template,
        'system_prompt':system_prompt,
        'user_prompts':user_prompts,
    }

    with open(out_fn, 'x') as f:
        json.dump(f, prompt_config)


def extract_placeholer_names(prompt):
    def extract_placeholder_names_single(s):
        return [v[1] for v in Formatter().parse(s) if v[1]]

    if type(prompt) == list:
        return [extract_placeholder_names_single(el) for el in prompt]
    else:
        return extract_placeholder_names_single(prompt)



def get_max_prompt_token_count(prompt, tokenize_f):
    """
    prompt may be a single prompt or a list of prompts (i.e. strings possibly with placeholders)
    tokenize_f is a function that takes a single prompt and returns the corresponding tokens only
    """

    if type(prompt) != list:
        prompt = [prompt]

    empty_formatted_prompts = [p.format_map({k:'' for k in extract_placeholer_names(p)}) for p in prompt]
    prompt_token_lengths = [len(tokenize_f(ep)) for ep in empty_formatted_prompts]
    return max(prompt_token_lengths)
    

        


