import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextIteratorStreamer
from llama_attn_replace_sft import replace_llama_attn
from threading import Thread
import gradio as gr
import logging


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    return args

title = "LongLoRA and LongAlpaca for Long-context LLMs"


SYSTEM_PROMPT = (
    #"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    #"If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    "You are a helpful, respectful and honest assistant specialized in writing dossiers for legislative documents. Always answer as helpfully as possible, while being safe. Your answers should not include any false information with respect to the original document."
)

PROMPT_DICT = {
    "prompt_no_input_llama2":(
        "<s>[INST] <<SYS>>\n"
        "{system_prompt}"
        "<</SYS>> \n\n {instruction} [/INST]"
    )
}

prompt_no_input_base = "Di seguito c'è un documento legislativo in italiano. Memorizza il documento legislativo, poi segui le istruzioni.\nINIZIO DOCUMENTO.\n{documento}\nFINE DOCUMENTO.\nScrivi un dossier riassuntivo del documento. Scrivi in italiano, non scrivere in inglese."

#prompt_no_input_base ="""Di seguito c'è un documento legislativo. Memorizza il documento legislativo, poi segui le istruzioni.
#Inizio documento.
#{documento}
#Fine documento.
#Sintetizza in modo completo il contenuto del documento legislativo usando la stessa lingua in cui è stato scritto.
#Sintesi: """

description = """
<font size=4>
</font>
"""

# Gradio
article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2308.00692' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='https://github.com/dvlab-research/LongLoRA' target='_blank'>   Github Repo </a></p>
"""



def read_txt_file(material_txt):
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def build_generator(
    model, tokenizer, use_cache=True
):
    def response(material, document, prompt_template, system_prompt, temperature, top_p, max_gen_len):
        #if material is None:
        #    return "Only support txt file."
#
        #if not material.name.split(".")[-1]=='txt':
        #    return "Only support txt file."
#
        #material = read_txt_file(material.name)
        #prompt_no_input = PROMPT_DICT["prompt_no_input"]


        if not prompt_template or prompt_template.strip() == '':
            prompt_template = prompt_no_input_base

        if not system_prompt or system_prompt.strip() == '':
            system_prompt = SYSTEM_PROMPT
        
        user_prompt = prompt_template.format_map({"documento": document})
        prompt = PROMPT_DICT["prompt_no_input_llama2"].format(system_prompt = system_prompt, instruction=user_prompt)

        logging.info(f'input: {prompt[:300]} ... {prompt[-300:]}')

        #print(prompt[:1000])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if len(inputs['input_ids'][0]) > 32768:
            return "This demo supports tokens less than 32768, while the current is %d. Please use material with less tokens."%len(inputs['input_ids'][0])
        torch.cuda.empty_cache()
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(**inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
            )

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
        return generated_text

    return response

def main(args):
    print(args)
    #if args.flash_attn:
    #    print('replacing attention')
    #    replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        #cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        #cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        #cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    #model.resize_token_embeddings(32001) # add pad token maybe (?)
    model.resize_token_embeddings(len(tokenizer)) # version used in the sft code

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    respond = build_generator(model, tokenizer, use_cache=True)


    demo = gr.Interface(
        respond,
        inputs=[
            gr.File(type="file", label="Document txt", visible=False),
            gr.Textbox(lines=1, placeholder=None, label="Document"),
            gr.Textbox(lines=1, placeholder = prompt_no_input_base, value=prompt_no_input_base, label="Prompt"),
            gr.Textbox(lines=1, placeholder = SYSTEM_PROMPT, value=SYSTEM_PROMPT, label='System Prompt'),
            gr.Slider(minimum=0.001, maximum=1.0, value=args.temperature, label='Temperature'),
            gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=args.top_p, label="Top p"),
            gr.Slider(minimum=1, maximum=32768, step=1, value=args.max_gen_len, label="Max gen. len."),
        ],
        outputs=[
            gr.Textbox(lines=1, placeholder=None, label="Text Output"),
        ],
        title=title,
        description=description,
        article=article,
        allow_flagging="auto",
    )

    demo.queue()
    demo.launch(show_error=True, share=True)

if __name__ == "__main__":
    args = parse_config()

    #args = {
    #    'base_model': 'Yukang/LongAlpaca-7B',
    #    'context_size': 8192,
    #    'flash_attn': False,
    #    'temperature': 0.6,
    #    'top_p': 0.9,
    #    'max_gen_len': 4096, 
    #}
                        
    main(args)
