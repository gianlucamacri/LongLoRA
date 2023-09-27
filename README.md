# LongLoRA (with GPTNeoX support): Efficient Fine-tuning of Long-Context Large Language Models

This repo provides on top of the original implementation, support for GPTNeoX with Flash-Attention and the LongLoRA's shifted short attention as needed.

## Highlights
**LongLoRA** speed up the context extension of pre-trained large language models in both attention-level and weight-level.
1. The proposed shifted short attention is easy to implement, compatible with Flash-Attention, and not required during inference. 


## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training
### Pre-trained weights
I used GPTNeoX model as the base model architecture, which was ported from the authors' original repo where Llama2 was used.
Some candidate pre-trained weights may include [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b), [Polyglot-ko-12.8B](https://huggingface.co/EleutherAI/polyglot-ko-12.8b) and other variants.


### Fine-tuning
```
torchrun --nproc_per_node=8 fine-tune.py  \
        --model_name_or_path path_to/gpt_neox_model_hf \
        --bf16 True \
        --output_dir path_to_saving_checkpoints       \
        --cache_dir path_to_cache \
        --model_max_length 8192 \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 1000     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage2.json" \
        --tf32 True \
        --max_steps 1000
```

- Please remember to change `path_to/gpt_neox_model_hf`, `path_to_saving_checkpoints`, `path_to_cache` to your own directory.
- Note that you can change `model_max_length` to other values.
- You could change `ds_configs/stage2.json` to `ds_configs/stage3.json` if you want.
- Please set `use_flash_attn` as `False` if you use V100 machines or do not install flash attention.
- You can set `low_rank_training` as `False` if you want to use fully fine-tuning. It will cost more GPU memory and slower, but the performance will be a bit better.
- When training is finished, to get the full model weight:
```
cd path_to_saving_checkpoints && python zero_to_fp32.py . pytorch_model.bin
```

### Supervised Fine-tuning
```
torchrun --nproc_per_node=8 supervised-fine-tune.py  \
        --model_name_or_path path_to_finetuned_models \
        --bf16 True \
        --output_dir path_to_saving_checkpoints       \
        --model_max_length 32768 \
        --use_flash_attn True \
        --data_path LongQA.json \
        --low_rank_training True \
        --num_train_epochs 3  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 1     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 1000     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage2.json" \
        --tf32 True
```
- We typically make supervised fine-tuning upon the fine-tuned context extended models, `path_to_finetuned_models`
- During our dataset collection, it is hard for us to collect many high-quality QA that are larger than 32768. Thus, if you use our `LongQA.json`, please also set `model_max_length` as 32768.


### Get trainable weights in low-rank training
In low-rank training, we set embedding and normalization layers as trainable. Please use the following line to extract the trainable weights `trainable_params.bin` from `pytorch_model.bin`
```
python3 get_trainable_weights.py --checkpoint_path path_to_saving_checkpoints --trainable_params "embed,norm"
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin` and trainable parameters `trainable_params.bin`, save the resulting model into your desired path in the Hugging Face format:
```
python3 merge_lora_weights_and_save_hf_model.py \
        --base_model path_to/Llama-2-7b-hf \
        --peft_model path_to_saving_checkpoints \
        --context_size 8192 \
        --save_path path_to_saving_merged_model
```
For example,
```
python3 merge_lora_weights_and_save_hf_model.py \
        --base_model /dataset/pretrained-models/Llama-2-7b-hf \
        --peft_model /dataset/yukangchen/hf_models/lora-models/Llama-2-7b-longlora-8k \
        --context_size 8192 \
        --save_path /dataset/yukangchen/models/Llama-2-7b-longlora-8k-merged
```


### Validation
To evaluate a model that is trained in the low-rank setting, please set both `base_model` and `peft_model`. `base_model` is the pre-trained weight. `peft_model` is the path to the saved checkpoint, which should contain `trainable_params.bin`, `adapter_model.bin` and `adapter_config.json`. For example,
```
python3 eval.py --seq_len 8192 --context_size 8192 --batch_size 1 --base_model path_to/Llama-2-7b-hf --peft_model path_to_saving_checkpoints --data_path pg19/test.bin
```

To evaluate a model that is fully fine-tuned, you only need to set `base_model` as the path to the saved checkpoint, which should contain `pytorch_model.bin` and `config.json`. `peft_model` should be ignored.
```
python3 eval.py --seq_len 8192 --context_size 8192 --batch_size 1 --base_model path_to_saving_checkpoints --data_path pg19/test.bin
```

- Note that `--seq_len` is to set the sequence length for evaluation. `--context_size` is to set the context length of the model during fine-tuning. `--seq_len` should not be larger than `--context_size`.

- We have already tokenized the validation and test splits of PG19 and proof-pile dataset into `pg19/validation.bin`, `pg19/test.bin`, and `proof-pile/test_sampled_data.bin`, with the tokenizer of LLaMA. `proof-pile/test_sampled_data.bin` contains 128 documents that are randomly sampled from the total proof-pile test split. For each document, it has at least 32768 tokens. We also release the sampled ids in [proof-pile/test_sampled_ids.bin](https://drive.google.com/file/d/1cnzWODLRQYAd7HeugzLCIhaqzaLZv7J5/view?usp=share_link). You can download them from the links below.

| Dataset    | Split      | Link                                                                                                         |
|:-----------|------------|--------------------------------------------------------------------------------------------------------------|
| PG19       | validation | [pg19/validation.bin](https://drive.google.com/file/d/1rbJvb0qRIf2mQoN2ON7S93TbTzMnlrN6/view?usp=share_link) |
| PG19       | test       | [pg19/test.bin](https://drive.google.com/file/d/1QANDMdctpacPAYgS04adDXqByGEq-Ret/view?usp=share_link)       |
| Proof-pile | test       | [proof-pile/test_sampled_data.bin](https://drive.google.com/file/d/1bUI5lPDvrqzY_XXJJ2sSuvZx0Y9AZClE/view?usp=share_link)         |
 

### Passkey Retrieval
We provide a manner to test the passkey retrieval accuracy. For example,
```
python3 passkey_retrivial.py \
        --context_size 32768 \
        --base_model path_to/Llama-2-7b-longlora-32k \
        --max_tokens 32768 \
        --interval 1000
```
- Note that the `context_size` is the context length during fine-tuning.
- `max_tokens` is maximum length for the document in passkey retrieval evaluation.
- `interval` is the interval during the document length increasing. It is a rough number because the document increases by sentences.

## Inference 
To chat with [Llama-2-13b-chat-longlora-32k-sft](https://huggingface.co/Yukang/Llama-2-13b-chat-longlora-32k-sft) or [Llama-2-70b-chat-longlora-32k-sft](https://huggingface.co/Yukang/Llama-2-70b-chat-longlora-32k-sft), you need to run `merge_lora_weights_and_save_hf_model.py` first, and then:
```
python3 inference.py  \
        --base_model path_to_model \
        --question $question \
        --context_size $context_length \
        --max_gen_len $max_gen_len \
        --flash_attn True \
        --material $material_content \
        --material_type $material_type \
        --material_title $material_title
```

For example, if we want to ask a question related to a book:
```
python3 inference.py  \
        --base_model /data/models/Llama-2-13b-chat-longlora-32k-sft \
        --question "Why doesn't Professor Snape seem to like Harry?" \
        --context_size 32768 \
        --max_gen_len 512 \
        --flash_attn True \
        --material "materials/Harry Potter and the Philosophers Stone_section2.txt" \
        --material_type "book" \
        --material_title "Harry Potter and the Philosophers Stone"
```
Note that you can ignore `material_type` or `material_title`.

If we want to ask a question related to a paper:
```
python3 inference.py  \
        --base_model /data/models/Llama-2-13b-chat-longlora-32k-sft \
        --question "What are the main contributions and novelties of this work?" \
        --context_size 32768 \
        --max_gen_len 512 \
        --flash_attn True \
        --material "materials/paper1.txt" \
        --material_type "paper"
```

## Demo
If you want to deploy your own demo, you can run the 
```
python3 demo.py  \
	--base_model path_to_model \
	--context_size $context_size \
	--max_gen_len $max_gen_len \
	--flash_attn True
```
For example, 
```
python3 demo.py  \
	--base_model /data/models/Llama-2-13b-chat-longlora-32k-sft \
	--context_size 32768 \
	--max_gen_len 512 \
	--flash_attn True
```
- Note that `flash_attn=True` will make the generation slow but save much GPU memory.

## Pdf2text
During our dataset collection, we convert paper and books from pdf to text. The conversion quality has a large influence on the final model quality. We think that this step is non-trivial. We release the tool for the pdf2txt conversion, in the folder `pdf2txt`. It is built upon `pdf2image`, `easyocr`, `ditod` and `detectron2`. Please refer to the [README.md](pdf2txt/README.md) in `pdf2txt` for more details.

## Citation 
If you find this project useful in your research, please consider citing:

```
@article{longlora,
  title={LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models},
  author={Yukang Chen and Shengju Qian and Haotian Tang and Xin Lai and Zhijian Liu and Song Han and Jiaya Jia},
  journal={arXiv:2309.12307},
  year={2023}
}
```

## Acknowledgement
- This work is an GPTNeoX port of the work from the original authors' code. [LongLoRA](https://github.com/dvlab-research/LongLoRA)
- This work is built upon the [GPTNeoX-HF](https://huggingface.co/docs/transformers/model_doc/gpt_neox) which is based upon [EleutherAI/GPTNeoX](https://github.com/EleutherAI/gpt-neox) as the pre-trained model architecture.
- This work is based on [DeepSpeed](https://github.com/microsoft/DeepSpeed), [peft](https://github.com/huggingface/peft), and [Flash-Attention2](https://github.com/Dao-AILab/flash-attention) for acceleration.
- Some evaluation code is modified upon [Landmark Attention](https://github.com/epfml/landmark-attention).
- We use [LongChat](https://github.com/DachengLi1/LongChat) for the retrieval evaluation.

