#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastModel, FastLanguageModel
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for training and inference')
    parser.add_argument('--num_generations', type=int, default=8, help='Number of generations per example for RL')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for sampling')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k for sampling')
    return parser.parse_args()

args = parse_args()

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llm_file_save/unsloth/qwen2.5-3b-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    load_in_8bit = False,
    # fast_inference = True,
    full_finetuning = True,
    local_files_only = True,
    # token = "hf_...",
)

from datasets import load_dataset
dataset = load_dataset("/datasets/gsm8k", "main", split = "train")
dataset

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()
extract_hash_answer(dataset[0]["answer"])

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "user",   "content": f"{x['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"},
    ],
    "answer": extract_hash_answer(x["answer"]),
})
print(dataset[0])


from parser import extract_answer
from grader import math_equal

def check_boxed_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    guesses = [extract_answer(r, "gsm8k") for r in responses]
    scores = []
    for guess, gt in zip(guesses, answer):
        if guess is None or gt is None or guess == "":
            scores.append(0.0)
            continue

        guess_str = str(guess).strip()
        gt_str = str(gt).strip()

        try:
            if math_equal(guess_str, gt_str):
                scores.append(3.0)
                continue
        except Exception:
            pass
        # if guess_str == gt_str:
        #     scores.append(2.0)
        #     continue
        norm_guess = guess_str.replace('$', '').replace(',', '').strip()
        norm_gt = gt_str.replace('$', '').replace(',', '').strip()
        if norm_guess == norm_gt:
            scores.append(1.0)
            continue
        try:
            g1 = float(norm_guess)
            g2 = float(norm_gt)
            if abs(g1 - g2) < 1e-8:
                scores.append(1.0)
                continue
            rel = abs(g1 - g2) / (abs(g2) + 1e-12)
            if rel <= 0.01:
                scores.append(0.5)
            elif rel <= 0.05:
                scores.append(0.1)
            else:
                scores.append(-0.5)
        except Exception:
            scores.append(0.0)
    return scores

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    # optim = "adamw_torch_fused",
    optim = 'adamw_8bit',
    logging_steps = 1,
    per_device_train_batch_size = 32,
    gradient_accumulation_steps = 1,
    num_generations = args.num_generations,
    max_prompt_length = max_prompt_length,
    # max_completion_length = max_seq_length - max_prompt_length,
    max_completion_length = 200,
    # num_train_epochs = 1,
    max_steps = 400,
    save_steps = 10,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "outputs/temp_{}_top_p_{}_top_k_{}_num_generations_{}".format(args.temperature, args.top_p, args.top_k, args.num_generations),
    temperature = args.temperature,
    top_p = args.top_p,
    # loss_type='grpo',
    # beta=0.1
    # top_k = args.top_k,
    # gradient_checkpointing=False,
    # dataloader_num_workers=8,
    # use_vllm = True,
    # vllm_mode = "colocate",
    # vllm_model_impl = "vllm",
)

print(training_args)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        check_boxed_answer,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

