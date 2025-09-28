import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

import setproctitle
setproctitle.setproctitle("math_eval@zhanghonglin")  # Set process name for monitoring


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str, help="Dataset names, separated by comma")
    parser.add_argument("--data_dir", default="./data", type=str, help="Dataset directory")
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str, help="Model name or path")
    parser.add_argument("--output_dir", default="./output", type=str, help="Output directory")
    parser.add_argument("--prompt_type", default="tool-integrated", type=str, help="Prompt type")
    parser.add_argument("--split", default="test", type=str, help="Dataset split")
    parser.add_argument("--num_test_sample", default=-1, type=int, help="Number of test samples, -1 for all")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--start", default=0, type=int, help="Start sample index")
    parser.add_argument("--end", default=-1, type=int, help="End sample index")
    parser.add_argument("--temperature", default=0, type=float, help="Sampling temperature")
    parser.add_argument("--n_sampling", default=1, type=int, help="Number of samples per example")
    parser.add_argument("--top_p", default=1, type=float, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens_per_call", default=2048, type=int, help="Max tokens per generation call")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample order")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--save_outputs", action="store_true", help="Save outputs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--use_safetensors", action="store_true", help="Use safetensors to load model")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    # If temperature is 0, top_p must be 1 (greedy sampling)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def prepare_data(data_name, args):
    """
    Load and prepare data, deduplicate, return samples to process, processed samples, and output file path
    """
    examples = load_data(data_name, args.split, args.data_dir)

    # Only take the specified number of test samples
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # Shuffle sample order if needed
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # Select samples in the start-end index range
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Construct output file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # Load all previously processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # Deduplicate to avoid repeated processing
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    # Only keep samples that have not been processed
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    """
    Load model, perform inference and evaluation on all datasets
    """
    # Get available GPU list
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        # Load model with vLLM
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        # Load model with transformers
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # Inference and evaluation for each dataset
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # Calculate average accuracy across all datasets
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # Print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    """
    Determine if the answer is a multiple-choice (only contains A-E)
    """
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    """
    Main process: prepare data, generate, execute, evaluate, and save results
    """
    import json
    import numpy as np

    def to_json_serializable(obj):
        # Recursively convert numpy arrays to lists for JSON serialization
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # torch.Tensor is not expected here, but just in case
        elif 'torch' in str(type(obj)):
            try:
                return obj.cpu().tolist()
            except Exception:
                return obj.tolist()
        else:
            return obj

    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, ", remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # Initialize Python code executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # Parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # Add remaining fields (such as difficulty, type, options, etc.)
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # Apply chat template if needed
    if args.apply_chat_template:
        for sample in samples:
            sample['prompt'] = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample['prompt'].strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
    
    # Repeat each sample n_sampling times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]

    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    # Maximum function call times (cot/pal only once, others up to 4 times)
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    # Stop words list
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.use_vllm:
        eos_token = llm.get_tokenizer().eos_token
        if eos_token not in stop_words:
            stop_words.append(eos_token)
            print('=' * 20)
            print('Add vLLM eos token:', eos_token)
            print('=' * 20)
    else:
        # Get special stop words from tokenizer if available
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            if tokenizer.eos_token not in stop_words:
                stop_words.append(tokenizer.eos_token)
                print('=' * 20)
                print('Add eos token:', tokenizer.eos_token)
                print('=' * 20)

    # Extend stop words according to prompt type
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # Store all token ids for inference sequences
    all_prompt_token_ids = []
    all_generate_token_ids = []

    # Start inference timer
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # Get all outputs
        prompts = [item[1] for item in current_prompts]
        outputs = None
        output_token_ids = None
        prompt_token_ids = None

        if args.use_vllm:
            # vLLM inference
            vllm_outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            # Sort by request_id to ensure order
            vllm_outputs = sorted(
                vllm_outputs, key=lambda x: int(x.request_id)
            )
            outputs = [output.outputs[0].text for output in vllm_outputs]
            # vLLM: prompt_token_ids and generate_token_ids are already separated
            prompt_token_ids = [output.prompt_token_ids for output in vllm_outputs]
            output_token_ids = [output.outputs[0].token_ids for output in vllm_outputs]
        else:
            # transformers inference
            # generate_completions should return token ids
            outputs, output_token_ids, prompt_token_ids = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
                return_token_ids=True,
                return_prompt_token_ids=True,
            )

        assert len(outputs) == len(current_prompts)
        assert len(output_token_ids) == len(current_prompts)
        assert len(prompt_token_ids) == len(current_prompts)

        # Process all outputs
        remain_prompts = []
        remain_codes = []
        for idx_in_batch, ((i, query), output, gen_token_ids, prmpt_token_ids) in enumerate(zip(current_prompts, outputs, output_token_ids, prompt_token_ids)):
            output = output.rstrip()
            query += output
            # Record prompt_token_ids and generate_token_ids
            all_prompt_token_ids.append(prmpt_token_ids)
            all_generate_token_ids.append(gen_token_ids)
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                # End of code block, extract program
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # Execute remaining codes
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # Mark as terminated if max function call limit reached
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # Unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # Sort by index
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # Remove input_prompt part from end_prompt to get code
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # Execute code and extract predictions
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time  # Record total time used

    # Fill results back into samples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        # Get prompt_token_ids and generate_token_ids
        prompt_token_ids = all_prompt_token_ids[i * args.n_sampling : (i + 1) * args.n_sampling]
        generate_token_ids = all_generate_token_ids[i * args.n_sampling : (i + 1) * args.n_sampling]
        # Convert token_ids to list if they are numpy arrays or torch tensors
        prompt_token_ids = [list(tid) for tid in prompt_token_ids]
        generate_token_ids = [list(tid) for tid in generate_token_ids]
        for j in range(len(preds)):
            # Clean multiple-choice answers
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # Remove non-option characters
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        # sample.pop("prompt")
        sample.update({
            "code": code,
            "pred": preds,
            "report": reports,
            "prompt_token_ids": prompt_token_ids,
            "generate_token_ids": generate_token_ids
        })
        all_samples.append(sample)

    # Add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # Only save outputs if both conditions are met:
    # 1. The number of newly processed samples is greater than the number of already processed samples (i.e., new samples have been inferred/evaluated, to avoid duplicate writes)
    # 2. The user explicitly requests to save outputs via the --save_outputs command line argument
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        # Convert all_samples to JSON serializable before saving
        serializable_samples = [to_json_serializable(sample) for sample in all_samples]
        save_jsonl(serializable_samples, out_file)

    # Record time used
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    # Save evaluation metrics
    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    # Main entry: parse arguments, set random seed, run main process
    args = parse_args()
    set_seed(args.seed)
    setup(args)
