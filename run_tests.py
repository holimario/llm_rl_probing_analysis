# Non-multi-head

from functools import partial

import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from src.transformer_lens import HookedTransformer

from src.eap.my_evaluate_non_heads import evaluate_graph, evaluate_baseline

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Configuration')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Dataset name to use')
    
    # Model series parameters
    parser.add_argument('--model_series', type=str, required=True,
                       choices=['deepseek', 'mistral', 'qwen', 'nvidia-qwen'],
                       help='Model series to use')
    
    # Model path parameters
    parser.add_argument('--model_name', type=str, required=True,
                       help='Path to the model')
    
    # Model type parameters
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['base', 'rl'],
                       help='Model type: base or rl')
    
    # r parameter
    parser.add_argument('--r', type=float, required=True,
                       help='R parameter value')

    parser.add_argument('--cut_coeff', type=float, required=True)

    parser.add_argument('--small_coeff', type=float, required=True)

    parser.add_argument('--big_coeff', type=float, required=True)

    parser.add_argument('--balance_coeff', type=float, required=True)

    parser.add_argument('--num_samples', type=int, required=True)
    
    return parser.parse_args()

args = parse_args()

dataset_name = args.dataset_name
model_series = args.model_series
model_name = args.model_name
model_type = args.model_type
r = args.r
cut_coeff = args.cut_coeff
small_coeff = args.small_coeff
big_coeff = args.big_coeff
balance_coeff = args.balance_coeff
num_samples = args.num_samples

model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda:1',
    dtype=torch.bfloat16 # use bfloat16 here
)
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

from src.eap.my_graph_non_heads import Graph

# Initialize graph
g = Graph.from_model(model)

print('Number of edges in the graph:', len(g.edges))
print('Number of nodes in the graph:', len(g.nodes))


from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

def collate(data_list, cut_len):
    """
    Batch collation function to pack samples into a batch.
    Args:
        data_list: a batch of samples, each is (clean, corrupted, labels)
    Returns:
        input_ids: token ids (including prompt and model answer)
        answer_range: the token id range corresponding to the model output
    """
    input_ids, answer_range = zip(*data_list)
    input_ids = list(input_ids)    
    answer_range = list(answer_range)
    origin_answer_token_ids = [x[y[0] : y[0] + cut_len] for x, y in zip(input_ids, answer_range)]
    input_ids = [x[:y[0] + cut_len - 1] for x, y in zip(input_ids, answer_range)]
    return input_ids, answer_range, origin_answer_token_ids

class MathDataset(Dataset):
    def __init__(self, filepath, model_type, cut_coeff=0.5, small_coeff=0.5, big_coeff=2.0, 
                rand_seed=42, num_samples=-1, verbose=True, balance_coeff=0.1):
        """
        Initialize and read JSONL file, dynamically determine filtering conditions based on statistics.
        
        Args:
            filepath: str, path to JSONL file
            model_type: str, model type
            cut_coeff: float, coefficient for cut_len, default 0.5
            small_coeff: float, min length coefficient, default 0.5
            big_coeff: float, max length coefficient, default 2.0
            rand_seed: int, random seed, default 42
            num_samples: int, number of samples to select, -1 means all
            verbose: bool, whether to print detailed info, default True
            balance_coeff: float, max allowed relative difference between base and rl answer lengths
        """
        import json
        import random
        import numpy as np
        from collections import Counter
        
        self.type = model_type
        
        # Step 1: Read all data and compute statistics
        if verbose:
            print("Reading data and computing statistics...")
        
        all_data = []
        base_answer_lengths = []
        rl_answer_lengths = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                all_data.append(row)
                base_answer_lengths.append(len(row['base_answer_tokens']))
                rl_answer_lengths.append(len(row['rl_answer_tokens']))
        
        original_count = len(all_data)
        
        def calculate_mean_with_stats(data, name=""):
            """Compute mean and print statistics"""
            mean_value = np.mean(data)
            
            if verbose and name:
                counter = Counter(data)
                max_count = max(counter.values())
                modes = [k for k, v in counter.items() if v == max_count]
                mode_value = sorted(modes)[len(modes) // 2] if len(modes) > 1 else modes[0]
                
                print(f"{name} statistics:")
                print(f"  Mean: {mean_value:.2f}")
                print(f"  Median: {np.median(data):.2f}")
                print(f"  Mode: {mode_value} (appeared {max_count} times)")
                print(f"  Std: {np.std(data):.2f}")
                print(f"  Range: [{min(data)}, {max(data)}]")
                if len(modes) > 1:
                    print(f"  Multiple modes: {sorted(modes)}")
            
            return mean_value
        
        # Compute means and statistics
        base_mean = calculate_mean_with_stats(base_answer_lengths, "base_answer_tokens length")
        rl_mean = calculate_mean_with_stats(rl_answer_lengths, "rl_answer_tokens length")
        
        # Compute reference value and length limits
        refer = min(base_mean, rl_mean)
        cut_len = max(int(refer * cut_coeff), 1)
        min_total_len = int(small_coeff * refer)
        max_total_len = int(big_coeff * refer)
        
        if verbose:
            print(f"\n=== Computed Parameters ===")
            print(f"Reference value refer = min({base_mean}, {rl_mean}) = {refer}")
            print(f"cut_len = int({refer} * {cut_coeff}) = {cut_len}")
            print(f"Total length filter range: [{min_total_len}, {max_total_len}]")
            print("========================")
        
        # Step 2: Apply filtering conditions and count filtering stats
        if verbose:
            print("\nApplying filtering conditions...")
        
        filter_stats = {
            'prompt_length_mismatch': 0,
            'total_length_out_of_range': 0,
            'diff_out_of_balance': 0,
            'passed_all_filters': 0
        }
        
        filtered_data = []
        
        for row in all_data:
            # Condition 1: prompt lengths must match
            if len(row['base_prompt_tokens']) != len(row['rl_prompt_tokens']):
                filter_stats['prompt_length_mismatch'] += 1
                continue
            
            # Condition 2: total length in specified range
            total_len = (len(row['base_answer_tokens']) + len(row['rl_answer_tokens']))/2
            
            if total_len > max_total_len or total_len < min_total_len:
                filter_stats['total_length_out_of_range'] += 1
                continue

            # Condition 3: two entries' lengths are balanced
            diff = abs(len(row['base_answer_tokens']) - len(row['rl_answer_tokens']))
            
            if diff/total_len > balance_coeff:
                filter_stats['diff_out_of_balance'] += 1
                continue

            filter_stats['passed_all_filters'] += 1
            filtered_data.append(row)
        
        filtered_count = len(filtered_data)
        
        # Step 3: Random sampling if needed
        need_sampling = num_samples > 0 and num_samples < filtered_count
        before_sampling_count = filtered_count
        
        if need_sampling:
            if verbose:
                print(f"Randomly selecting {num_samples} samples from {filtered_count}")
            random.seed(rand_seed)
            self.data = random.sample(filtered_data, num_samples)
            final_count = len(self.data)
        else:
            self.data = filtered_data
            final_count = len(self.data)
        
        # Save parameters for later use
        self.refer = refer
        self.cut_len = cut_len
        
        # Print final statistics
        if verbose:
            print(f"\n=== Final Filtering Statistics ===")
            print(f"Original data count: {original_count}")
            print(f"\nFilter condition statistics:")
            print(f"  Prompt length mismatch: {filter_stats['prompt_length_mismatch']}")
            print(f"  Total length out of range: {filter_stats['total_length_out_of_range']}")
            print(f"  Length difference out of balance: {filter_stats['diff_out_of_balance']}")
            print(f"  Passed all filters: {filter_stats['passed_all_filters']}")
            
            if need_sampling:
                print(f"\nSample count before random selection: {before_sampling_count}")
                print(f"Final sample count: {final_count}")
            else:
                print(f"\nFinal sample count: {final_count}")
            
            print(f"Data retention rate: {final_count/original_count*100:.2f}%")
            print("====================")
        else:
            print(f"Original data count: {original_count}")
            if need_sampling:
                print(f"Sample count before random selection: {before_sampling_count}")
            print(f"Final sample count: {final_count}")


    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)
    
    def shuffle(self):
        """
        Shuffle the dataset in place.
        """
        import random
        random.shuffle(self.data)

    def head(self, n: int):
        """
        Keep only the first n samples (for debugging or small experiments).
        Args:
            n: int, number of samples to keep
        """
        self.data = self.data[:n]
    
    def __getitem__(self, index):
        """
        Get the sample at index.
        Returns:
            _all_tokens: token ids (including prompt and answer)
            _answer_range: the token id range for the model output, left-closed right-open, e.g. [0, 126] means index_0 to index_125
        """
        row = self.data[index]
        if self.type == 'base':
            return row['base_prompt_tokens'] + row['base_answer_tokens'], row['base_answer_tokens_range']
        elif self.type == 'rl':
            return row['rl_prompt_tokens'] + row['rl_answer_tokens'], row['rl_answer_tokens_range']
    
    def to_dataloader(self, batch_size: int):
        """
        Convert to PyTorch DataLoader.
        Args:
            batch_size: int, batch size
        Returns:
            DataLoader object
        """
        from functools import partial
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(collate, cut_len=self.cut_len))
    

def logit_diff(logits, input_lengths, answer_range, origin_answer_token_ids):
    """
    Compute the average cross-entropy for the extrapolation part.
    Args:
        logits: FloatTensor [B, T, V]
        input_lengths: LongTensor [B]  (only for printing, can be ignored)
        answer_range: list[list[int]]  length=B, each element=[prompt_len, unused]
        origin_answer_token_ids: list[list[int]]  length=B, each sublist=ground truth token id sequence
    Returns:
        loss: FloatTensor [B]  average cross-entropy for each sample
    """
    B, T, V = logits.shape
    device = logits.device
    losses = []

    for i in range(B):
        prompt_len = answer_range[i][0]          # extrapolation start
        tgt_ids = origin_answer_token_ids[i]     # ground truth token ids
        L = len(tgt_ids)                         # extrapolation length

        # Get corresponding logits: shape [L, V]
        logits_i = logits[i, prompt_len-1 : prompt_len-1+L, :]   # -1 because logits[t] predicts t+1

        # Convert to log-prob
        log_probs = F.log_softmax(logits_i, dim=-1)             # [L, V]

        # Convert ground truth token ids to tensor
        targets = torch.tensor(tgt_ids, device=device, dtype=torch.long)  # [L]

        # Cross-entropy = - sum( log_prob[indices] ) / L
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [L]
        loss = nll.mean()
        losses.append(loss)

    return torch.stack(losses).mean()

dataset_path = f'/datasets/processed/{model_series}/{dataset_name}/all_correct.jsonl'

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Read data
base_prompt_lens = []
base_answer_lens = []
rl_prompt_lens = []
rl_answer_lens = []

with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if 'base_prompt_tokens' in item and isinstance(item['base_prompt_tokens'], list):
            base_prompt_lens.append(len(item['base_prompt_tokens']))
        if 'base_answer_tokens' in item and isinstance(item['base_answer_tokens'], list):
            base_answer_lens.append(len(item['base_answer_tokens']))
        if 'rl_prompt_tokens' in item and isinstance(item['rl_prompt_tokens'], list):
            rl_prompt_lens.append(len(item['rl_prompt_tokens']))
        if 'rl_answer_tokens' in item and isinstance(item['rl_answer_tokens'], list):
            rl_answer_lens.append(len(item['rl_answer_tokens']))

def print_stats(name, lens):
    arr = np.array(lens)
    print(f"{name} length statistics:")
    print(f"  Number of samples: {len(arr)}")
    print(f"  Mean: {arr.mean():.2f}")
    print(f"  Std: {arr.std():.2f}")
    print(f"  Min: {arr.min()}")
    print(f"  Max: {arr.max()}")
    print(f"  Median: {np.median(arr)}")
    print(f"  Quantiles (25%, 75%): {np.percentile(arr, 25)}, {np.percentile(arr, 75)}")
    print()

print_stats('base_prompt_tokens', base_prompt_lens)
print_stats('base_answer_tokens', base_answer_lens)
print_stats('rl_prompt_tokens', rl_prompt_lens)
print_stats('rl_answer_tokens', rl_answer_lens)

# Plot distributions
save_dir = "token_length_analysis"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(10,6))
plt.hist(base_prompt_lens, bins=30, alpha=0.6, label='base_prompt_tokens')
plt.hist(base_answer_lens, bins=30, alpha=0.6, label='base_answer_tokens')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.title('Token Length Distribution (Base)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'{model_series}_{dataset_name}_base_token_length_distribution.png'))
plt.show()

plt.figure(figsize=(10,6))
plt.hist(rl_prompt_lens, bins=30, alpha=0.6, label='rl_prompt_tokens')
plt.hist(rl_answer_lens, bins=30, alpha=0.6, label='rl_answer_tokens')
plt.xlabel('Token Length')
plt.ylabel('Count')
plt.title('Token Length Distribution (RL)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'{model_series}_{dataset_name}_rl_token_length_distribution.png'))
plt.show()

# Compute the difference distribution between base and rl answer token counts
base_answer_arr = np.array(base_answer_lens)
rl_answer_arr = np.array(rl_answer_lens)
min_len = min(len(base_answer_arr), len(rl_answer_arr))
if min_len == 0:
    print("base_answer_tokens or rl_answer_tokens is empty, cannot compute difference distribution.")
    diff_arr = np.array([])
else:
    diff_arr = rl_answer_arr[:min_len] - base_answer_arr[:min_len]
    print("rl_answer_tokens - base_answer_tokens difference statistics:")
    print(f"  Number of samples: {len(diff_arr)}")
    print(f"  Mean: {diff_arr.mean():.2f}")
    print(f"  Std: {diff_arr.std():.2f}")
    print(f"  Min: {diff_arr.min()}")
    print(f"  Max: {diff_arr.max()}")
    print(f"  Median: {np.median(diff_arr)}")
    print(f"  Quantiles (25%, 75%): {np.percentile(diff_arr, 25)}, {np.percentile(diff_arr, 75)}")
    print()

    # Plot difference distribution
    plt.figure(figsize=(10,6))
    plt.hist(diff_arr, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Token Length Difference (rl_answer_tokens - base_answer_tokens)')
    plt.ylabel('Count')
    plt.title('Token Length Difference Distribution (RL Answer - Base Answer)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_series}_{dataset_name}_{model_type}_cut{cut_coeff}_answer_token_length_difference_distribution.png'))
    plt.show()

print(f"Distribution plots saved to {save_dir} folder.")


ds = MathDataset(dataset_path, model_type=model_type, num_samples=num_samples, cut_coeff=cut_coeff, 
                    small_coeff=small_coeff,
                    big_coeff=big_coeff,
                    balance_coeff=balance_coeff)
dataloader = ds.to_dataloader(1)

from src.eap.my_attribute_non_heads import attribute 

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
all_scores = attribute(model, g, dataloader, logit_diff, method='test1', device='cuda:1')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create save directory
save_dir = "heatmap_results"
os.makedirs(save_dir, exist_ok=True)

# 1. Basic heatmap (assuming g.scores is a 2D array)
plt.figure(figsize=(18, 15))

# High-contrast color map
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw heatmap and adjust axes
ax = sns.heatmap(g.scores.T, 
                cmap=cmap, 
                center=0,
                robust=True,  # Use 2-98 percentile range for color scaling
                # square=True,
                cbar_kws={"shrink": 0.8})

# Set axes so that 0 index is at the bottom left
ax.invert_yaxis()  # Reverse y-axis
ax.xaxis.tick_bottom()  # x-axis ticks at the bottom

# Adjust labels and title
plt.title(f'Edge Scores Heatmap ({model_series}-{dataset_name}-{model_type}-cut{cut_coeff})', pad=20)
plt.xlabel('Forward Index')
plt.ylabel('Backward Index')

# Save figure
plt.savefig(os.path.join(save_dir, f'{model_series}_{dataset_name}_{model_type}_cut{cut_coeff}_basic_heatmap.png'), 
           dpi=300, 
           bbox_inches='tight',
           transparent=False)
plt.close()

print(f"Heatmap saved to {save_dir}/{model_series}_{dataset_name}_{model_type}_cut{cut_coeff}_basic_heatmap.png")


import numpy as np
import matplotlib.pyplot as plt
import os

# Save raw data
save_dir = f"subgraph_extract_results"
os.makedirs(save_dir, exist_ok=True)
data_save_path = os.path.join(save_dir, f"results_{model_series}_{dataset_name}_{model_type}_cut{cut_coeff}.npz")
np.savez(data_save_path, scores=g.scores.cpu().numpy(), all_scores=all_scores.float().cpu().numpy())
