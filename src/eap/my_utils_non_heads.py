from typing import List, Optional, Tuple, Union
from functools import partial
import pickle

from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from src.transformer_lens import HookedTransformer
from src.transformer_lens.utils import get_attention_mask
from einops import einsum

from .my_graph_non_heads import Graph, AttentionNode, LogitNode


def token_padding(model: HookedTransformer, inputs: List[str], padding_token_id: int):
    """
    Tokenize a list of input string IDs using the given model, and return the tokenized results,
    attention mask, input lengths, and the maximum sequence length in the batch.

    Args:
        model (HookedTransformer): The model instance used for tokenization.
        inputs (List[str]): List of string IDs with varying lengths.
        padding_token_id (int): The token ID used for padding.

    Returns:
        tuple: A tuple containing:
            - tokens (torch.Tensor): Tokenized tensor of shape [batch, seq_len].
            - attention_mask (torch.Tensor): Attention mask, 1 for valid tokens, 0 for padding, shape [batch, seq_len].
            - input_lengths (torch.Tensor): Number of valid tokens for each input, shape [batch].
            - n_pos (int): Maximum sequence length in the current batch (i.e., seq_len).
    """

    assert model.tokenizer.padding_side == 'right', "padding_side must be right"
    
    # Find the maximum length
    max_len = max(len(seq) for seq in inputs)
    # Pad each sequence on the right
    tokens = []
    for seq in inputs:
        padded = seq + [padding_token_id] * (max_len - len(seq))
        tokens.append(padded)
    tokens = torch.tensor(tokens, dtype=torch.long, device=model.cfg.device)

    # Generate attention mask: 1 for valid tokens, 0 for padding
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)

    # Calculate the number of valid tokens for each input (sum of each row in attention_mask)
    input_lengths = attention_mask.sum(1)
    # Get the maximum sequence length in the current batch
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos

def tokenize_plus(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
    """
    Tokenize a list of input strings using the given model, and return the tokenized results,
    attention mask, input lengths, and the maximum sequence length in the batch.

    Args:
        model (HookedTransformer): The model instance used for tokenization.
        inputs (List[str]): List of strings to tokenize.
        max_length (Optional[int]): (Optional) Maximum sequence length. If specified, sequences will be truncated or padded.

    Returns:
        tuple: A tuple containing:
            - tokens (torch.Tensor): Tokenized tensor of shape [batch, seq_len].
            - attention_mask (torch.Tensor): Attention mask, 1 for valid tokens, 0 for padding, shape [batch, seq_len].
            - input_lengths (torch.Tensor): Number of valid tokens for each input, shape [batch].
            - n_pos (int): Maximum sequence length in the current batch (i.e., seq_len).
    """
    if max_length is not None:
        # Temporarily modify the model's n_ctx parameter to fit the tokenization length
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    # Use the model's to_tokens method for tokenization
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    if max_length is not None:
        # Restore the original n_ctx parameter
        model.cfg.n_ctx = old_n_ctx
    # Generate attention mask: 1 for valid tokens, 0 for padding
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    # Calculate the number of valid tokens for each input (sum of each row in attention_mask)
    input_lengths = attention_mask.sum(1)
    # Get the maximum sequence length in the current batch
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor]):
    """Creates a matrix and hooks to fill it and the score matrix.

    Args:
        model (HookedTransformer): Model to attribute.
        graph (Graph): Graph to attribute.
        batch_size (int): Size of the current batch.
        n_pos (int): Size of the position dimension.
        scores (Tensor): The scores tensor to fill. If None, hooks/matrices are for evaluation only (do not use backward hooks).

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted - clean activations. 
        The first set of hooks adds activations (run on corrupted input), the second subtracts activations (run on clean input), 
        and the third computes gradients and updates the scores matrix.
    """
    separate_activations = model.cfg.use_normalization_before_and_after and scores is None
    if separate_activations:
        activation_difference = torch.zeros((2, batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
        
    # Fills up the activation difference matrix. In the default case (not separate_activations), 
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient, 
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach()
        try:
            if separate_activations:
                if add:
                    activation_difference[0, :, :, index] += acts
                else:
                    activation_difference[1, :, :, index] += acts
            else:
                if add:
                    activation_difference[:, :, index] += acts
                else:
                    activation_difference[:, :, index] -= acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass 
            hook (_type_): (unused)

        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), activation_difference.device, grads.size(), grads.device)
            print(prev_index, bwd_index, scores.size(), s.size())
            raise e
    
    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
    
    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate('qkv'):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
        
    node = graph.nodes['logits']
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def compute_mean_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, per_position=False):
    """
    Compute the mean activations of a graph's nodes over a dataset.
    """
    def activation_hook(index, activations, hook, means=None, input_lengths=None):
        # This hook fills up the means tensor. Means is of shape
        # (n_pos, graph.n_forward, model.cfg.d_model) if per_position is True, otherwise
        # (graph.n_forward, model.cfg.d_model) 
        acts = activations.detach()

        # If input_lengths is provided, we assume you want to mean over positions
        if input_lengths is not None:
            mask = torch.zeros_like(activations)
            # Mask out all padding positions
            mask[torch.arange(activations.size(0)), input_lengths - 1] = 1
            
            # Use ... in einsum in case there is a head index as well
            item_means = einsum(acts, mask, 'batch pos ... hidden, batch pos ... hidden -> batch ... hidden')
            
            # Mean over the positions we did take, position-wise
            if len(item_means.size()) == 3:
                item_means /= input_lengths.unsqueeze(-1).unsqueeze(-1)
            else:
                item_means /= input_lengths.unsqueeze(-1)

            means[index] += item_means.sum(0)
        else:
            means[:, index] += acts.sum(0)

    # Gather all out hooks / indices needed for making hooks
    # But we can't make them until we have input length masks
    processed_attn_layers = set()
    hook_points_indices = []
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)
        
        if not isinstance(node, LogitNode):
            hook_points_indices.append((node.out_hook, graph.forward_index(node)))

    means_initialized = False
    total = 0
    for batch in tqdm(dataloader, desc='Computing mean'):
        # The dataset may be a tuple or just raw strings
        batch_inputs = batch[0] if isinstance(batch, tuple) else batch
        tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, batch_inputs, max_length=512)
        total += len(batch_inputs)

        if not means_initialized:
            # This is where we store the means
            if per_position:
                means = torch.zeros((n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            else:
                means = torch.zeros((graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            means_initialized = True

        if per_position:
            input_lengths = None
        add_to_mean_hooks = [(hook_point, partial(activation_hook, index, means=means, input_lengths=input_lengths)) for hook_point, index in hook_points_indices]

        with model.hooks(fwd_hooks=add_to_mean_hooks):
            model(tokens, attention_mask=attention_mask)

    means = means.squeeze(0)
    means /= total
    return means if per_position else means.mean(0)

def make_hooks_and_matrices_test1(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Optional[Tensor], output_activation: bool=False):
    """
    Build activation difference matrix and forward/backward hook lists for attribution.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph structure for attribution.
        batch_size (int): Number of samples in the current batch.
        n_pos (int): Sequence length (number of positions).
        scores (Tensor): The score tensor to fill. If None, only used for evaluation (no backward hooks).

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: 
            - Two hook lists (forward hooks, backward hooks)
            - Activation difference tensor of shape [batch, pos, n_src_nodes, d_model], storing corrupted - clean activations
    """

    fwd_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)

    # Forward hooks (for clean/corrupted input) and backward hook lists
    fwd_hooks = []
    bwd_hooks = []

    def activation_hook(index, activations, hook):
        acts = activations.detach()
        try:
            fwd_activations[:, :, index] = acts
        except RuntimeError as e:
            print(hook.name, fwd_activations[:, :, index].size(), acts.size())
            raise e

    # Backward hook: use gradients and forward activations to update the score matrix
    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """
        Called during backward pass to compute and accumulate scores.

        Args:
            prev_index (int): Forward index of the previous node.
            bwd_index (int or slice): Backward index of the current node.
            gradients (Tensor): Gradients from the current backward pass.
            hook: Unused.
        """
        grads = gradients.detach()
        try:
            # If gradients are 3D, expand one dimension for einsum
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            elif grads.ndim == 4:
                grads = grads.sum(dim=2, keepdim=True)
            # Compute: activation difference times gradient, accumulate to scores
            s = einsum(-fwd_activations[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1).to(scores.device)  # Remove extra dimension
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, fwd_activations.size(), fwd_activations.device, grads.size(), grads.device)
            print(prev_index, bwd_index, scores.size(), s.size())
            raise e
    
    # Register forward hook for input node
    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks.append((node.out_hook, partial(activation_hook, fwd_index)))

    # Iterate over each layer, register attention and MLP node hooks
    for layer in range(graph.cfg['n_layers']):
        # Attention output node
        node = graph.nodes[f'a{layer}']
        fwd_index = graph.forward_index(node)
        fwd_hooks.append((node.out_hook, partial(activation_hook, fwd_index)))
        prev_index = graph.prev_index(node)
        # Register backward hooks for qkv inputs of attention
        for i, letter in enumerate('qkv'):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        # MLP node
        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks.append((node.out_hook, partial(activation_hook, fwd_index)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
        
    # Register only backward hook for logits node
    node = graph.nodes['logits']
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
    
    if output_activation:
        return fwd_hooks, bwd_hooks, fwd_activations
    else:
        return fwd_hooks, bwd_hooks