from typing import Callable, List, Union, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from src.transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from .my_utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations, token_padding, make_hooks_and_matrices_test1
from .my_graph import Graph, AttentionNode


def evaluate_graph(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metrics: Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]],
    quiet: bool = False,
    intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'zero',
    intervention_dataloader: Optional[DataLoader] = None,
    skip_clean: bool = True
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate a circuit (i.e., a graph with some nodes set to False, usually created by calling graph.apply_threshold).
    Pruning is usually required beforehand to ensure the circuit is valid.

    Args:
        model (HookedTransformer): The model to run the circuit on.
        graph (Graph): The circuit graph to evaluate.
        dataloader (DataLoader): The dataset for evaluation.
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): Evaluation metrics, can be a single function or a list of functions.
        quiet (bool, optional): Whether to disable tqdm progress bar. Default is False.
        intervention (Literal['patching', 'zero', 'mean','mean-positional'], optional): Type of intervention.
            'patching' means swap intervention; 'mean-positional' means take mean by position over the given dataset. Default is 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): Dataset for computing means, required only for mean interventions.
        skip_clean (bool, optional): Whether to skip computing clean logits. Default is True.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: One or more tensors of faithfulness scores.
    """
    # Check model config to ensure attention result is used
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    if model.cfg.n_key_value_heads is not None:
        # If grouped key-value heads are used, ensure config is correct
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    # Check if intervention type is valid
    assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"

    # For mean interventions, intervention_dataloader must be provided and means computed
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    # Prune the graph to remove invalid components and ensure connectivity
    graph.prune()

    # Build a matrix indicating which edges are in the graph
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

    # Do the same for neurons if applicable
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

        # If an edge is in the graph but not all its neurons are, still need to update the edge
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # # Invert the matrix as a mask to specify which edges should be "broken"
    # in_graph_matrix = 1 - in_graph_matrix
    # if neuron_matrix is not None:
    #     neuron_matrix = 1 - neuron_matrix

    # For now, do not invert
    in_graph_matrix = in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    if model.cfg.use_normalization_before_and_after:
        # If the model uses normalization before and after attention head output, handle specially
        attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device='cuda', dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        # Mask for non-attention heads
        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        # Attention biases for all layers
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    # For each node, construct an input hook. If the corresponding edge is not in the graph, "break" its input
    # by adding the activation difference (difference between clean and corrupted activations)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # For gemma models (with normalization after attention)
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]

                # Get clean outputs for all previous attention heads
                clean_attention_results = einsum(
                    activation_matrix[1, :, :, :len(in_graph_vector)],
                    attention_head_mask[:len(in_graph_vector)],
                    'batch pos previous hidden, previous layer -> batch pos layer hidden'
                )

                # Compute update for non-attention heads and difference for clean/corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(
                        activation_differences[:, :, :len(in_graph_vector)],
                        neuron_matrix[:len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[:len(in_graph_vector)],
                        'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden'
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, :len(in_graph_vector)],
                        neuron_matrix[:len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[:len(in_graph_vector)],
                        'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden'
                    )
                else:
                    non_attention_update = einsum(
                        activation_differences[:, :, :len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[:len(in_graph_vector)],
                        'batch pos previous hidden, previous ..., previous -> batch pos ... hidden'
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, :len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[:len(in_graph_vector)],
                        'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden'
                    )

                # Add bias to attention results and compute corrupted attention results using the difference
                # If in_graph_vector is 2D, we are handling all attention heads
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

                # Both clean and corrupted attention results go through layernorm, and the difference is added to the update
                update = non_attention_update
                valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])

            else:
                # For non-gemma models, the process is simpler
                activation_save = activation_matrix.to(activations.device)
                # ... is used for compatibility with the head dimension of attention layers
                if neuron_matrix is not None:
                    update = einsum(
                        activation_save[:, :, :len(in_graph_vector)],
                        neuron_matrix[:len(in_graph_vector)],
                        in_graph_vector,
                        'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden'
                    )
                else:
                    update = einsum(
                        activation_save[:, :, :len(in_graph_vector)],
                        in_graph_vector,
                        'batch pos previous hidden, previous ... -> batch pos ... hidden'
                    )
            activations = update  # Directly replace here
            return activations
        return input_construction_hook

    # Construct all required input construction hooks
    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in this layer is in the graph, construct input hook for the whole layer
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)) and \
                not (neuron_matrix is None and all(parent_edge.in_graph for head in range(model.cfg.n_heads) for parent_edge in graph.nodes[f'a{layer}.h{head}'].parent_edges)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))

            # If the MLP node in this layer is in the graph, also add MLP hook
            if graph.nodes[f'm{layer}'].in_graph and \
                not (neuron_matrix is None and all(parent_edge.in_graph for parent_edge in graph.nodes[f'm{layer}'].parent_edges)):
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))

        # The logits node always needs a hook
        if not (neuron_matrix is None and all(parent_edge.in_graph for parent_edge in graph.nodes['logits'].parent_edges)):
            node = graph.nodes['logits']
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks

    # If metrics is not a list, convert to list
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]

    # Actually run/evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    for all_tokens, answer_range, origin_answer_token_ids in dataloader:
        # Tokenize clean and corrupted samples
        padded_tokens, attention_mask, input_lengths, n_pos = token_padding(model, all_tokens, model.tokenizer.pad_token_id)

        # fwd_hooks_corrupted: used to add corrupted activations to activation_difference
        # fwd_hooks_clean: used to subtract clean activations
        # activation_difference shape: (batch, pos, src_nodes, hidden)
        fwd_hooks, _, fwd_activations = make_hooks_and_matrices_test1(model, graph, len(all_tokens), n_pos, None, output_activation=True)

        # Construct input hooks
        input_construction_hooks = make_input_construction_hooks(fwd_activations, in_graph_matrix, neuron_matrix)
        with torch.inference_mode():
            # # Some metrics (e.g., accuracy, KL) require clean logits
            # clean_logits = None if skip_clean else model(clean_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks + input_construction_hooks):
                logits = model(padded_tokens, attention_mask=attention_mask)

        # Compute all metrics
        for i, metric in enumerate(metrics):
            r = metric(logits, input_lengths, answer_range, origin_answer_token_ids).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    # Concatenate results from all batches
    results = [torch.cat(rs) for rs in results]
    # If only one metric, return tensor directly
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_baseline(
    model: HookedTransformer,
    dataloader: DataLoader,
    metrics: List[Callable[[Tensor], Tensor]],
    run_corrupted: bool = False,
    quiet: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate the model on the given dataset without any intervention. Used to compute the model's baseline performance.

    Args:
        model (HookedTransformer): The model to evaluate.
        dataloader (DataLoader): The dataset for evaluation.
        metrics (List[Callable[[Tensor], Tensor]]): List of evaluation metrics.
        run_corrupted (bool, optional): Whether to evaluate on corrupted samples. Default is False.
        quiet (bool, optional): Whether to disable the progress bar. Default is False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: One or more tensors of performance scores.
    """
    # If metrics is not a list, convert to list
    if not isinstance(metrics, list):
        metrics = [metrics]

    results = [[] for _ in metrics]
    if not quiet:
        dataloader = tqdm(dataloader)
    for all_tokens, answer_range, origin_answer_token_ids in dataloader:
        # Tokenize clean and corrupted samples
        padded_tokens, attention_mask, input_lengths, n_pos = token_padding(model, all_tokens, model.tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(padded_tokens, attention_mask=attention_mask)
        for i, metric in enumerate(metrics):
            r = metric(logits, input_lengths, answer_range, origin_answer_token_ids).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    # Concatenate results from all batches
    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results
