from typing import Callable, List, Optional, Literal, Tuple
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from src.transformer_lens import HookedTransformer

from tqdm import tqdm

from .my_utils_non_heads import tokenize_plus, make_hooks_and_matrices, compute_mean_activations, token_padding, make_hooks_and_matrices_test1
from .my_evaluate_non_heads import evaluate_graph, evaluate_baseline
from .my_graph_non_heads import Graph

def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                     intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching', 
                     intervention_dataloader: Optional[DataLoader] = None, quiet=False):
    """
    Get scores using the exact patching method, by repeatedly calling evaluate_graph.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']): Intervention method, default is 'patching'.
        intervention_dataloader (Optional[DataLoader]): Dataset used to compute means, default is None.
        quiet (bool): Whether to run in quiet mode, default is False.
    """

    graph.in_graph |= graph.real_edge_mask  # Add all real edges to the graph
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()  # Compute baseline score
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False  # Remove current edge
        intervened_performance = evaluate_graph(
            model, graph, dataloader, metric, 
            intervention=intervention, 
            intervention_dataloader=intervention_dataloader, 
            quiet=True, skip_clean=True
        ).mean().item()  # Evaluate performance after removing the edge
        edge.score = intervened_performance - baseline  # Compute score
        edge.in_graph = True  # Restore the edge

    # Only to keep return type consistent; actual scores are updated in the score matrix
    return graph.scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching', 
                   intervention_dataloader: Optional[DataLoader] = None, quiet=False, device='cuda'):
    """
    Get edge attribution scores using the EAP method.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']): Intervention method.
        intervention_dataloader (Optional[DataLoader]): Dataset used to compute means.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device, default is 'cuda'.

    Returns:
        Tensor: Score tensor of shape [src_nodes, dst_nodes].
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)    

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Mean intervention requires intervention_dataloader"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with torch.inference_mode():
            if intervention == 'patching':
                # patching: replace clean activations with corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif 'mean' in intervention:
                # zero/mean ablation: do not add corrupted activations, but mean needs to add means
                activation_difference += means

            # Some metrics (e.g., accuracy or KL) require clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False, device='cuda'):
    """
    Get edge attribution scores using EAP combined with Integrated Gradients (IG).

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        steps (int): Number of IG steps, default is 30.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device.

    Returns:
        Tensor: Score tensor of shape [src_nodes, dst_nodes].
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)

        if n_pos != n_pos_corrupted:
            print(f"Number of positions mismatch: {n_pos} (clean) != {n_pos_corrupted} (corrupted)")
            print(clean)
            print(corrupted)
            raise ValueError("Number of positions must match")

        # Get forward/backward hooks and activation difference matrix
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            # Input interpolation hook for IG
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                if torch.isnan(metric_value).any().item():
                    print("Metric value is NaN")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    print(f"Label: {label}")
                    print(f"Metric: {metric}")
                    raise ValueError("Metric value is NaN")
                metric_value.backward()
            
            if torch.isnan(scores).any().item():
                print("Score is NaN")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                print(f"Label: {label}")
                print(f"Metric: {metric}")
                print(f'Step: {step}')
                raise ValueError("Score is NaN")

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                              metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching', 
                              steps=30, intervention_dataloader: Optional[DataLoader] = None, quiet=False, device='cuda'):
    """
    Get scores using activation interpolation with Integrated Gradients.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']): Intervention method.
        steps (int): Number of IG steps.
        intervention_dataloader (Optional[DataLoader]): Dataset used to compute means.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device.

    Returns:
        Tensor: Score tensor of shape [src_nodes, dst_nodes].
    """

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Mean intervention requires intervention_dataloader"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # Get various hooks and activation differences
        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)
            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            # Output interpolation hook
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k / steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output
            return hook_fn

        total_steps = 0

        # Build the list of nodes to interpolate
        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps + 1):
                total_steps += 1
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                               metric: Callable[[Tensor], Tensor], quiet=False, device='cuda'):
    """
    Get scores using the clean-corrupted method: similar to EAP-IG, but only on clean and corrupted inputs, without intermediate interpolation.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device.

    Returns:
        Tensor: Score tensor.
    """

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, input_lengths, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_information_flow_routes(model: HookedTransformer, graph: Graph, dataloader: DataLoader, quiet=False, device='cuda') -> torch.Tensor:
    """
    Get scores using the information flow routes method proposed by Ferrando et al. (2024).

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device.

    Returns:
        Tensor: Scores based on information flow routes.
    """
    # This could hack make_hooks_and_matrices, but is implemented directly here
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)    

    def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
        # Save output activations for each node
        output_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)

        def output_hook(index, activations, hook):
            # Output hook, save activations
            try:
                acts = activations.detach()
                output_activations[:, :, index] = acts
            except RuntimeError as e:
                print(hook.name, output_activations[:, :, index].size(), output_activations.size())
                raise e

        def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
            # Input hook, directly compute scores
            acts = activations.detach()
            try:
                if acts.ndim == 3:
                    acts = acts.unsqueeze(2)
                # acts: batch pos backward hidden
                # output_activations: batch pos forward hidden
                acts = acts.unsqueeze(2)
                unsqueezed_output_activations = output_activations.unsqueeze(3)

                # acts: batch pos 1 backward hidden
                # output_activations: batch pos forward 1 hidden
                proximity = torch.clamp(
                    - torch.linalg.vector_norm(unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1, dim=-1)
                    + torch.linalg.vector_norm(acts, ord=1, dim=-1), min=0
                )
                importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
                # importance: batch pos forward backward
                # Aggregate over positions (sum/mean), get forward backward
                # First mask out padding positions
                max_len = input_lengths.max()
                mask = torch.arange(max_len, device=input_lengths.device,
                            dtype=input_lengths.dtype).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                importance *= mask
                importance = importance.sum(1) / input_lengths.view(-1,1,1) # Mean over positions
                importance = importance.sum(0)

                # importance: forward backward
                # Squeeze backward dim (e.g., for MLP)
                importance = importance.squeeze(1)
                scores[:prev_index, bwd_index] += importance

            except RuntimeError as e:
                print(hook.name, unsqueezed_output_activations[:, :, prev_index].size(), acts.size())
                raise e
            
        hooks = []
        node = graph.nodes['input']
        fwd_index = graph.forward_index(node)
        hooks.append((node.out_hook, partial(output_hook, fwd_index)))
        
        for layer in range(graph.cfg['n_layers']):
            node = graph.nodes[f'a{layer}.h0']
            fwd_index = graph.forward_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            prev_index = graph.prev_index(node)
            for i, letter in enumerate('qkv'):
                bwd_index = graph.backward_index(node, qkv=letter)
                hooks.append((node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths)))

            node = graph.nodes[f'm{layer}']
            fwd_index = graph.forward_index(node)
            bwd_index = graph.backward_index(node)
            prev_index = graph.prev_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
            
        node = graph.nodes['logits']
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
        return hooks
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, _, _ in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)

        hooks = make_hooks(n_pos, input_lengths)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                _ = model(clean_tokens, attention_mask=attention_mask)

    scores /= total_items

    return scores


def get_scores_test1(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching', 
                   intervention_dataloader: Optional[DataLoader] = None, quiet=False, device='cuda', out_out_all_score=True):
    """
    Get edge attribution scores using the EAP method.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']): Intervention method.
        intervention_dataloader (Optional[DataLoader]): Dataset used to compute means.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device, default is 'cuda'.

    Returns:
        Tensor: Score tensor of shape [src_nodes, dst_nodes].
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    if out_out_all_score:
        all_scores = torch.zeros((len(dataloader), graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    i = 0
    for all_tokens, answer_range, origin_answer_token_ids in dataloader:
        batch_size = len(all_tokens)
        total_items += batch_size
        padded_tokens, attention_mask, input_lengths, n_pos = token_padding(model, all_tokens, model.tokenizer.pad_token_id)

        temp_score = scores.clone()

        fwd_hooks, bwd_hooks = make_hooks_and_matrices_test1(
            model, graph, batch_size, n_pos, scores
        )

        with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
            logits = model(padded_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, input_lengths, answer_range, origin_answer_token_ids)
            metric_value.backward()
        
        all_scores[i] = scores - temp_score
        i += batch_size

    scores /= total_items

    if out_out_all_score:
        return scores, all_scores
    else:
        return scores



allowed_aggregations = {'sum', 'mean'}    
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], 
              method: Literal['test1'], 
              intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching', aggregation='sum', 
              ig_steps: Optional[int] = None, intervention_dataloader: Optional[DataLoader] = None, quiet=False, device='cuda'):
    """
    Main attribution function, selects different attribution implementations according to the method.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph to attribute.
        dataloader (DataLoader): The data used for attribution.
        metric (Callable[[Tensor], Tensor]): The metric used for attribution.
        method (Literal[...]): Attribution method.
        intervention (Literal[...]): Intervention method.
        aggregation (str): Aggregation method, 'sum' or 'mean'.
        ig_steps (Optional[int]): Number of IG steps.
        intervention_dataloader (Optional[DataLoader]): Dataset used to compute means.
        quiet (bool): Whether to run in quiet mode.
        device (str): Device.

    Returns:
        None, scores are written to graph.scores
    """
    # assert model.cfg.use_attn_result, "Model must be configured with use_attn_result"
    assert model.cfg.use_split_qkv_input, "Model must be configured with use_split_qkv_input"
    assert model.cfg.use_hook_mlp_in, "Model must be configured with use_hook_mlp_in"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured with ungroup_grouped_query_attention"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be one of {allowed_aggregations}, got {aggregation}')
        
    # By default, sum over d_model dimension, scores are [n_src_nodes, n_dst_nodes] tensor
    if method == 'test1':
        scores, all_scores = get_scores_test1(
            model, graph, dataloader, metric, 
            intervention=intervention, 
            intervention_dataloader=intervention_dataloader, 
            quiet=quiet, device=device
        )
    else:
        raise ValueError(
            f"method must be one of [], got {method}"
        )

    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] = scores.to(graph.scores.device)

    return all_scores
