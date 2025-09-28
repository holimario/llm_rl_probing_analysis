from typing import List, Dict, Union, Tuple, Literal, Optional, Set
import json
import heapq

from einops import einsum
import torch
from src.transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np

from .visualization import get_color, generate_random_color

class Node:
    """
    Node class for computation graph. in_hook is the input hook in TL (TransformerLens), out_hook is the output hook.
    """
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    score: Optional[float]
    neurons: Optional[torch.Tensor]
    neurons_scores: Optional[torch.Tensor]
    qkv_inputs: Optional[List[str]]

    def __init__(self, name: str, layer: int, in_hook: List[str], out_hook: str, index: Tuple,
                 graph: 'Graph', qkv_inputs: Optional[List[str]] = None):
        # Node initialization
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.graph = graph
        self.parents = set()         # Set of parent nodes
        self.children = set()        # Set of child nodes
        self.parent_edges = set()    # Set of edges pointing to this node
        self.child_edges = set()     # Set of edges from this node
        self.qkv_inputs = qkv_inputs # For attention nodes, list of QKV input hooks

    def __repr__(self):
        return f'Node({self.name}, in_graph: {self.in_graph})'

    def __hash__(self):
        return hash(self.name)

    # Node in_graph/score/neurons_in_graph/neurons_scores state is directly fetched from graph
    @property
    def in_graph(self):
        # Whether this node is in the graph
        return self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)]

    @in_graph.setter
    def in_graph(self, value):
        # Set whether this node is in the graph
        self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def score(self):
        # Get node score
        if self.graph.nodes_scores is None:
            return None
        return self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def score(self, value):
        # Set node score
        if self.graph.nodes_scores is None:
            raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
        self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def neurons(self):
        # Get neuron mask for this node
        if self.graph.neurons is None:
            return None
        return self.graph.neurons[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def neurons(self, value):
        # Set neuron mask for this node
        if self.graph.neurons is None:
            raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
        self.graph.neurons[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def neurons_scores(self):
        # Get neuron scores for this node
        if self.graph.neurons_scores is None:
            return None
        return self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def neurons_scores(self, value):
        # Set neuron scores for this node
        if self.graph.neurons_scores is None:
            raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
        self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)] = value

class LogitNode(Node):
    # Logits node, output layer
    def __init__(self, n_layers: int, graph: 'Graph'):
        name = 'logits'
        index = slice(None)
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index, graph)

    @property
    def in_graph(self):
        # Logits node is always in the graph
        return True

    @in_graph.setter
    def in_graph(self, value):
        # Logits node cannot be removed
        raise ValueError(f"Cannot set in_graph for logits node (always True)")

class MLPNode(Node):
    # MLP node
    def __init__(self, layer: int, graph: 'Graph'):
        name = f'm{layer}'
        index = slice(None)
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index, graph)

class AttentionNode(Node):
    # Attention head node
    def __init__(self, layer: int, graph: 'Graph'):
        name = f'a{layer}'
        index = slice(None)
        # qkv_inputs: three input hooks
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.hook_attn_out", index, graph, qkv_inputs=[f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    # Input node
    def __init__(self, graph: 'Graph'):
        name = 'input'
        index = slice(None)
        super().__init__(name, 0, '', "hook_embed", index, graph)

class Edge:
    """
    Edge class in the graph.
    Attributes:
        name: (str) Name of the edge, format [PARENT]->[CHILD]<[QKV]>, QKV only applies to attention nodes
        parent: (Node) Parent node of the edge
        child: (Node) Child node of the edge
        hook: (str) Hook pointing to the child node
        index: (Tuple) Index of the child node (mainly for attention nodes)
        score: (Optional[float]) Edge score (given by attribution methods)
        in_graph: (bool) Whether this edge is in the graph
    """

    name: str
    parent: Node
    child: Node
    hook: str
    index: Tuple
    graph: 'Graph'
    def __init__(self, graph: 'Graph', parent: Node, child: Node, qkv: Optional[Literal["q", "k", "v"]] = None):
        self.graph = graph
        # Edge naming, attention edges have <q>/<k>/<v>
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent
        self.child = child
        self.qkv = qkv
        # matrix_index is used to locate this edge in the tensor
        self.matrix_index = (graph.forward_index(parent, attn_slice=False), graph.backward_index(child, qkv, attn_slice=False))

        if isinstance(child, AttentionNode):
            # Edges to attention nodes must specify qkv
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            self.index = child.index
        else:
            self.index = child.index
            self.hook = child.in_hook

    def __repr__(self):
        return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'

    def __hash__(self):
        return hash(self.name)

    @property
    def score(self):
        # Get edge score
        return self.graph.scores[self.matrix_index]

    @score.setter
    def score(self, value):
        # Set edge score
        self.graph.scores[self.matrix_index] = value

    @property
    def in_graph(self):
        # Whether this edge is in the graph
        return self.graph.in_graph[self.matrix_index]

    @in_graph.setter
    def in_graph(self, value):
        # Set whether this edge is in the graph
        self.graph.in_graph[self.matrix_index] = value

class GraphConfig(dict):
    # Graph configuration class, inherits from dict, supports dot notation
    def __init__(self, *args, **kwargs):
        super(GraphConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Graph:
    """
    Computation graph class, containing nodes and edges.

    Attributes:
        nodes (Dict[str, Node]): All nodes in the graph, key is node name, value is node object
        edges (Dict[str, Edge]): All edges in the graph, key is edge name, value is edge object
        n_forward (int): Number of forward nodes (i.e., nodes related to output activations)
        n_backward (int): Number of backward nodes (i.e., nodes related to input gradients, attention heads have 3 inputs)
        cfg (HookedTransformerConfig): Graph configuration object
    """
    nodes: Dict[str, Node]  # Mapping from node name to node object
    edges: Dict[str, Edge]  # Mapping from edge name to edge object
    n_forward: int          # Number of forward nodes
    n_backward: int         # Number of backward nodes
    scores: torch.Tensor    # (n_forward, n_backward) Edge score tensor
    in_graph: torch.Tensor  # (n_forward, n_backward) Boolean tensor for whether edge is in graph
    neurons_scores: Optional[torch.Tensor]  # (n_forward, d_model) Neuron scores
    neurons_in_graph: Optional[torch.Tensor]# (n_forward, d_model) Whether neuron is in graph
    nodes_scores: Optional[torch.Tensor]    # (n_forward) Node scores
    nodes_in_graph: torch.Tensor            # (n_forward) Whether node is in graph
    forward_to_backward: torch.Tensor       # (n_forward, n_backward) Mapping from forward to backward
    real_edge_mask: torch.Tensor            # (n_forward, n_backward) Whether this is a real edge
    cfg: GraphConfig

    def __init__(self):
        # Initialize graph structure
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(self, parent: Node, child: Node, qkv: Optional[Literal["q", "k", "v"]] = None):
        # Add an edge to the graph
        edge = Edge(self, parent, child, qkv)
        self.real_edge_mask[edge.matrix_index] = True
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def prev_index(self, node: Node) -> Union[int, slice]:
        """
        Return all forward indices that contribute to this node.
        Args:
            node (Node): Target node
        Returns:
            Union[int, slice]: Forward index
        """
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            if self.cfg['parallel_attn_mlp']:
                return 1 + node.layer * (1 + 1)
            else:
                return 1 + node.layer * (1 + 1) + 1
        elif isinstance(node, AttentionNode):
            i = 1 + node.layer * (1 + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    @classmethod
    def _n_forward(cls, cfg) -> int:
        # Compute number of forward nodes
        return 1 + cfg.n_layers * (1 + 1)

    @classmethod
    def _n_backward(cls, cfg) -> int:
        # Compute number of backward nodes
        return cfg.n_layers * (3 * 1 + 1) + 1

    @classmethod
    def _forward_index(cls, cfg, node_name: str, attn_slice: bool = False) -> int:
        """
        Return the forward index of a node given model config and node name.
        Args:
            cfg: Config object
            node_name: Node name
            attn_slice: Whether to return a slice (for attention heads)
        Returns:
            int: Forward index
        """
        if node_name == 'input':
            return 0
        elif node_name == 'logits':
            return 1 + cfg.n_layers * (1 + 1)
        elif node_name[0] == 'm':
            layer = int(node_name[1:])
            return 1 + layer * (1 + 1) + 1
        elif node_name[0] == 'a':
            layer = int(node_name[1:])
            return 1 + layer * (1 + 1)
        else:
            raise ValueError(f"Invalid node: {node_name}")

    def forward_index(self, node: Node, attn_slice=True) -> int:
        # Get forward index of a node
        return Graph._forward_index(self.cfg, node.name, attn_slice)

    @classmethod
    def _backward_index(cls, cfg, node_name: str, qkv=None, attn_slice=False) -> int:
        """
        Return the backward index of a node given model config and node name.
        Args:
            cfg: Config object
            node_name: Node name
            qkv: For attention heads, specify q/k/v
            attn_slice: Whether to return a slice
        Returns:
            int: Backward index
        """
        if node_name == 'input':
            raise ValueError(f"No backward for input node")
        elif node_name == 'logits':
            return -1
        elif node_name[0] == 'm':
            layer = int(node_name[1:])
            return (layer) * (3 * 1 + 1) + 3 * 1
        elif node_name[0] == 'a':
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            layer = int(node_name[1:])
            return layer * (3 * 1 + 1) + ('qkv'.index(qkv) * 1)
        else:
            raise ValueError(f"Invalid node: {node_name}")

    def backward_index(self, node: Node, qkv=None, attn_slice=True) -> int:
        # Get backward index of a node
        return Graph._backward_index(self.cfg, node.name, qkv, attn_slice)

    def get_dst_nodes(self) -> List[str]:
        # Get all destination node names (including qkv for attention heads, mlp, and logits)
        heads = []
        for layer in range(self.cfg['n_layers']):
            for letter in 'qkv':
                heads.append(f'a{layer}<{letter}>')
            heads.append(f'm{layer}')
        heads.append('logits')
        return heads

    def weighted_edge_count(self) -> float:
        """
        Compute weighted edge count (if neuron mask exists, weight by number of neurons, else count edges).
        Returns:
            float: Weighted edge count
        """
        if self.neurons_in_graph is not None:
            return (einsum(self.in_graph.float(), self.neurons_in_graph.float(), 'forward backward, forward d_model ->') / self.cfg['d_model']).item()
        else:
            return float(self.count_included_edges())

    def count_included_edges(self) -> int:
        # Count number of edges included in the graph
        return self.in_graph.sum().item()

    def count_included_nodes(self) -> int:
        # Count number of nodes included in the graph
        return self.nodes_in_graph.sum().item()

    def count_included_neurons(self) -> int:
        # Count number of neurons included in the graph
        return self.neurons_in_graph.sum().item()

    def reset(self, empty=True):
        """
        Reset the graph, set all contents to 0. If empty=False, set all to True.
        Args:
            empty (bool): True to clear the graph, False to select all
        """
        if empty:
            self.nodes_in_graph *= False
            self.in_graph *= False
            if self.neurons_in_graph is not None:
                self.neurons_in_graph *= False
        else:
            self.nodes_in_graph[:] = True
            self.in_graph[:] = True
            self.in_graph &= self.real_edge_mask
            if self.neurons_in_graph is not None:
                self.neurons_in_graph[:] = True

    def apply_threshold(self, threshold: float, absolute: bool = True, reset: bool = True, level: Literal['edge', 'node', 'neuron'] = 'edge', prune=True):
        """
        Select by threshold, add edges/nodes/neurons with score above threshold to the graph. Unscored nodes/neurons are kept by default.
        Args:
            threshold (float): Threshold
            absolute (bool): Whether to use absolute value
            reset (bool): Whether to reset the graph first
            level (str): Selection level ('edge'/'node'/'neuron')
            prune (bool): Whether to prune the graph
        """
        threshold = float(threshold)
        if reset:
            self.reset()

        if level == 'neuron':
            # Neuron-level selection
            unscored_neurons = torch.isnan(self.neurons_scores)
            neuron_score_copy = self.neurons_scores.clone()
            if absolute:
                neuron_score_copy = torch.abs(neuron_score_copy)
            # Set unscored neurons to inf to ensure they are kept
            neuron_score_copy[unscored_neurons] = torch.inf
            included_neurons = (neuron_score_copy >= threshold)
            self.neurons_in_graph[:] = included_neurons

            if reset:
                # After reset, activate nodes with any neuron and their outgoing edges
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == 'node':
            # Node-level selection
            unscored_nodes = torch.isnan(self.nodes_scores)
            node_score_copy = self.nodes_scores.clone()
            if absolute:
                node_score_copy = torch.abs(node_score_copy)
            node_score_copy[unscored_nodes] = torch.inf
            included_nodes = (node_score_copy >= threshold)
            self.nodes_in_graph[:] = included_nodes
            if reset:
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == 'edge':
            # Edge-level selection
            edge_scores = self.scores.clone()
            if absolute:
                edge_scores = torch.abs(edge_scores)
            # Set non-real edges to -inf
            edge_scores[~self.real_edge_mask] = -torch.inf
            surpass_threshold = edge_scores >= threshold
            self.in_graph[:] = surpass_threshold
            if reset:
                nodes_with_outgoing = self.in_graph.any(dim=1)
                nodes_with_ingoing = einsum(self.in_graph.any(dim=0).float(), self.forward_to_backward.float(), 'backward, forward backward -> forward') > 0
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing
        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune()

    def apply_topn(self, n: int, absolute: bool = True, level: Literal['edge', 'node', 'neuron'] = 'edge', reset: bool = True, prune: bool = True):
        """
        Keep only the top-n edges/nodes/neurons. Level determines selection granularity.
        Args:
            n (int): Number to keep
            absolute (bool): Whether to sort by absolute value
            reset (bool): Whether to reset the graph first
            level (str): Selection level
            prune (bool): Whether to prune the graph
        """
        if reset:
            self.reset()

        if level == 'neuron':
            # Neuron-level top-n
            scored_neurons = ~torch.isnan(self.neurons_scores)
            n_scored_neurons = scored_neurons.sum()
            assert n <= n_scored_neurons, f"Requested n ({n}) is greater than the number of scored neurons ({n_scored_neurons})"
            neuron_score_copy = self.neurons_scores.clone()
            if absolute:
                neuron_score_copy = torch.abs(neuron_score_copy)
            neuron_score_copy[~scored_neurons] = -torch.inf
            sorted_neurons = torch.argsort(neuron_score_copy.view(-1), descending=True)
            # Set top-n neurons to True, others to False
            self.neurons_in_graph.view(-1)[sorted_neurons[:n]] = True
            self.neurons_in_graph.view(-1)[sorted_neurons[n:]] = False
            # Force keep unscored neurons
            self.neurons_in_graph.view(-1)[~scored_neurons.view(-1)] = True
            if reset:
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == 'node':
            # Node-level top-n
            scored_nodes = ~torch.isnan(self.nodes_scores)
            n_scored_nodes = scored_nodes.sum()
            assert n <= n_scored_nodes, f"Requested n ({n}) is greater than the number of scored nodes ({n_scored_nodes})"
            node_score_copy = self.nodes_scores.clone()
            if absolute:
                node_score_copy = torch.abs(node_score_copy)
            node_score_copy[~scored_nodes] = -torch.inf
            sorted_nodes = torch.argsort(node_score_copy.view(-1), descending=True)
            self.nodes_in_graph.view(-1)[sorted_nodes[:n]] = True
            self.nodes_in_graph.view(-1)[sorted_nodes[n:]] = False
            self.nodes_in_graph.view(-1)[~scored_nodes.view(-1)] = True
            if reset:
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == 'edge':
            # Edge-level top-n
            assert n <= self.real_edge_mask.sum(), f"Requested n ({n}) is greater than the number of edges ({self.real_edge_mask.sum()})"
            edge_scores = self.scores.clone()
            if absolute:
                edge_scores = torch.abs(edge_scores)
            edge_scores[~self.real_edge_mask] = -torch.inf
            sorted_edges = torch.argsort(edge_scores.view(-1), descending=True)
            self.in_graph.view(-1)[sorted_edges[:n]] = True  # Mark which edges are in the graph
            self.in_graph.view(-1)[sorted_edges[n:]] = False
            if reset:
                nodes_with_outgoing = self.in_graph.any(dim=1)

                # Because attention structure has multiple input heads, need this handling
                nodes_with_ingoing = einsum(self.in_graph.any(dim=0).float(), self.forward_to_backward.float(), 'backward, forward backward -> forward') > 0
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing

        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune()

    def apply_greedy(self, n_edges: int, absolute: bool = True, reset: bool = True, prune: bool = True):
        """
        Use greedy algorithm to select top-n edges from logits upwards. Only applies to edge level.
        Args:
            n_edges (int): Number of edges to keep
            reset (bool): Whether to reset the graph first
            absolute (bool): Whether to sort by absolute value
        """
        if n_edges > len(self.edges):
            raise ValueError(f"n ({n_edges}) is greater than the number of edges ({len(self.edges)})")

        if reset:
            self.nodes_in_graph *= False
            self.in_graph *= False

        def abs_id(s: float):
            return abs(s) if absolute else s

        # Only consider edges whose child node is in the graph
        candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph], key=lambda edge: abs_id(edge.score), reverse=True)

        edges = heapq.merge(candidate_edges, key=lambda edge: abs_id(edge.score), reverse=True)
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key=lambda edge: abs_id(edge.score), reverse=True)
                edges = heapq.merge(edges, parent_parent_edges, key=lambda edge: abs_id(edge.score), reverse=True)

        if prune:
            self.prune()

    def prune(self):
        """
        Prune the graph to ensure connectivity. Remove useless nodes/edges/neurons.
        Steps:
            1. Remove nodes without any neurons
            2. Iteratively remove nodes without incoming or outgoing edges
            3. Remove edges whose parent/child node is not in the graph
            4. Remove neurons of nodes not in the graph
        """

        # Remove nodes without any neurons
        if self.neurons_in_graph is not None:
            self.nodes_in_graph *= self.neurons_in_graph.any(dim=1)

        old_new_same = False
        # May need multiple iterations
        while not old_new_same:
            # Remove nodes without outgoing or incoming edges
            nodes_with_outgoing = self.in_graph.any(dim=1)  # Whether this node's output is used as input to any other node
            nodes_with_ingoing = einsum(self.in_graph.any(dim=0).float(), self.forward_to_backward.float(), 'backward, forward backward -> forward') > 0  # Whether this node receives output from any other node
            nodes_with_ingoing[0] = True  # Input node always has incoming edge
            old_nodes_in_graph = self.nodes_in_graph.clone()
            self.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing

            # Remove edges whose parent/child node is not in the graph
            forward_in_graph = self.nodes_in_graph.float()
            backward_in_graph = (self.nodes_in_graph.float() @ self.forward_to_backward.float())
            backward_in_graph[-1] = 1  # Logits node is always kept
            edge_remask = einsum(forward_in_graph, backward_in_graph, 'forward, backward -> forward backward') > 0
            old_edges_in_graph = self.in_graph.clone()
            self.in_graph *= edge_remask
            old_new_same = (
                torch.all(old_nodes_in_graph == self.nodes_in_graph) and
                torch.all(old_edges_in_graph == self.in_graph)
            )

        # Remove neurons of nodes not in the graph
        if self.neurons_in_graph is not None:
            self.neurons_in_graph *= self.nodes_in_graph.view(-1, 1)

    @classmethod
    def from_model(cls, model_or_config: Union[HookedTransformer, HookedTransformerConfig, Dict], neuron_level: bool = False, node_scores: bool = False) -> 'Graph':
        """
        Instantiate Graph from model or config object.
        Args:
            model_or_config: HookedTransformer/HookedTransformerConfig/Dict
            neuron_level (bool): Whether to enable neuron-level
            node_scores (bool): Whether to enable node scores
        Returns:
            Graph: Constructed graph
        """
        graph = Graph()
        graph.cfg = GraphConfig()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg.update({'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp': cfg.parallel_attn_mlp, 'd_model': cfg.d_model})
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg.update({'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp': cfg.parallel_attn_mlp, 'd_model': cfg.d_model})
        elif isinstance(model_or_config, dict):
            graph.cfg.update(model_or_config)
        else:
            raise ValueError(f"Invalid input type: {type(model_or_config)}")

        graph.n_forward = 1 + graph.cfg['n_layers'] * (1 + 1)
        graph.n_backward = graph.cfg['n_layers'] * (3 * 1 + 1) + 1
        graph.forward_to_backward = torch.zeros((graph.n_forward, graph.n_backward)).bool()

        graph.scores = torch.zeros((graph.n_forward, graph.n_backward))
        graph.real_edge_mask = torch.zeros((graph.n_forward, graph.n_backward)).bool()
        graph.in_graph = torch.zeros((graph.n_forward, graph.n_backward)).bool()
        graph.nodes_in_graph = torch.zeros(graph.n_forward).bool()
        if node_scores:
            graph.nodes_scores = torch.zeros(graph.n_forward)
            graph.nodes_scores[:] = torch.nan
        else:
            graph.nodes_scores = None
        if neuron_level:
            graph.neurons_in_graph = torch.zeros((graph.n_forward, graph.cfg['d_model'])).bool()
            graph.neurons_scores = torch.zeros((graph.n_forward, graph.cfg['d_model']))
            graph.neurons_scores[:] = torch.nan
        else:
            graph.neurons_in_graph = None
            graph.neurons_scores = None

        input_node = InputNode(graph)
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg['n_layers']):
            attn_node = AttentionNode(layer, graph)
            mlp_node = MLPNode(layer, graph)

            graph.nodes[attn_node.name] = attn_node
            for letter in 'qkv':
                graph.forward_to_backward[graph.forward_index(attn_node, attn_slice=False), graph.backward_index(attn_node, attn_slice=False, qkv=letter)] = True

            graph.nodes[mlp_node.name] = mlp_node
            graph.forward_to_backward[graph.forward_index(mlp_node, attn_slice=False), graph.backward_index(mlp_node, attn_slice=False)] = True

            if graph.cfg['parallel_attn_mlp']:
                # Parallel attn+mlp
                for node in residual_stream:
                    for letter in 'qkv':
                        graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)

                residual_stream.append(attn_node)
                residual_stream.append(mlp_node)

            else:
                # Serial attn+mlp
                for node in residual_stream:
                    for letter in 'qkv':
                        graph.add_edge(node, attn_node, qkv=letter)
                residual_stream.append(attn_node)

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg['n_layers'], graph)
        for node in residual_stream:
            graph.add_edge(node, logit_node)

        graph.nodes[logit_node.name] = logit_node

        return graph

    def to_json(self, filename: str):
        """
        Export the graph to a JSON file.
        Args:
            filename (str): File name
        """
        # Non-serializable info
        d = {'cfg': dict(self.cfg)}
        node_dict = {}
        for node_name, node in self.nodes.items():
            node_dict[node_name] = {'in_graph': bool(node.in_graph)}
            if self.nodes_scores is not None:
                node_dict[node_name]['score'] = float(node.score)
            if self.neurons_in_graph is not None:
                node_dict[node_name]['neurons'] = self.neurons_in_graph[self.forward_index(node)].tolist()
                node_dict[node_name]['neurons_scores'] = self.neurons_scores[self.forward_index(node)].tolist()
        d['nodes'] = node_dict

        edge_dict = {}
        for edge_name, edge in self.edges.items():
            edge_dict[edge_name] = {'score': edge.score.item(), 'in_graph': bool(edge.in_graph)}

        d['edges'] = edge_dict

        with open(filename, 'w') as f:
            json.dump(d, f)

    def to_pt(self, filename: str):
        """
        Export the graph to a .pt file.
        Args:
            filename (str): File name
        """
        src_nodes = [node.name for node in self.nodes.values() if not isinstance(node, LogitNode)]
        dst_nodes = self.get_dst_nodes()
        d = {'cfg': dict(self.cfg), 'src_nodes': src_nodes, 'dst_nodes': dst_nodes, 'edges_scores': self.scores, 'edges_in_graph': self.in_graph, 'nodes_in_graph': self.nodes_in_graph}
        if self.nodes_scores is not None:
            d['nodes_scores'] = self.nodes_scores
        if self.neurons_in_graph is not None:
            d['neurons_in_graph'] = self.neurons_in_graph
            d['neurons_scores'] = self.neurons_scores
        torch.save(d, filename)

    @classmethod
    def from_json(cls, json_path: str) -> 'Graph':
        """
        Load a graph object from a JSON file.
        JSON must contain the following keys:
            1. 'cfg': config dict
            2. 'nodes': mapping from node name to in_graph/score/neuron mask
            3. 'edges': mapping from edge name to score and in_graph
            4. 'neurons': optional, mapping from node name to neuron mask

        Note: Not suitable for very large graphs at neuron resolution.
        """
        with open(json_path, 'r') as f:
            d = json.load(f)
            assert all([k in d.keys() for k in ['cfg', 'nodes', 'edges']]), "Bad input JSON format - Missing keys"

        g = Graph.from_model(d['cfg'], neuron_level=True, node_scores=True)
        any_node_scores, any_neurons, any_neurons_scores = False, False, False
        for name, node_dict in d['nodes'].items():
            if name == 'logits':
                continue
            g.nodes[name].in_graph = node_dict['in_graph']
            if 'score' in node_dict:
                any_node_scores = True
                g.nodes[name].score = node_dict['score']
            if 'neurons' in node_dict:
                any_neurons = True
                g.neurons_in_graph[g.forward_index(g.nodes[name])] = torch.tensor(node_dict['neurons']).float()
            if 'neurons_scores' in node_dict:
                any_neurons_scores = True
                g.neurons_scores[g.forward_index(g.nodes[name])] = torch.tensor(node_dict['neurons_scores']).float()

        if not any_node_scores:
            g.nodes_scores = None
        if not any_neurons:
            g.neurons_in_graph = None
        if not any_neurons_scores:
            g.neurons_scores = None

        for name, info in d['edges'].items():
            g.edges[name].score = info['score']
            g.edges[name].in_graph = info['in_graph']

        return g

    @classmethod
    def from_pt(cls, pt_path: str) -> 'Graph':
        """
        Load a graph object from a pytorch serialized file.
        File must contain:
            1. 'cfg': config dict
            2. 'src_nodes': list of source node names
            3. 'dst_nodes': list of destination node names
            4. 'edges_scores': edge score tensor
            5. 'edges_in_graph': whether edge is in graph
            6. 'neurons': [optional] neuron mask
        """
        d = torch.load(pt_path)
        required_keys = ['cfg', 'src_nodes', 'dst_nodes', 'edges_scores', 'edges_in_graph', 'nodes_in_graph']
        assert all([k in d.keys() for k in required_keys]), f"Bad torch circuit file format. Found keys - {d.keys()}, missing keys - {set(required_keys) - set(d.keys())}"
        assert d['edges_scores'].shape == d['edges_in_graph'].shape, "Bad edges array shape"

        g = Graph.from_model(d['cfg'])

        g.in_graph[:] = d['edges_in_graph']
        g.scores[:] = d['edges_scores']
        g.nodes_in_graph[:] = d['nodes_in_graph']

        if 'nodes_scores' in d:
            g.nodes_scores = d['nodes_scores']

        if 'neurons_in_graph' in d:
            g.neurons_in_graph = d['neurons_in_graph']

        if 'neurons_scores' in d:
            g.neurons_scores = d['neurons_scores']

        return g

    def to_image(
        self,
        filename: str,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str = "dot",
        seed: Optional[int] = None
    ):

        """
        Export the graph as a .png image.
        Args:
            filename: File name
            colorscheme: Color scheme
            minimum_penwidth: Minimum edge width
            maximum_penwidth: Maximum edge width
            layout: Graph layout
            seed: Random seed
        """
        import pygraphviz as pgv
        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        # Assign a color to each node
        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(node.name,
                           fillcolor=colors[node.name],
                           color="black",
                           style="filled, rounded",
                           shape="box",
                           fontname="Helvetica",
                           )

        scores = self.scores.view(-1).abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                normalized_score = (abs(edge.score) - min_score) / (max_score - min_score) if max_score != min_score else abs(edge.score)
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(edge.parent.name,
                           edge.child.name,
                           penwidth=str(penwidth),
                           color=get_color(edge.qkv, edge.score),
                           )
        g.draw(filename, prog="dot")