from __future__ import annotations

"""Hook Points.

Helpers to access activations in models.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn
import torch.utils.hooks as hooks
from torch import Tensor

from src.transformer_lens.utils import Slice, SliceInput


@dataclass
class LensHandle:
    """Dataclass that holds information about a PyTorch hook."""

    hook: hooks.RemovableHandle
    """Reference to the Hook's Removable Handle."""

    is_permanent: bool = False
    """Indicates if the Hook is Permanent."""

    context_level: Optional[int] = None
    """Context level associated with the hooks context manager for the given hook."""


# Define type aliases
NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]


@runtime_checkable
class _HookFunctionProtocol(Protocol):
    """Protocol for hook functions."""

    def __call__(self, tensor: Tensor, *, hook: "HookPoint") -> Union[Any, None]:
        ...


HookFunction = _HookFunctionProtocol  # Callable[..., _HookFunctionProtocol]

DeviceType = Optional[torch.device]
_grad_t = Union[tuple[Tensor, ...], Tensor]


class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """

    def __init__(self):
        super().__init__()
        self.fwd_hooks: list[LensHandle] = []
        self.bwd_hooks: list[LensHandle] = []
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name: Optional[str] = None

    def add_perma_hook(self, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd") -> None:
        self.add_hook(hook, dir=dir, is_permanent=True)

    def add_hook(
        self,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
    ) -> None:
        """
        Add a hook to the HookPoint.

        Args:
            hook: HookFunction, the hook function, format fn(activation, hook), where activation is the activation value, hook is the current HookPoint instance.
            dir: str, "fwd" for forward hook, "bwd" for backward hook, default is "fwd".
            is_permanent: bool, whether the hook is permanent (not removed by context manager), default is False.
            level: Optional[int], context_level of the context manager, used to distinguish nested hook scopes.
            prepend: bool, whether to insert the hook at the beginning of the hook list (higher priority), default is False.

        Functionality:
            - Wraps the user-defined hook function as a PyTorch hook (forward/backward hook).
            - Supports both forward and backward hooks.
            - Supports hook priority (prepend).
            - Supports permanent and context-level hooks.
        """

        def full_hook(
            module: torch.nn.Module,
            module_input: Any,
            module_output: Any,
        ):
            # For backward hooks, module_output is a tuple (grad,), need to take the first element
            if dir == "bwd":
                module_output = module_output[0]
            # Call the user-defined hook function, passing activation and current HookPoint instance
            return hook(module_output, hook=self)

        # Set __name__ attribute for full_hook for debugging and tracing
        if isinstance(hook, partial):
            # partial.__repr__() can be slow if arguments are large objects, only show function name here
            full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
        else:
            full_hook.__name__ = hook.__repr__()

        # Register forward or backward hook according to direction
        if dir == "fwd":
            pt_handle = self.register_forward_hook(full_hook, prepend=prepend)
            visible_hooks = self.fwd_hooks
        elif dir == "bwd":
            pt_handle = self.register_full_backward_hook(full_hook, prepend=prepend)
            visible_hooks = self.bwd_hooks
        else:
            raise ValueError(f"Invalid direction {dir}")

        # Wrap RemovableHandle with LensHandle, record permanence and context level
        handle = LensHandle(pt_handle, is_permanent, level)

        if prepend:
            # If prepend=True, insert hook at the beginning (higher priority)
            visible_hooks.insert(0, handle)
        else:
            # Default: append to the end
            visible_hooks.append(handle)

    def remove_hooks(
        self,
        dir: Literal["fwd", "bwd", "both"] = "fwd",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ) -> None:
        def _remove_hooks(handles: list[LensHandle]) -> list[LensHandle]:
            output_handles = []
            for handle in handles:
                if including_permanent:
                    handle.hook.remove()
                elif (not handle.is_permanent) and (level is None or handle.context_level == level):
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles

        if dir == "fwd" or dir == "both":
            self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        if dir == "bwd" or dir == "both":
            self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def forward(self, x: Tensor) -> Tensor:
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        if self.name is None:
            raise ValueError("Name cannot be None")
        split_name = self.name.split(".")
        return int(split_name[1])


# %%
class HookedRootModule(nn.Module):
    """A class building on nn.Module to interface nicely with HookPoints.

    Adds various nice utilities, most notably run_with_hooks to run the model with temporary hooks,
    and run_with_cache to run the model on some input and return a cache of all activations.

    Notes:

    The main footgun with PyTorch hooking is that hooks are GLOBAL state. If you add a hook to the
    module, and then run it a bunch of times, the hooks persist. If you debug a broken hook and add
    the fixed version, the broken one is still there. To solve this, run_with_hooks will remove
    hooks at the end by default, and I recommend using the API of this and run_with_cache. If you
    want to add hooks into global state, I recommend being intentional about this, and I recommend
    using reset_hooks liberally in your code to remove any accidentally remaining global state.

    The main time this goes wrong is when you want to use backward hooks (to cache or intervene on
    gradients). In this case, you need to keep the hooks around as global state until you've run
    loss.backward() (and so need to disable the reset_hooks_end flag on run_with_hooks)
    """

    name: Optional[str]
    mod_dict: dict[str, nn.Module]
    hook_dict: dict[str, HookPoint]

    def __init__(self, *args: Any):
        super().__init__()
        self.is_caching = False
        self.context_level = 0

    def setup(self):
        """
        Helper method to initialize the model.

        This method must be called in the model's `__init__` after all layers are defined.
        Main functions:
        1. Assigns a unique name (name) to each submodule (including itself), for later indexing by name.
        2. Builds a dictionary (mod_dict) mapping module names to module instances, for convenient access by name.
        3. For modules of type HookPoint, also adds them to hook_dict for unified management.

        Note:
        - Only modules with non-empty names are processed (the root module name "" is skipped).
        - HookPoint is a special module type for inserting and managing forward/backward hooks.
        """
        self.mod_dict = {}
        self.hook_dict = {}
        for name, module in self.named_modules():
            if name == "":
                continue
            module.name = name
            self.mod_dict[name] = module
            if isinstance(module, HookPoint):
                self.hook_dict[name] = module

    def hook_points(self):
        return self.hook_dict.values()

    def remove_all_hook_fns(
        self,
        direction: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ):
        for hp in self.hook_points():
            hp.remove_hooks(direction, including_permanent=including_permanent, level=level)

    def clear_contexts(self):
        for hp in self.hook_points():
            hp.clear_context()

    def reset_hooks(
        self,
        clear_contexts: bool = True,
        direction: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ):
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction, including_permanent, level=level)
        self.is_caching = False

    def check_and_add_hook(
        self,
        hook_point: HookPoint,
        hook_point_name: str,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
    ) -> None:
        """Runs checks on the hook, and then adds it to the hook point"""

        self.check_hooks_to_add(
            hook_point,
            hook_point_name,
            hook,
            dir=dir,
            is_permanent=is_permanent,
            prepend=prepend,
        )
        hook_point.add_hook(hook, dir=dir, is_permanent=is_permanent, level=level, prepend=prepend)

    def check_hooks_to_add(
        self,
        hook_point: HookPoint,
        hook_point_name: str,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        prepend: bool = False,
    ) -> None:
        """Override this function to add checks on which hooks should be added"""
        pass

    def add_hook(
        self,
        name: Union[str, Callable[[str], bool]],
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
    ) -> None:
        if isinstance(name, str):
            hook_point = self.mod_dict[name]
            assert isinstance(
                hook_point, HookPoint
            )  # TODO does adding assert meaningfully slow down performance? I've added them for type checking purposes.
            self.check_and_add_hook(
                hook_point,
                name,
                hook,
                dir=dir,
                is_permanent=is_permanent,
                level=level,
                prepend=prepend,
            )
        else:
            # Otherwise, name is a Boolean function on names
            for hook_point_name, hp in self.hook_dict.items():
                if name(hook_point_name):
                    self.check_and_add_hook(
                        hp,
                        hook_point_name,
                        hook,
                        dir=dir,
                        is_permanent=is_permanent,
                        level=level,
                        prepend=prepend,
                    )

    def add_perma_hook(
        self,
        name: Union[str, Callable[[str], bool]],
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
    ) -> None:
        self.add_hook(name, hook, dir=dir, is_permanent=True)

    def _enable_hook_with_name(self, name: str, hook: Callable, dir: Literal["fwd", "bwd"]):
        """This function takes a key for the mod_dict and enables the related hook for that module

        Args:
            name (str): The module name
            hook (Callable): The hook to add
            dir (Literal["fwd", "bwd"]): The direction for the hook
        """
        self.mod_dict[name].add_hook(hook, dir=dir, level=self.context_level)  # type: ignore[operator]

    def _enable_hooks_for_points(
        self,
        hook_points: Iterable[tuple[str, HookPoint]],
        enabled: Callable,
        hook: Callable,
        dir: Literal["fwd", "bwd"],
    ):
        """Enables hooks for a list of points

        Args:
            hook_points (Dict[str, HookPoint]): The hook points
            enabled (Callable): _description_
            hook (Callable): _description_
            dir (Literal["fwd", "bwd"]): _description_
        """
        for hook_name, hook_point in hook_points:
            if enabled(hook_name):
                hook_point.add_hook(hook, dir=dir, level=self.context_level)

    def _enable_hook(self, name: Union[str, Callable], hook: Callable, dir: Literal["fwd", "bwd"]):
        """Enables an individual hook on a hook point

        Args:
            name (str): The name of the hook
            hook (Callable): The actual hook
            dir (Literal["fwd", "bwd"], optional): The direction of the hook. Defaults to "fwd".
        """
        if isinstance(name, str):
            self._enable_hook_with_name(name=name, hook=hook, dir=dir)
        else:
            self._enable_hooks_for_points(
                hook_points=self.hook_dict.items(), enabled=name, hook=hook, dir=dir
            )

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        """
        Context manager for temporarily adding hooks to the model.

        Args:
            fwd_hooks: List[Tuple[name, hook]], where name can be a hook point name string or a boolean function (for filtering hook names), hook is the function to add to that hook point.
            bwd_hooks: Same as fwd_hooks, but for backward (gradient) hooks.
            reset_hooks_end (bool): If True, removes all hooks added by this context manager when exiting.
            clear_contexts (bool): If True, clears all hook contexts when resetting hooks.

        Example usage:

        .. code-block:: python

            with model.hooks(fwd_hooks=my_hooks):
                hooked_loss = model(text, return_type="loss")

        Notes:
            - This context manager increments context_level on entry and adds forward/backward hooks to the specified hook points.
            - On exit (finally), if reset_hooks_end is True, removes all hooks added at this context_level and optionally clears context.
            - context_level is used to distinguish nested hook contexts, ensuring correct scope and lifecycle.
        """
        try:
            self.context_level += 1

            # Add forward hooks
            for name, hook in fwd_hooks:
                self._enable_hook(name=name, hook=hook, dir="fwd")
            # Add backward hooks
            for name, hook in bwd_hooks:
                self._enable_hook(name=name, hook=hook, dir="bwd")
            yield self
        finally:
            if reset_hooks_end:
                self.reset_hooks(
                    clear_contexts, including_permanent=False, level=self.context_level
                )
            self.context_level -= 1

    def run_with_hooks(
        self,
        *model_args: Any,
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs: Any,
    ):
        """
        Runs the model with specified forward and backward hooks.

        Args:
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                either the name of a hook point or a boolean function on hook names, and hook is the
                function to add to that hook point. Hooks with names that evaluate to True are added
                respectively.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                backward pass.
            reset_hooks_end (bool): If True, all hooks are removed at the end, including those added
                during this run. Default is True.
            clear_contexts (bool): If True, clears hook contexts whenever hooks are reset. Default is
                False.
            *model_args: Positional arguments for the model.
            **model_kwargs: Keyword arguments for the model's forward function. See your related
                models forward pass for details as to what sort of arguments you can pass through.

        Note:
            If you want to use backward hooks, set `reset_hooks_end` to False, so the backward hooks
            remain active. This function only runs a forward pass.
        """
        if len(bwd_hooks) > 0 and reset_hooks_end:
            logging.warning(
                "WARNING: Hooks will be reset at the end of run_with_hooks. This removes the backward hooks before a backward pass can occur."
            )

        with self.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_model:
            return hooked_model.forward(*model_args, **model_kwargs)

    def add_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> dict:
        """Adds hooks to the model to cache activations. Note: It does NOT actually run the model to get activations, that must be done separately.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Defaults to same device as model.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list

        assert callable(names_filter), "names_filter must be a callable"

        self.is_caching = True

        def save_hook(tensor: Tensor, hook: HookPoint, is_backward: bool):
            assert hook.name is not None
            hook_name = hook.name
            if is_backward:
                hook_name += "_grad"
            if remove_batch_dim:
                cache[hook_name] = tensor.detach().to(device)[0]
            else:
                cache[hook_name] = tensor.detach().to(device)

        for name, hp in self.hook_dict.items():
            if names_filter(name):
                hp.add_hook(partial(save_hook, is_backward=False), "fwd")
                if incl_bwd:
                    hp.add_hook(partial(save_hook, is_backward=True), "bwd")
        return cache

    def run_with_cache(
        self,
        *model_args: Any,
        names_filter: NamesFilter = None,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
        incl_bwd: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
        **model_kwargs: Any,
    ):
        """
        Run the model and return the model output and the cached activations (Cache object).

        Args:
            *model_args: All positional arguments to pass to the model.
            names_filter (NamesFilter, optional): Filter for which activations to cache. Can be None, a string, a list of strings, or a function taking a string and returning a bool. Default is None (cache all activations).
            device (str or torch.Device, optional): Device to store cached activations. Defaults to model device. Warning: setting this to a different device than the model will cause significant performance drop.
            remove_batch_dim (bool, optional): If True, removes the batch dimension when caching (only works for batch_size==1). Default is False.
            incl_bwd (bool, optional): If True, calls backward on the model output and also caches gradients. Assumes model output is a scalar (e.g. return_type="loss"). Custom loss functions are not supported. Default is False.
            reset_hooks_end (bool, optional): If True, removes all hooks added by this function after running. Default is True.
            clear_contexts (bool, optional): If True, clears all hook contexts when resetting hooks. Default is False.
            pos_slice: Slice to apply to cached outputs. Default is None (no slicing).
            **model_kwargs: All keyword arguments to pass to the model's forward function. See the model's forward implementation for details.

        Returns:
            tuple: (model output, cached activations dictionary Cache)

        Usage:
            This method automatically adds hooks to cache activations, runs the model, and cleans up hooks as needed.
            If incl_bwd=True, backward is called and gradients are cached as well.
        """

        # Unwrap pos_slice to ensure it is a Slice or None
        pos_slice = Slice.unwrap(pos_slice)

        # Get hooks and cache dict for caching activations
        cache_dict, fwd, bwd = self.get_caching_hooks(
            names_filter,
            incl_bwd,
            device,
            remove_batch_dim=remove_batch_dim,
            pos_slice=pos_slice,
        )

        # Use context manager to automatically register and remove hooks
        with self.hooks(
            fwd_hooks=fwd,
            bwd_hooks=bwd,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            # Run model forward, get output
            model_out = self(*model_args, **model_kwargs)
            # If backward gradients are needed, call backward
            if incl_bwd:
                model_out.backward()

        # Return model output and cached activations
        return model_out, cache_dict

    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
    ) -> tuple[dict, list, list]:
        """
        Create hooks for caching activations (does not register to model, only returns hook lists and cache dict).

        Args:
            names_filter (NamesFilter, optional): Filter for which activations to cache. Can be None, string, list of strings, or function taking a string and returning bool.
                - None: cache all activations
                - str: only cache the hook with this name
                - list[str]: only cache hooks with names in this list
                - Callable[[str], bool]: custom function, cache hooks where this returns True
            incl_bwd (bool, optional): Whether to also cache backward (gradient) activations. Default is False.
            device (DeviceType, optional): Device to store cached activations. If None, uses the layer's device. Otherwise, moves cached tensors to the specified device.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension when caching (only for batch_size=1). Default is False.
            cache (Optional[dict], optional): Dictionary to store cached activations. If None, a new empty dict is created.
            pos_slice (Optional[Union[Slice, SliceInput]], optional): Slice to apply to cached outputs. Default is None (no slicing).

        Returns:
            tuple:
                - cache (dict): Dictionary storing cached activations
                - fwd_hooks (list): List of forward hooks (each is (hook_name, hook_fn))
                - bwd_hooks (list): List of backward hooks (each is (hook_name, hook_fn)), empty if incl_bwd=False
        """
        if cache is None:
            cache = {}

        # Unwrap pos_slice to ensure it is a Slice or None
        pos_slice = Slice.unwrap(pos_slice)

        # Process names_filter to a function
        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif isinstance(names_filter, list):
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        elif callable(names_filter):
            names_filter = names_filter
        else:
            raise ValueError("names_filter must be a string, list of strings, or function")
        assert callable(names_filter)

        self.is_caching = True

        def save_hook(tensor: Tensor, hook: HookPoint, is_backward: bool = False):
            """
            Actual hook function for caching activations.

            Args:
                tensor (Tensor): Activation tensor at this hook point
                hook (HookPoint): The current hook point object
                is_backward (bool): Whether this is a backward (gradient) hook
            """
            if hook.name is None:
                raise RuntimeError("Hook should have a valid name attribute")

            hook_name = hook.name
            if is_backward:
                hook_name += "_grad"

            resid_stream = tensor.detach().to(device)
            if remove_batch_dim:
                resid_stream = resid_stream[0]

            # Determine which dimension is the position dimension for slicing
            if (
                hook.name.endswith("hook_q")
                or hook.name.endswith("hook_k")
                or hook.name.endswith("hook_v")
                or hook.name.endswith("hook_z")
                or hook.name.endswith("hook_result")
            ):
                pos_dim = -3  # For attention head-related hooks, pos is the third-to-last dim
            else:
                pos_dim = -2  # For other modules, pos is the second-to-last dim

            # Only slice if tensor has enough dimensions
            if tensor.dim() >= -pos_dim:
                resid_stream = pos_slice.apply(resid_stream, dim=pos_dim)
            cache[hook_name] = resid_stream

        fwd_hooks = []
        bwd_hooks = []
        for name, _ in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, partial(save_hook, is_backward=False)))
                if incl_bwd:
                    bwd_hooks.append((name, partial(save_hook, is_backward=True)))

        return cache, fwd_hooks, bwd_hooks

    def cache_all(
        self,
        cache: Optional[dict],
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
    ):
        logging.warning(
            "cache_all is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=lambda name: True,
            cache=cache,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )

    def cache_some(
        self,
        cache: Optional[dict],
        names: Callable[[str], bool],
        incl_bwd: bool = False,
        device: DeviceType = None,
        remove_batch_dim: bool = False,
    ):
        """Cache a list of hook provided by names, Boolean function on names"""
        logging.warning(
            "cache_some is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=names,
            cache=cache,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )


# %%
