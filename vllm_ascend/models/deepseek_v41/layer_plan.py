# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static layer topology for DeepSeek V4.1.

This module contains no device code.  It converts the model configuration into
the source/consumer graph used by model construction, KV-cache allocation and
attention metadata building.  Keeping this mapping in one place prevents the
three subsystems from independently interpreting ``compress_ratios``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


def _read(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        try:
            return config[name]
        except KeyError as exc:
            raise ValueError(f"DeepSeek V4.1 config is missing {name!r}") from exc
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"DeepSeek V4.1 config is missing {name!r}") from exc


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )
