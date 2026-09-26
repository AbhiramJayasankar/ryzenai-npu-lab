"""Read BF16 model tensors from a local safetensors checkpoint without PyTorch."""

import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


class BF16Checkpoint:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        with self.path.open("rb") as handle:
            header_bytes = handle.read(8)
            if len(header_bytes) != 8:
                raise ValueError("Checkpoint is missing its safetensors header")
            self.header_len = int.from_bytes(header_bytes, "little")
            self.header = json.loads(handle.read(self.header_len))
        self.data_start = 8 + self.header_len

    def load(self, name):
        spec = self.header[name]
        if spec["dtype"] != "BF16":
            raise ValueError(f"{name} has dtype {spec['dtype']}, expected BF16")
        start, end = spec["data_offsets"]
        shape = tuple(spec["shape"])
        if end - start != int(np.prod(shape)) * 2:
            raise ValueError(f"{name} has an invalid BF16 byte length")
        mapped = np.memmap(
            self.path,
            dtype=bfloat16,
            mode="r",
            offset=self.data_start + start,
            shape=shape,
        )
        return np.array(mapped, dtype=bfloat16)


def recurrent_layer_data(checkpoint, index):
    prefix = f"model.layers.{index}."
    tensors = {
        "operator_gamma": checkpoint.load(prefix + "operator_norm.weight"),
        "input": checkpoint.load(prefix + "conv.in_proj.weight"),
        "output": checkpoint.load(prefix + "conv.out_proj.weight"),
        "ffn_gamma": checkpoint.load(prefix + "ffn_norm.weight"),
        "w1": checkpoint.load(prefix + "feed_forward.w1.weight"),
        "w3": checkpoint.load(prefix + "feed_forward.w3.weight"),
        "w2": checkpoint.load(prefix + "feed_forward.w2.weight"),
        "conv_weight": checkpoint.load(prefix + "conv.conv.weight")[:, 0, :],
    }
    expected_shapes = {
        "operator_gamma": (1024,),
        "input": (3072, 1024),
        "output": (1024, 1024),
        "ffn_gamma": (1024,),
        "w1": (2560, 1024),
        "w3": (2560, 1024),
        "w2": (1024, 2560),
        "conv_weight": (1024, 3),
    }
    for name, shape in expected_shapes.items():
        if tensors[name].shape != shape:
            raise ValueError(
                f"Layer {index} {name} shape {tensors[name].shape} != {shape}"
            )
    return tensors


def pack_recurrent_weights(tensors):
    return np.concatenate(
        [
            np.pad(tensors["operator_gamma"], (0, 4096 - 1024)),
            tensors["input"].reshape(-1),
            tensors["output"].reshape(-1),
            np.pad(tensors["ffn_gamma"], (0, 4096 - 1024)),
            tensors["w1"].reshape(-1),
            tensors["w3"].reshape(-1),
            tensors["w2"].reshape(-1),
        ]
    ).astype(bfloat16)


def attention_tail_data(checkpoint, index):
    prefix = f"model.layers.{index}."
    tensors = {
        "output": checkpoint.load(prefix + "self_attn.out_proj.weight"),
        "ffn_gamma": checkpoint.load(prefix + "ffn_norm.weight"),
        "w1": checkpoint.load(prefix + "feed_forward.w1.weight"),
        "w3": checkpoint.load(prefix + "feed_forward.w3.weight"),
        "w2": checkpoint.load(prefix + "feed_forward.w2.weight"),
    }
    expected = {
        "output": (1024, 1024),
        "ffn_gamma": (1024,),
        "w1": (2560, 1024),
        "w3": (2560, 1024),
        "w2": (1024, 2560),
    }
    for name, shape in expected.items():
        if tensors[name].shape != shape:
            raise ValueError(f"Attention layer {index} {name} shape {tensors[name].shape} != {shape}")
    return tensors


def pack_attention_tail_weights(tensors):
    return np.concatenate([
        tensors["output"].reshape(-1),
        np.pad(tensors["ffn_gamma"], (0, 4096 - 1024)),
        tensors["w1"].reshape(-1),
        tensors["w3"].reshape(-1),
        tensors["w2"].reshape(-1),
    ]).astype(bfloat16)
