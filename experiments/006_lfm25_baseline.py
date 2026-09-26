"""Reproducible CPU/CUDA reference for one-token-at-a-time LFM2.5-230M decode.

This is a correctness and latency baseline for the planned all-NPU implementation.
It deliberately does not claim that PyTorch CPU/GPU use the same kernels as the NPU.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "cache" / "lfm25-230m"
PROMPT = "Reply with one short sentence about what an NPU does."


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate(model, input_ids, steps, capture=False, stop_at_eos=False):
    device = input_ids.device
    times = []
    generated = []
    snapshots = {}
    past = None
    hook = None
    if capture:
        def capture_attention_context(_module, inputs):
            if step in (0, 1, 2):
                snapshots[f"step{step}_attn2_context"] = (
                    inputs[0][0, -1].float().cpu().numpy().copy()
                )

        hook = model.model.layers[2].self_attn.out_proj.register_forward_pre_hook(
            capture_attention_context
        )
    with torch.inference_mode():
        for step in range(steps):
            synchronize(device)
            start = time.perf_counter()
            output = model(
                input_ids,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=capture,
            )
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
            if capture and step in (0, 1, 2):
                snapshots[f"step{step}_logits"] = (
                    output.logits[0, -1].float().cpu().numpy()
                )
                snapshots[f"step{step}_conv0_state"] = (
                    output.past_key_values.layers[0].conv_states[0]
                    .float()
                    .cpu()
                    .numpy()
                    .copy()
                )
                snapshots[f"step{step}_conv1_state"] = (
                    output.past_key_values.layers[1].conv_states[0]
                    .float()
                    .cpu()
                    .numpy()
                    .copy()
                )
                snapshots[f"step{step}_attn2_keys"] = (
                    output.past_key_values.layers[2].keys.float().cpu().numpy().copy()
                )
                snapshots[f"step{step}_attn2_values"] = (
                    output.past_key_values.layers[2].values.float().cpu().numpy().copy()
                )
                for layer, hidden in enumerate(output.hidden_states):
                    snapshots[f"step{step}_hidden{layer}"] = (
                        hidden[0, -1].float().cpu().numpy()
                    )
            input_ids = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(input_ids.item()))
            past = output.past_key_values
            if stop_at_eos and generated[-1] == model.config.eos_token_id:
                break
    if hook is not None:
        hook.remove()
    return times, generated, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2 to measure decode")

    model_dir = args.model.resolve(strict=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).eval().to(device)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    # Warm the model and allocator without contaminating the timed KV cache.
    generate(model, input_ids, 3)
    times, generated, _ = generate(model, input_ids, args.tokens, stop_at_eos=True)
    result = {
        "model": "LiquidAI/LFM2.5-230M",
        "model_sha256": hashlib.sha256((model_dir / "model.safetensors").read_bytes()).hexdigest(),
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "AMD Ryzen 9 8945HS"
        ),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": "bfloat16",
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "prompt": PROMPT,
        "prompt_tokens": input_ids.shape[-1],
        "generated_tokens": len(generated),
        "prefill_ms": times[0],
        "decode_ms_median": statistics.median(times[1:]),
        "decode_tokens_per_second": 1000 * (len(times) - 1) / sum(times[1:]),
        "step_ms": times,
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }
    if args.reference:
        projection_inputs = []
        projection_outputs = []
        conv_outputs_to_projection = []
        conv_projected_outputs = []
        ffn_norm_inputs = []
        ffn_norm_outputs = []
        ffn_projection_outputs = {name: [] for name in ("w1", "w2", "w3")}
        ffn_w2_inputs = []

        def capture_projection(_module, inputs, output):
            projection_inputs.append(inputs[0][0, -1].float().cpu().numpy())
            projection_outputs.append(output[0, -1].float().cpu().numpy())

        def capture_conv_output(_module, inputs, output):
            conv_outputs_to_projection.append(inputs[0][0, -1].float().cpu().numpy())
            conv_projected_outputs.append(output[0, -1].float().cpu().numpy())

        def capture_ffn_norm(_module, inputs, output):
            ffn_norm_inputs.append(inputs[0][0, -1].float().cpu().numpy())
            ffn_norm_outputs.append(output[0, -1].float().cpu().numpy())

        def capture_ffn_projection(name):
            def capture(_module, inputs, output):
                ffn_projection_outputs[name].append(output[0, -1].float().cpu().numpy())
                if name == "w2":
                    ffn_w2_inputs.append(inputs[0][0, -1].float().cpu().numpy())
            return capture

        hook = model.model.layers[0].conv.in_proj.register_forward_hook(capture_projection)
        conv_hook = model.model.layers[0].conv.out_proj.register_forward_hook(
            capture_conv_output
        )
        ffn_hook = model.model.layers[0].ffn_norm.register_forward_hook(
            capture_ffn_norm
        )
        ffn_projection_hooks = [
            getattr(model.model.layers[0].feed_forward, name).register_forward_hook(
                capture_ffn_projection(name)
            )
            for name in ("w1", "w2", "w3")
        ]
        try:
            _, reference_tokens, snapshots = generate(model, input_ids, 3, capture=True)
        finally:
            hook.remove()
            conv_hook.remove()
            ffn_hook.remove()
            for projection_hook in ffn_projection_hooks:
                projection_hook.remove()
        for step in range(3):
            snapshots[f"step{step}_first_projection_input"] = projection_inputs[step]
            snapshots[f"step{step}_first_projection_output"] = projection_outputs[step]
            snapshots[f"step{step}_conv0_output_projection_input"] = (
                conv_outputs_to_projection[step]
            )
            snapshots[f"step{step}_conv0_output_projection_output"] = (
                conv_projected_outputs[step]
            )
            snapshots[f"step{step}_conv0_residual"] = ffn_norm_inputs[step]
            snapshots[f"step{step}_ffn0_norm_output"] = ffn_norm_outputs[step]
            snapshots[f"step{step}_ffn0_w2_input"] = ffn_w2_inputs[step]
            for name in ("w1", "w2", "w3"):
                snapshots[f"step{step}_ffn0_{name}_output"] = (
                    ffn_projection_outputs[name][step]
                )
        snapshots["first_projection_weight"] = (
            model.model.layers[0].conv.in_proj.weight.detach().float().cpu().numpy()
        )
        snapshots["first_operator_norm_weight"] = (
            model.model.layers[0].operator_norm.weight.detach().float().cpu().numpy()
        )
        snapshots["conv0_depthwise_weight"] = (
            model.model.layers[0].conv.conv.weight[:, 0, :].detach().float().cpu().numpy()
        )
        snapshots["conv0_output_projection_weight"] = (
            model.model.layers[0].conv.out_proj.weight.detach().float().cpu().numpy()
        )
        snapshots["first_ffn_norm_weight"] = (
            model.model.layers[0].ffn_norm.weight.detach().float().cpu().numpy()
        )
        for name in ("w1", "w2", "w3"):
            snapshots[f"first_ffn_{name}_weight"] = (
                getattr(model.model.layers[0].feed_forward, name)
                .weight.detach().float().cpu().numpy()
            )
        snapshots["prompt_ids"] = input_ids.cpu().numpy()
        snapshots["reference_token_ids"] = np.asarray(reference_tokens)
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.reference, **snapshots)
        result["reference"] = str(args.reference)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
