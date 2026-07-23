#!/usr/bin/env python3
"""Generate per-layer BF16 reference vectors with the official Vortex model."""

import argparse
import csv
import json
import re
import subprocess
import sys
import traceback
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official Vortex without FlashAttention/FP8 and dump every block."
    )
    parser.add_argument("--vortex-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompts-csv", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--debug-layer", type=int)
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args()


def install_optional_vortex_ops_namespace(vortex_root: Path) -> None:
    """Let Vortex import its Triton RoPE while its optional FlashAttention is absent."""
    import vortex

    namespace = types.ModuleType("vortex.ops")
    namespace.__file__ = str(vortex_root / "vortex" / "ops" / "__init__.py")
    namespace.__package__ = "vortex.ops"
    namespace.__path__ = [str(vortex_root / "vortex" / "ops")]
    sys.modules["vortex.ops"] = namespace
    vortex.ops = namespace


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    if args.prompt_index < 0:
        raise ValueError("--prompt-index must be nonnegative")
    with args.prompts_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or args.prompt_index >= len(rows) - 1:
        raise ValueError("--prompt-index is outside the prompt CSV")
    return rows[args.prompt_index + 1][0]


def main() -> int:
    args = parse_args()
    try:
        if not args.vortex_root.is_dir():
            raise ValueError(f"Vortex root not found: {args.vortex_root}")
        if not args.config.is_file():
            raise ValueError(f"config not found: {args.config}")
        if not args.checkpoint.is_file():
            raise ValueError(f"checkpoint not found: {args.checkpoint}")
        if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
            raise ValueError("--checkpoint-sha256 must be 64 lowercase hexadecimal characters")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prompt = read_prompt(args)
        tokens = list(prompt.encode("utf-8"))
        if not tokens:
            raise ValueError("prompt must encode to at least one byte")

        sys.path.insert(0, str(args.vortex_root))
        import vortex

        install_optional_vortex_ops_namespace(args.vortex_root)
        from vortex.model.model import StripedHyena
        from vortex.model.utils import dotdict, load_checkpoint

        raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise ValueError("Vortex config root must be a mapping")
        overrides: dict[str, object] = {
            "use_fp8_input_projections": False,
            "use_flash_attn": False,
            "use_flashfft": False,
            "use_flash_depthwise": False,
            "use_flash_rmsnorm": False,
            "use_hcs_kernel": False,
            "use_hcm_kernel": False,
            "use_hcl_kernel": False,
        }
        raw_config.update(overrides)
        config = dotdict(raw_config)

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        if torch.cuda.device_count() != 4:
            raise RuntimeError(
                f"reference generation requires exactly four visible GPUs, found "
                f"{torch.cuda.device_count()}"
            )
        model = StripedHyena(config)
        load_checkpoint(model, checkpoint_path=str(args.checkpoint))
        model.eval()
        if args.debug_layer is not None and not 0 <= args.debug_layer < len(model.blocks):
            raise ValueError("--debug-layer is outside the model")

        handles: list[torch.utils.hooks.RemovableHandle] = []

        def save_tensor(filename: str, output: object) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise TypeError(f"debug hook {filename} received an unexpected value")
            np.save(
                args.output_dir / filename,
                value[0].detach().float().cpu().numpy(),
                allow_pickle=False,
            )

        for layer, block in enumerate(model.blocks):

            def capture(
                _module: torch.nn.Module,
                _inputs: tuple[object, ...],
                output: object,
                *,
                layer_index: int = layer,
            ) -> None:
                if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
                    raise TypeError(f"Vortex block {layer_index} returned an unexpected value")
                values = output[0][0].detach().float().cpu().numpy()
                np.save(
                    args.output_dir / f"layer_{layer_index:02d}.npy",
                    values,
                    allow_pickle=False,
                )

            handles.append(block.register_forward_hook(capture))

        if args.debug_layer is not None:
            debug_block = model.blocks[args.debug_layer]
            if not hasattr(debug_block, "filter"):
                raise ValueError("--debug-layer currently requires a Hyena layer")
            debug_prefix = f"layer_{args.debug_layer}"

            def capture_named(
                filename: str,
            ) -> Callable[[torch.nn.Module, tuple[object, ...], object], None]:
                def capture(
                    _module: torch.nn.Module,
                    _inputs: tuple[object, ...],
                    output: object,
                ) -> None:
                    save_tensor(filename, output)

                return capture

            def capture_input(
                filename: str,
            ) -> Callable[[torch.nn.Module, tuple[object, ...]], None]:
                def capture(
                    _module: torch.nn.Module,
                    inputs: tuple[object, ...],
                ) -> None:
                    if not inputs:
                        raise TypeError(f"debug pre-hook {filename} received no input")
                    save_tensor(filename, inputs[0])

                return capture

            handles.extend(
                [
                    debug_block.pre_norm.register_forward_hook(
                        capture_named(f"{debug_prefix}_pre_norm.npy")
                    ),
                    debug_block.filter.register_forward_hook(
                        capture_named(f"{debug_prefix}_mixer_output.npy")
                    ),
                    debug_block.post_norm.register_forward_pre_hook(
                        capture_input(f"{debug_prefix}_mixer_residual.npy")
                    ),
                    debug_block.post_norm.register_forward_hook(
                        capture_named(f"{debug_prefix}_post_norm.npy")
                    ),
                    debug_block.mlp.register_forward_hook(
                        capture_named(f"{debug_prefix}_mlp_output.npy")
                    ),
                ]
            )

        input_ids = torch.tensor(tokens, dtype=torch.int, device="cuda:0").unsqueeze(0)
        try:
            with torch.inference_mode():
                model_output = model(input_ids)
        finally:
            for handle in handles:
                handle.remove()
        if not isinstance(model_output, tuple) or not isinstance(model_output[0], torch.Tensor):
            raise TypeError("Vortex model returned an unexpected value")
        logits = model_output[0][0].detach().float().cpu()
        np.save(args.output_dir / "logits.npy", logits.numpy(), allow_pickle=False)

        top_values, top_tokens = torch.topk(logits[-1], k=10)
        commit = subprocess.run(
            ["git", "-C", str(args.vortex_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest: dict[str, object] = {
            "schema": 1,
            "producer": "official-vortex-bf16",
            "vortex_commit": commit,
            "vortex_version": getattr(vortex, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "prompt": prompt,
            "tokens": tokens,
            "debug_layer": args.debug_layer,
            "layers": int(config.num_layers),
            "hidden_size": int(config.hidden_size),
            "vocab_size": int(config.vocab_size),
            "overrides": overrides,
            "last_top10_tokens": [int(token) for token in top_tokens.tolist()],
            "last_top10_logits": [float(value) for value in top_values.tolist()],
            "peak_allocated_bytes": [
                int(torch.cuda.max_memory_allocated(torch.device(f"cuda:{device}")))
                for device in range(4)
            ],
        }
        manifest_partial = args.output_dir / ".manifest.json.partial"
        manifest_partial.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_partial.replace(args.output_dir / "manifest.json")
        print(json.dumps(manifest, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"generate_vortex_vectors: error: {error}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
