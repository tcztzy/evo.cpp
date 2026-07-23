#!/usr/bin/env python3
"""Generate per-layer BF16 reference vectors with the official Vortex model."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import traceback
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

OFFICIAL_FORCE_PROMPT_THRESHOLD = 3000


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
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument(
        "--midpoint-prompt",
        action="store_true",
        help="diagnostic midpoint split without changing generation mode",
    )
    parser.add_argument(
        "--official-quality",
        action="store_true",
        help=(
            "use the official midpoint split and cached long-prompt generation "
            "with a 3000-token parallel-prefill threshold"
        ),
    )
    parser.add_argument("--debug-layer", type=int)
    parser.add_argument("--logits-only", action="store_true")
    parser.add_argument("--generate-tokens", type=int, default=0)
    parser.add_argument(
        "--cached-generate",
        action="store_true",
        help="use Vortex's cached generation with official prompt forcing",
    )
    parser.add_argument("--force-prompt-threshold", type=int, default=3000)
    parser.add_argument(
        "--software-fp8",
        action="store_true",
        help=(
            "emulate TE 2.3 inference-time E4M3 Hyena projections with "
            "checkpoint-fixed scales"
        ),
    )
    parser.add_argument(
        "--software-fp8-accumulator",
        choices=("fp32", "e8m13-rne", "e8m13-rtz", "h100-qgmma"),
        default="fp32",
        help="accumulator model for --software-fp8",
    )
    parser.add_argument(
        "--software-fp8-promotion-interval",
        type=int,
        default=0,
        help=(
            "K elements between e8m13-to-FP32 promotions; 0 keeps the "
            "e8m13 accumulator for the full inner dimension"
        ),
    )
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args()


def apply_official_quality_mode(args: argparse.Namespace) -> None:
    """Apply the generation semantics used by Evo 2's published quality test."""
    if not args.official_quality:
        return
    if args.prompts_csv is None:
        raise ValueError("--official-quality requires --prompts-csv")
    if args.generate_tokens <= 0:
        raise ValueError("--official-quality requires --generate-tokens")
    args.midpoint_prompt = True
    args.cached_generate = True
    args.force_prompt_threshold = OFFICIAL_FORCE_PROMPT_THRESHOLD


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
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt_index < 0:
        raise ValueError("--prompt-index must be nonnegative")
    with args.prompts_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or args.prompt_index >= len(rows) - 1:
        raise ValueError("--prompt-index is outside the prompt CSV")
    return rows[args.prompt_index + 1][0]


def official_midpoint_split(sequence: str) -> tuple[str, str]:
    midpoint = 2 * (len(sequence) // 4)
    return sequence[:midpoint], sequence[midpoint:]


def vortex_commit(vortex_root: Path) -> str:
    git = shutil.which("git")
    if git is not None:
        return subprocess.run(
            [git, "-C", str(vortex_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    head = (vortex_root / ".git" / "HEAD").read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose_ref = vortex_root / ".git" / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="ascii").strip()
    for line in (vortex_root / ".git" / "packed-refs").read_text(
        encoding="ascii"
    ).splitlines():
        if line and not line.startswith(("#", "^")):
            commit, name = line.split(" ", maxsplit=1)
            if name == ref:
                return commit
    raise RuntimeError(f"cannot resolve Vortex git ref {ref}")


def software_e4m3_quantize(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Return TE's scaled E4M3 payload expanded losslessly to FP32."""
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
        raise ValueError(f"software FP8 scale must be finite and positive, got {scale}")
    scale_tensor = torch.tensor(scale, dtype=torch.float32, device=value.device)
    scaled = value.float().mul_(scale_tensor).clamp_(min=-448.0, max=448.0)
    quantized = scaled.to(torch.float8_e4m3fn)
    return quantized.float()


def software_fp8_output_scale(scale_inv: torch.Tensor) -> float:
    """Multiply TE's two stored FP32 inverse scales with FP32 semantics."""
    if (
        scale_inv.dtype != torch.float32
        or tuple(scale_inv.shape) != (3,)
        or not bool(torch.isfinite(scale_inv).all())
        or not bool((scale_inv > 0.0).all())
    ):
        raise ValueError("software FP8 inverse scales must be finite positive F32[3]")
    return float((scale_inv[0] * scale_inv[1]).item())


def install_software_fp8_projections(
    model: torch.nn.Module,
    checkpoint: Path,
    accumulator: str,
    promotion_interval: int,
) -> list[dict[str, float | int]]:
    """Replace 42 Hyena projections with fixed-scale E4M3 software emulation."""
    import io

    torch.serialization.add_safe_globals([io.BytesIO])
    state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise TypeError("checkpoint root must be a dictionary")

    installed: list[dict[str, float | int]] = []
    for layer, block in enumerate(model.blocks):
        if not hasattr(block, "projections"):
            continue
        key = f"blocks.{layer}.projections._extra_state"
        extra_state = state.get(key)
        if not isinstance(extra_state, io.BytesIO):
            raise TypeError(f"{key} is missing or is not an io.BytesIO")
        extra_state.seek(0)
        fp8_state = torch.load(extra_state, map_location="cpu", weights_only=False)
        if not isinstance(fp8_state, dict):
            raise TypeError(f"{key} payload must be a dictionary")
        scale = fp8_state.get("scale_fwd")
        scale_inv = fp8_state.get("scale_inv_fwd")
        history = fp8_state.get("amax_history_fwd")
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.float32
            or tuple(scale.shape) != (3,)
            or not isinstance(scale_inv, torch.Tensor)
            or scale_inv.dtype != torch.float32
            or tuple(scale_inv.shape) != (3,)
            or not isinstance(history, torch.Tensor)
            or history.dtype != torch.float32
            or tuple(history.shape) != (16, 3)
        ):
            raise ValueError(f"{key} has invalid forward scale/history tensors")

        projection = block.projections
        weight = projection.weight.detach()
        weight_amax = float(weight.abs().max().float().item())
        history_weight_amax = float(history[:, 1].max().item())
        input_scale = float(scale[0].item())
        weight_scale = float(scale[1].item())
        quantized_weight = software_e4m3_quantize(weight, weight_scale).to(
            torch.bfloat16
        )
        projection._evo2c_software_fp8_weight = quantized_weight
        projection._evo2c_software_fp8_input_scale = input_scale
        projection._evo2c_software_fp8_output_scale = (
            software_fp8_output_scale(scale_inv)
        )

        def software_forward(
            module: torch.nn.Module,
            value: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, None]:
            rounded = software_e4m3_quantize(
                value,
                module._evo2c_software_fp8_input_scale,
            ).to(torch.bfloat16)
            if accumulator == "fp32":
                import torch.nn.functional as functional

                output = functional.linear(
                    rounded.float(),
                    module._evo2c_software_fp8_weight.float(),
                ).mul_(module._evo2c_software_fp8_output_scale)
            elif accumulator.startswith("e8m13-"):
                from evo2c.triton_fp8 import e8m13_linear

                output = e8m13_linear(
                    rounded.contiguous(),
                    module._evo2c_software_fp8_weight,
                    module._evo2c_software_fp8_output_scale,
                    rounding=accumulator.removeprefix("e8m13-"),
                    promotion_interval=(
                        None if promotion_interval == 0 else promotion_interval
                    ),
                )
            else:
                from evo2c.triton_fp8 import h100_qgmma_linear

                output = h100_qgmma_linear(
                    rounded.contiguous(),
                    module._evo2c_software_fp8_weight,
                    module._evo2c_software_fp8_output_scale,
                )
            if module.bias is not None:
                output.add_(module.bias.float())
            output = output.to(value.dtype)
            if module.te_return_bias:
                return output
            return output, None

        projection.forward = types.MethodType(software_forward, projection)
        block.config["use_fp8_input_projections"] = True
        installed.append(
            {
                "layer": layer,
                "input_scale": input_scale,
                "weight_scale": weight_scale,
                "output_scale": projection._evo2c_software_fp8_output_scale,
                "weight_amax": weight_amax,
                "history_weight_amax": history_weight_amax,
            }
        )

    if len(installed) != 42:
        raise ValueError(f"installed {len(installed)} software FP8 projections; expected 42")
    return installed


def main() -> int:
    args = parse_args()
    try:
        apply_official_quality_mode(args)
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
        sequence = read_prompt(args)
        if args.midpoint_prompt:
            prompt, target = official_midpoint_split(sequence)
        else:
            prompt, target = sequence, ""
        tokens = list(prompt.encode("utf-8"))
        if not tokens:
            raise ValueError("prompt must encode to at least one byte")
        if args.generate_tokens < 0:
            raise ValueError("--generate-tokens must be nonnegative")
        if args.generate_tokens > 0 and not args.logits_only:
            raise ValueError("--generate-tokens requires --logits-only")
        if args.cached_generate and args.generate_tokens == 0:
            raise ValueError("--cached-generate requires --generate-tokens")
        if args.force_prompt_threshold <= 0:
            raise ValueError("--force-prompt-threshold must be positive")
        if (
            args.software_fp8_promotion_interval < 0
            or args.software_fp8_promotion_interval % 32 != 0
        ):
            raise ValueError(
                "--software-fp8-promotion-interval must be zero or a positive multiple of 32"
            )
        if (
            not args.software_fp8_accumulator.startswith("e8m13-")
            and args.software_fp8_promotion_interval != 0
        ):
            raise ValueError(
                "--software-fp8-promotion-interval requires an e8m13 accumulator"
            )

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
        software_fp8 = (
            install_software_fp8_projections(
                model,
                args.checkpoint,
                args.software_fp8_accumulator,
                args.software_fp8_promotion_interval,
            )
            if args.software_fp8
            else []
        )
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
            if args.logits_only:
                break

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
        generated: list[int] = []
        step_logits: list[torch.Tensor] = []
        try:
            with torch.inference_mode():
                if args.generate_tokens > 0:
                    if args.cached_generate:
                        from vortex.model.generation import generate
                        from vortex.model.tokenizer import CharLevelTokenizer

                        result = generate(
                            prompt_seqs=[prompt],
                            n_tokens=args.generate_tokens,
                            model=model,
                            tokenizer=CharLevelTokenizer(int(config.vocab_size)),
                            top_k=1,
                            top_p=1.0,
                            temperature=1.0,
                            cached_generation=True,
                            force_prompt_threshold=args.force_prompt_threshold,
                            verbose=0,
                            device="cuda:0",
                        )
                        generated = list(result.sequences[0].encode("utf-8"))
                        if len(generated) != args.generate_tokens:
                            raise ValueError(
                                "cached generation returned an unexpected byte count"
                            )
                        logits = result.logits[0][0].detach().float().cpu()
                    else:
                        for _ in range(args.generate_tokens):
                            model_output = model(input_ids)
                            if (
                                not isinstance(model_output, tuple)
                                or not isinstance(model_output[0], torch.Tensor)
                            ):
                                raise TypeError(
                                    "Vortex model returned an unexpected value"
                                )
                            last_logits = (
                                model_output[0][0, -1].detach().float().cpu()
                            )
                            step_logits.append(last_logits)
                            token = int(torch.argmax(last_logits).item())
                            if token > 255:
                                raise ValueError(
                                    f"generated token {token} has no byte representation"
                                )
                            generated.append(token)
                            next_token = torch.tensor(
                                [[token]],
                                dtype=input_ids.dtype,
                                device=input_ids.device,
                            )
                            input_ids = torch.cat((input_ids, next_token), dim=1)
                        logits = torch.stack(step_logits)
                    (args.output_dir / "generated.bin").write_bytes(
                        bytes(generated)
                    )
                else:
                    model_output = model(input_ids)
                    if (
                        not isinstance(model_output, tuple)
                        or not isinstance(model_output[0], torch.Tensor)
                    ):
                        raise TypeError(
                            "Vortex model returned an unexpected value"
                        )
                    logits = model_output[0][0].detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        np.save(args.output_dir / "logits.npy", logits.numpy(), allow_pickle=False)

        top_values, top_tokens = torch.topk(logits[-1], k=10)
        commit = vortex_commit(args.vortex_root)
        manifest: dict[str, object] = {
            "schema": 1,
            "producer": (
                "official-vortex-software-e4m3"
                if args.software_fp8
                else "official-vortex-bf16"
            ),
            "vortex_commit": commit,
            "vortex_version": getattr(vortex, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "prompt": prompt,
            "target": target,
            "midpoint_prompt": args.midpoint_prompt,
            "tokens": tokens,
            "generated_tokens": generated,
            "cached_generate": args.cached_generate,
            "official_quality": args.official_quality,
            "force_prompt_threshold": args.force_prompt_threshold,
            "debug_layer": args.debug_layer,
            "layers": int(config.num_layers),
            "hidden_size": int(config.hidden_size),
            "vocab_size": int(config.vocab_size),
            "overrides": overrides,
            "software_fp8": software_fp8,
            "software_fp8_accumulator": args.software_fp8_accumulator,
            "software_fp8_promotion_interval": (
                args.software_fp8_promotion_interval
            ),
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
