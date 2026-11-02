"""Clean-room local Hugging Face and affine SAE adapters.

State slice: interpretability-lab-sandbox-v1.

The adapters are deliberately dependency-optional.  Importing this module is
safe on the synthetic runner; model libraries are required only when a real
packet explicitly opens an internal run.  The GPT-2 module path is canonical
(``transformer.h[layer].mlp``) and is checked before any hook is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class RealAdapterError(RuntimeError):
    """Raised when a real adapter cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class GPT2Spec:
    model_id: str
    revision: str
    model_root: Path
    layer: int = 6
    hook: str = "mlp_out"
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.model_id != "openai-community/gpt2":
            raise RealAdapterError("only the frozen GPT-2 exploratory model is supported")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise RealAdapterError("model revision must be a 40-character hexadecimal commit")
        if self.layer != 6:
            raise RealAdapterError("canonical GPT-2 SAE path is transformer.h[6].mlp")
        if self.hook not in {"mlp_out", "resid_mid"}:
            raise RealAdapterError("unsupported GPT-2 hook; choose mlp_out or resid_mid")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise RealAdapterError("unsupported torch dtype")


def _torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by the real runner
        raise RealAdapterError("torch is required for real model execution") from exc
    return torch


def _dtype(torch: Any, name: str) -> Any:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


class GPT2Runtime:
    """Offline GPT-2 loader with native and hooked forward paths."""

    def __init__(self, spec: GPT2Spec) -> None:
        root = Path(spec.model_root)
        if not root.is_dir():
            raise RealAdapterError(f"model root is missing: {root}")
        torch = _torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("transformers is required for real model execution") from exc
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        self.torch = torch
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, revision=spec.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            revision=spec.revision,
            torch_dtype=_dtype(torch, spec.dtype),
        )
        self.model.to(spec.device)
        self.model.eval()
        blocks = getattr(getattr(self.model, "transformer", None), "h", None)
        if blocks is None or spec.layer >= len(blocks):
            raise RealAdapterError("canonical GPT-2 transformer block registry is missing")
        self.block = blocks[spec.layer]
        self.mlp = getattr(self.block, "mlp", None)
        if self.mlp is None:
            raise RealAdapterError("canonical GPT-2 MLP module is missing")
        self.ln_2 = getattr(self.block, "ln_2", None)
        if spec.hook == "resid_mid" and self.ln_2 is None:
            raise RealAdapterError("canonical GPT-2 ln_2 module is missing for resid_mid")

    def _run(self, row: str, *, capture: bool, patched_activation: Any | None = None) -> tuple[Any, Any | None]:
        if not isinstance(row, str) or not row.strip():
            raise RealAdapterError("rows must be nonempty text")
        encoded = self.tokenizer(row, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.spec.device) for key, value in encoded.items()}
        state: dict[str, Any] = {}

        def _patch(value: Any) -> Any:
            value = value
            if patched_activation is not None:
                value = value.clone()
                if tuple(patched_activation.shape) != tuple(value[:, -1, :].shape):
                    raise RealAdapterError("patched activation shape does not match final token")
                value[:, -1, :] = patched_activation.to(value.device, dtype=value.dtype)
            if capture:
                state["activation"] = value[:, -1, :].detach().float().cpu()
            return value

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            value = _patch(value)
            return (value, *output[1:]) if isinstance(output, tuple) else value

        def pre_hook(_module: Any, inputs: Any) -> Any:
            if not inputs:
                raise RealAdapterError("resid_mid hook received no inputs")
            value = _patch(inputs[0])
            return (value, *inputs[1:])

        module = self.mlp if self.spec.hook == "mlp_out" else self.ln_2
        handle = module.register_forward_hook(hook) if (capture or patched_activation is not None) and self.spec.hook == "mlp_out" else None
        pre_handle = module.register_forward_pre_hook(pre_hook) if (capture or patched_activation is not None) and self.spec.hook == "resid_mid" else None
        try:
            with self.torch.inference_mode():
                result = self.model(**encoded, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
            if pre_handle is not None:
                pre_handle.remove()
        logits = result.logits[:, -1, :].detach().float().cpu()
        activation = state.get("activation")
        return logits, activation

    def logits(self, rows: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in self._run(row, capture=False)[0][0]) for row in rows)

    def capture(self, rows: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in self._run(row, capture=True)[1][0]) for row in rows)

    def patched_logits(self, row: str, activation: Sequence[float]) -> tuple[float, ...]:
        patch = self.torch.as_tensor(tuple(float(v) for v in activation), dtype=self.torch.float32, device=self.spec.device).reshape(1, -1)
        logits, _ = self._run(row, capture=False, patched_activation=patch)
        return tuple(float(v) for v in logits[0])


@dataclass(frozen=True)
class Qwen38Spec:
    """Frozen contract for the Qwen3.8-27B language-model stack.

    State slice: ``interpretability-lab-sandbox-v1``.

    Qwen3.8 is exposed by Transformers as a multimodal wrapper.  The
    language-model decoder registry is therefore addressed explicitly as
    ``model.language_model.layers[layer]``; silently deriving a layer from an
    arbitrary module name would allow feature-asset shopping.
    """

    model_id: str
    revision: str
    model_root: Path
    layer: int = 32
    hook: str = "decoder_out"
    device: str = "cpu"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.model_id != "Qwen/Qwen3.8-27B":
            raise RealAdapterError("only the frozen Qwen3.8-27B exploratory model is supported")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise RealAdapterError("model revision must be a 40-character hexadecimal commit")
        if not 0 <= self.layer < 64:
            raise RealAdapterError("canonical Qwen3.8 decoder layer must be in 0..63")
        if self.hook not in {"decoder_in", "decoder_out"}:
            raise RealAdapterError("unsupported Qwen3.8 hook; choose decoder_in or decoder_out")
        if self.dtype not in {"bfloat16", "float32"}:
            raise RealAdapterError("Qwen3.8 supports only bfloat16 or float32 in this adapter")

    @property
    def canonical_path(self) -> str:
        return f"model.language_model.layers[{self.layer}]"


class Qwen38Runtime:
    """Offline Qwen3.8-27B text-only path with a declared layer hook."""

    def __init__(self, spec: Qwen38Spec) -> None:
        root = Path(spec.model_root)
        if not root.is_dir():
            raise RealAdapterError(f"model root is missing: {root}")
        torch = _torch()
        try:
            from transformers import AutoProcessor  # type: ignore
            try:
                from transformers import AutoModelForMultimodalLM  # type: ignore
            except ImportError:
                # Transformers 4.57.x exposes the same Qwen3.5 wrapper through
                # this task alias; keep the packet's architecture/path fixed.
                from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM  # type: ignore
        except ImportError as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("a Transformers release with Qwen3.8 support is required") from exc
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        self.torch = torch
        self.spec = spec
        try:
            self.processor = AutoProcessor.from_pretrained(root, local_files_only=True, revision=spec.revision)
            self.model = AutoModelForMultimodalLM.from_pretrained(
                root,
                local_files_only=True,
                revision=spec.revision,
                torch_dtype=_dtype(torch, spec.dtype),
            )
        except Exception as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("Qwen3.8 checkpoint could not be loaded offline") from exc
        self.model.to(spec.device)
        self.model.eval()
        wrapper = getattr(self.model, "model", None)
        language_model = getattr(wrapper, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None or len(layers) != 64 or spec.layer >= len(layers):
            raise RealAdapterError("Qwen3.8 language-model decoder registry must contain 64 layers")
        self.block = layers[spec.layer]

    def _encode(self, row: str) -> dict[str, Any]:
        if not isinstance(row, str) or not row.strip():
            raise RealAdapterError("rows must be nonempty text")
        try:
            encoded = self.processor(text=[row], return_tensors="pt", padding=False)
        except Exception as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("Qwen3.8 text-only processor encoding failed") from exc
        return {key: value.to(self.spec.device) for key, value in encoded.items() if hasattr(value, "to")}

    def _run(self, row: str, *, capture: bool, patched_activation: Any | None = None, capture_all: bool = False) -> tuple[Any, Any | None]:
        encoded = self._encode(row)
        state: dict[str, Any] = {}

        def _patch(value: Any) -> Any:
            if value.ndim != 3:
                raise RealAdapterError("Qwen3.8 decoder output must be [batch, sequence, hidden]")
            if patched_activation is not None:
                value = value.clone()
                if tuple(patched_activation.shape) != tuple(value[:, -1, :].shape):
                    raise RealAdapterError("patched activation shape does not match final token")
                value[:, -1, :] = patched_activation.to(value.device, dtype=value.dtype)
            if capture:
                state["activation_all" if capture_all else "activation"] = (
                    value.detach().float().cpu() if capture_all else value[:, -1, :].detach().float().cpu()
                )
            return value

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            value = _patch(value)
            return (value, *output[1:]) if isinstance(output, tuple) else value

        def pre_hook(_module: Any, inputs: Any) -> Any:
            if not inputs:
                raise RealAdapterError("Qwen3.8 decoder input hook received no inputs")
            value = _patch(inputs[0])
            return (value, *inputs[1:])

        handle = self.block.register_forward_hook(hook) if (capture or patched_activation is not None) and self.spec.hook == "decoder_out" else None
        pre_handle = self.block.register_forward_pre_hook(pre_hook) if (capture or patched_activation is not None) and self.spec.hook == "decoder_in" else None
        try:
            with self.torch.inference_mode():
                result = self.model(**encoded, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
            if pre_handle is not None:
                pre_handle.remove()
        logits = result.logits[:, -1, :].detach().float().cpu()
        return logits, state.get("activation_all" if capture_all else "activation")

    def logits(self, rows: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in self._run(row, capture=False)[0][0]) for row in rows)

    def capture(self, rows: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in self._run(row, capture=True)[1][0]) for row in rows)

    def capture_all(self, rows: Sequence[str]) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Capture every unpadded token at the frozen decoder hook."""
        captured: list[tuple[tuple[float, ...], ...]] = []
        for row in rows:
            value = self._run(row, capture=True, capture_all=True)[1]
            if value is None:
                raise RealAdapterError("Qwen3.8 all-token capture produced no activation")
            captured.append(tuple(tuple(float(v) for v in token) for token in value[0]))
        return tuple(captured)

    def patched_logits(self, row: str, activation: Sequence[float]) -> tuple[float, ...]:
        patch = self.torch.as_tensor(tuple(float(v) for v in activation), dtype=self.torch.float32, device=self.spec.device).reshape(1, -1)
        logits, _ = self._run(row, capture=False, patched_activation=patch)
        return tuple(float(v) for v in logits[0])


class AffineSAE:
    """Model-matched affine SAE with deterministic top-k encoding."""

    def __init__(self, weights_path: Path, config: Mapping[str, Any]) -> None:
        torch = _torch()
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("safetensors is required for real feature execution") from exc
        if not Path(weights_path).is_file():
            raise RealAdapterError(f"SAE weights are missing: {weights_path}")
        weights = load_file(str(weights_path), device="cpu")
        required = {"W_enc", "W_dec", "b_enc", "b_dec"}
        if set(weights) != required:
            raise RealAdapterError("SAE tensor roster mismatch")
        self.torch = torch
        self.w_enc = weights["W_enc"].float()
        self.w_dec = weights["W_dec"].float()
        self.b_enc = weights["b_enc"].float()
        self.b_dec = weights["b_dec"].float()
        if self.w_enc.ndim != 2 or self.w_dec.ndim != 2:
            raise RealAdapterError("SAE matrices must be rank two")
        if (self.w_enc.shape[1], self.w_enc.shape[0]) != tuple(self.w_dec.shape):
            raise RealAdapterError("SAE encoder and decoder dimensions disagree")
        if self.w_enc.shape[0] != self.w_dec.shape[1]:
            raise RealAdapterError("SAE input width is inconsistent")
        expected_d_in = config.get("d_in", int(self.w_enc.shape[0]))
        expected_d_sae = config.get("d_sae", int(self.w_enc.shape[1]))
        if int(expected_d_in) != self.w_enc.shape[0] or int(expected_d_sae) != self.w_enc.shape[1]:
            raise RealAdapterError("SAE dimensions do not match its frozen config")
        self.k = int(config.get("activation_fn_kwargs", {}).get("k", 32))
        if self.k <= 0 or self.k > self.w_enc.shape[1]:
            raise RealAdapterError("invalid SAE top-k")
        self.apply_b_dec = bool(config.get("apply_b_dec_to_input", True))
        if config.get("normalize_activations") != "layer_norm":
            raise RealAdapterError("the frozen GPT-2 SAE requires layer_norm activation normalization")
        self.eps = 1e-5

    def encode_tensor(self, activation: Any) -> Any:
        x = activation.float().reshape(-1)
        if x.numel() != self.w_enc.shape[0]:
            raise RealAdapterError(f"activation width must be {self.w_enc.shape[0]}")
        mean = x.mean()
        std = x.std()
        normalized = (x - mean) / (std + self.eps)
        centered = normalized - self.b_dec if self.apply_b_dec else normalized
        pre = centered @ self.w_enc + self.b_enc
        values, indices = self.torch.topk(pre, self.k, largest=True, sorted=True)
        features = self.torch.zeros_like(pre)
        return features.scatter(0, indices, self.torch.relu(values))

    def encode(self, activation: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in self.encode_tensor(self.torch.as_tensor(tuple(activation))).detach().cpu().tolist())

    def decode_tensor(self, features: Any, *, mean: Any | None = None, std: Any | None = None) -> Any:
        f = features.float().reshape(-1)
        if f.numel() != self.w_enc.shape[1]:
            raise RealAdapterError(f"feature width must be {self.w_enc.shape[1]}")
        decoded = f @ self.w_dec + self.b_dec
        if mean is not None and std is not None:
            decoded = decoded * std + mean
        return decoded

    def decode(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in self.decode_tensor(self.torch.as_tensor(tuple(features))).detach().cpu().tolist())

    def reconstruct(self, activation: Sequence[float]) -> tuple[float, ...]:
        original = self.torch.as_tensor(tuple(activation)).float().reshape(-1)
        mean = original.mean()
        std = original.std()
        features = self.encode_tensor(original)
        return tuple(float(v) for v in self.decode_tensor(features, mean=mean, std=std).tolist())

    def intervene(self, activation: Sequence[float], feature_index: int, value: float) -> tuple[float, ...]:
        original = self.torch.as_tensor(tuple(activation)).float().reshape(-1)
        mean = original.mean()
        std = original.std()
        features = self.encode_tensor(original)
        if not 0 <= int(feature_index) < features.numel():
            raise IndexError("feature_index out of range")
        features[int(feature_index)] = float(value)
        return tuple(float(v) for v in self.decode_tensor(features, mean=mean, std=std).tolist())


class Qwen38TopKSAE:
    """Safe loader for the Qwen3.8 Scope ``.pt`` checkpoint format.

    The checkpoint is loaded with ``weights_only=True`` so an untrusted pickle
    cannot execute code.  Its stored scale is part of the frozen asset
    contract; no runtime normalization or post-hoc rescaling is allowed.
    """

    def __init__(self, weights_path: Path, *, d_in: int = 5120, d_sae: int = 40960, top_k: int = 50, device: str = "cpu") -> None:
        torch = _torch()
        if not Path(weights_path).is_file():
            raise RealAdapterError(f"Qwen3.8 SAE weights are missing: {weights_path}")
        try:
            loaded = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except Exception as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("Qwen3.8 SAE checkpoint could not be loaded safely") from exc
        if not isinstance(loaded, Mapping):
            raise RealAdapterError("Qwen3.8 SAE checkpoint must be a mapping")
        state = loaded.get("state", loaded)
        if not isinstance(state, Mapping):
            raise RealAdapterError("Qwen3.8 SAE state must be a mapping")
        required = {"b_pre", "W_enc", "b_enc", "W_dec"}
        if set(state) != required:
            raise RealAdapterError("Qwen3.8 SAE tensor roster mismatch")
        self.torch = torch
        try:
            target_device = torch.device(device)
        except Exception as exc:
            raise RealAdapterError("Qwen3.8 SAE device is invalid") from exc
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RealAdapterError("Qwen3.8 SAE CUDA device is unavailable")
        self.device = target_device
        self.b_pre = state["b_pre"].float().to(target_device).reshape(-1)
        self.w_enc = state["W_enc"].float().to(target_device)
        self.b_enc = state["b_enc"].float().to(target_device).reshape(-1)
        self.w_dec = state["W_dec"].float().to(target_device)
        scale = loaded.get("scale")
        if scale is None:
            raise RealAdapterError("Qwen3.8 SAE normalization scale is missing")
        scale_tensor = torch.as_tensor(scale).float().reshape(-1)
        if scale_tensor.numel() != 1 or not bool(torch.isfinite(scale_tensor).all()) or float(scale_tensor.item()) == 0.0:
            raise RealAdapterError("Qwen3.8 SAE normalization scale is invalid")
        self.scale = scale_tensor[0].to(target_device)
        stored_k = int(loaded.get("k", top_k))
        stored_width = int(loaded.get("width", d_sae))
        if stored_k != top_k or stored_width != d_sae:
            raise RealAdapterError("Qwen3.8 SAE top-k or width does not match its frozen packet")
        if self.b_pre.numel() != d_in or self.b_enc.numel() != d_sae:
            raise RealAdapterError("Qwen3.8 SAE bias dimensions do not match its frozen packet")
        if tuple(self.w_enc.shape) != (d_in, d_sae) or tuple(self.w_dec.shape) != (d_sae, d_in):
            raise RealAdapterError("Qwen3.8 SAE matrix dimensions do not match its frozen packet")
        self.k = stored_k

    def encode_tensor(self, activation: Any) -> Any:
        x = activation.float().to(self.device).reshape(-1)
        if x.numel() != self.b_pre.numel():
            raise RealAdapterError(f"activation width must be {self.b_pre.numel()}")
        pre = (x * self.scale - self.b_pre) @ self.w_enc + self.b_enc
        values, indices = self.torch.topk(pre, self.k, largest=True, sorted=True)
        features = self.torch.zeros_like(pre)
        return features.scatter(0, indices, self.torch.relu(values))

    def encode(self, activation: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in self.encode_tensor(self.torch.as_tensor(tuple(activation))).tolist())

    def encode_batch_tensor(self, activations: Any) -> Any:
        """Encode a two-dimensional token batch on the frozen SAE device."""
        x = activations.float().to(self.device)
        if x.ndim != 2 or x.shape[1] != self.b_pre.numel():
            raise RealAdapterError(f"activation batch must have shape [n, {self.b_pre.numel()}]")
        pre = (x * self.scale - self.b_pre) @ self.w_enc + self.b_enc
        values, indices = self.torch.topk(pre, self.k, dim=1, largest=True, sorted=True)
        features = self.torch.zeros_like(pre)
        return features.scatter(1, indices, self.torch.relu(values))

    def decode_tensor(self, features: Any) -> Any:
        f = features.float().to(self.device).reshape(-1)
        if f.numel() != self.w_dec.shape[0]:
            raise RealAdapterError(f"feature width must be {self.w_dec.shape[0]}")
        return (f @ self.w_dec + self.b_pre) / self.scale

    def decode(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in self.decode_tensor(self.torch.as_tensor(tuple(features))).tolist())

    def decode_batch_tensor(self, features: Any) -> Any:
        """Decode a two-dimensional feature batch on the frozen SAE device."""
        f = features.float().to(self.device)
        if f.ndim != 2 or f.shape[1] != self.w_dec.shape[0]:
            raise RealAdapterError(f"feature batch must have shape [n, {self.w_dec.shape[0]}]")
        return (f @ self.w_dec + self.b_pre) / self.scale

    def reconstruct(self, activation: Sequence[float]) -> tuple[float, ...]:
        original = self.torch.as_tensor(tuple(activation)).float().reshape(-1)
        return self.decode(self.encode(original))

    def intervene(self, activation: Sequence[float], feature_index: int, value: float) -> tuple[float, ...]:
        original = self.torch.as_tensor(tuple(activation), device=self.device).float().reshape(-1)
        features = self.encode_tensor(original)
        if not 0 <= int(feature_index) < features.numel():
            raise IndexError("feature_index out of range")
        features[int(feature_index)] = float(value)
        return tuple(float(v) for v in self.decode_tensor(features).detach().cpu().tolist())


@dataclass(frozen=True)
class Gemma3Spec:
    """Frozen contract for the official Gemma 3 27B fallback path.

    State slice: ``interpretability-lab-sandbox-v1``.

    Gemma Scope 2 records its layer-31 transcoder hook points as
    ``model.layers.31.*``. The Transformers multimodal wrapper exposes the
    same language decoder below ``model.language_model.layers``; this adapter
    checks that exact registry and never searches alternate modules.
    """

    model_id: str
    revision: str
    model_root: Path
    layer: int = 31
    hook: str = "decoder_out"
    device: str = "cpu"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.model_id != "google/gemma-3-27b-pt":
            raise RealAdapterError("only the frozen Gemma 3 27B exploratory model is supported")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise RealAdapterError("model revision must be a 40-character hexadecimal commit")
        if self.layer != 31:
            raise RealAdapterError("canonical Gemma 3 Scope layer is 31")
        if self.hook != "decoder_out":
            raise RealAdapterError("the frozen Gemma 3 Scope asset requires decoder_out")
        if self.dtype not in {"bfloat16", "float32"}:
            raise RealAdapterError("Gemma 3 supports only bfloat16 or float32 in this adapter")

    @property
    def canonical_path(self) -> str:
        return f"model.language_model.layers[{self.layer}]"


class Gemma3Runtime:
    """Offline Gemma 3 27B text path with the fixed Scope layer-31 hook."""

    def __init__(self, spec: Gemma3Spec) -> None:
        root = Path(spec.model_root)
        if not root.is_dir():
            raise RealAdapterError(f"model root is missing: {root}")
        torch = _torch()
        try:
            from transformers import AutoProcessor  # type: ignore
            try:
                from transformers import Gemma3ForConditionalGeneration  # type: ignore
            except ImportError:
                from transformers import AutoModelForImageTextToText as Gemma3ForConditionalGeneration  # type: ignore
        except ImportError as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("a Transformers release with Gemma 3 support is required") from exc
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        self.torch = torch
        self.spec = spec
        try:
            self.processor = AutoProcessor.from_pretrained(root, local_files_only=True, revision=spec.revision)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                root,
                local_files_only=True,
                revision=spec.revision,
                torch_dtype=_dtype(torch, spec.dtype),
            )
        except Exception as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("Gemma 3 checkpoint could not be loaded offline") from exc
        self.model.to(spec.device)
        self.model.eval()
        wrapper = getattr(self.model, "model", None)
        language_model = getattr(wrapper, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None or len(layers) != 62 or spec.layer >= len(layers):
            raise RealAdapterError("Gemma 3 language-model decoder registry must contain 62 layers")
        model_config = getattr(self.model, "config", None)
        text_config = getattr(model_config, "text_config", model_config)
        if getattr(text_config, "hidden_size", None) != 5376:
            raise RealAdapterError("Gemma 3 model hidden width must be 5376")
        self.block = layers[spec.layer]
        self.hook_module = getattr(self.block, "post_feedforward_layernorm", None)
        if self.hook_module is None:
            raise RealAdapterError("Gemma 3 canonical post-feedforward hook is missing")
        self.input_hook_module = getattr(self.block, "pre_feedforward_layernorm", None)
        if self.input_hook_module is None:
            raise RealAdapterError("Gemma 3 canonical pre-feedforward hook is missing")

    def _encode(self, row: str) -> dict[str, Any]:
        if not isinstance(row, str) or not row.strip():
            raise RealAdapterError("rows must be nonempty text")
        try:
            encoded = self.processor(text=[row], return_tensors="pt", padding=False)
        except Exception as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("Gemma 3 text-only processor encoding failed") from exc
        return {key: value.to(self.spec.device) for key, value in encoded.items() if hasattr(value, "to")}

    def _run(self, row: str, *, capture: bool, patched_activation: Any | None = None, capture_all: bool = False, capture_input: bool = False) -> tuple[Any, Any | None]:
        encoded = self._encode(row)
        state: dict[str, Any] = {}

        def _patch(value: Any) -> Any:
            if value.ndim != 3:
                raise RealAdapterError("Gemma 3 decoder output must be [batch, sequence, hidden]")
            if patched_activation is not None:
                value = value.clone()
                if tuple(patched_activation.shape) != tuple(value[:, -1, :].shape):
                    raise RealAdapterError("patched activation shape does not match final token")
                value[:, -1, :] = patched_activation.to(value.device, dtype=value.dtype)
            if capture:
                state["activation_all" if capture_all else "activation"] = (
                    value.detach().float().cpu() if capture_all else value[:, -1, :].detach().float().cpu()
                )
            return value

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            value = _patch(value)
            return (value, *output[1:]) if isinstance(output, tuple) else value

        def input_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim != 3:
                raise RealAdapterError("Gemma 3 pre-feedforward output must be [batch, sequence, hidden]")
            state["activation_input_all"] = value.detach().float().cpu()
            return output

        handle = self.hook_module.register_forward_hook(hook) if (capture or patched_activation is not None) else None
        input_handle = self.input_hook_module.register_forward_hook(input_hook) if capture_input else None
        try:
            with self.torch.inference_mode():
                result = self.model(**encoded, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
            if input_handle is not None:
                input_handle.remove()
        logits = result.logits[:, -1, :].detach().float().cpu()
        output_key = "activation_all" if capture_all else "activation"
        if capture_input:
            return logits, (state.get("activation_input_all"), state.get(output_key))
        return logits, state.get(output_key)

    def logits(self, rows: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in self._run(row, capture=False)[0][0]) for row in rows)

    def capture(self, rows: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in self._run(row, capture=True)[1][0]) for row in rows)

    def capture_all(self, rows: Sequence[str]) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Capture every unpadded token at the frozen decoder hook."""
        captured: list[tuple[tuple[float, ...], ...]] = []
        for row in rows:
            value = self._run(row, capture=True, capture_all=True)[1]
            if value is None:
                raise RealAdapterError("Gemma 3 all-token capture produced no activation")
            captured.append(tuple(tuple(float(v) for v in token) for token in value[0]))
        return tuple(captured)

    def capture_transcoder(self, rows: Sequence[str]) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], tuple[tuple[tuple[float, ...], ...], ...]]:
        """Capture paired pre-feedforward inputs and post-feedforward outputs."""
        inputs: list[tuple[tuple[float, ...], ...]] = []
        outputs: list[tuple[tuple[float, ...], ...]] = []
        for row in rows:
            value = self._run(row, capture=True, capture_all=True, capture_input=True)[1]
            if not isinstance(value, tuple) or len(value) != 2 or value[0] is None or value[1] is None:
                raise RealAdapterError("Gemma 3 paired transcoder capture is incomplete")
            input_value, output_value = value
            inputs.append(tuple(tuple(float(v) for v in token) for token in input_value[0]))
            outputs.append(tuple(tuple(float(v) for v in token) for token in output_value[0]))
        return tuple(inputs), tuple(outputs)

    def patched_logits(self, row: str, activation: Sequence[float]) -> tuple[float, ...]:
        patch = self.torch.as_tensor(tuple(float(v) for v in activation), dtype=self.torch.float32, device=self.spec.device).reshape(1, -1)
        logits, _ = self._run(row, capture=False, patched_activation=patch)
        return tuple(float(v) for v in logits[0])


class Gemma3JumpReLUSAE:
    """Loader for the official Gemma Scope 2 JumpReLU SAE format."""

    def __init__(self, weights_path: Path, *, d_in: int = 5376, d_sae: int = 65536, device: str = "cpu") -> None:
        if not Path(weights_path).is_file():
            raise RealAdapterError(f"Gemma 3 SAE weights are missing: {weights_path}")
        torch = _torch()
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("safetensors is required for real feature execution") from exc
        try:
            weights = load_file(str(weights_path), device="cpu")
        except Exception as exc:  # pragma: no cover - real runner only
            raise RealAdapterError("Gemma Scope 2 parameters could not be loaded safely") from exc
        required = {"affine_skip_connection", "b_dec", "b_enc", "threshold", "w_dec", "w_enc"}
        if set(weights) != required:
            raise RealAdapterError("Gemma Scope 2 tensor roster mismatch")
        self.torch = torch
        try:
            target_device = torch.device(device)
        except Exception as exc:
            raise RealAdapterError("Gemma 3 SAE device is invalid") from exc
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RealAdapterError("Gemma 3 SAE CUDA device is unavailable")
        self.device = target_device
        self.b_dec = weights["b_dec"].float().to(target_device).reshape(-1)
        self.b_enc = weights["b_enc"].float().to(target_device).reshape(-1)
        self.threshold = weights["threshold"].float().to(target_device).reshape(-1)
        self.w_dec = weights["w_dec"].float().to(target_device)
        self.w_enc = weights["w_enc"].float().to(target_device)
        self.affine_skip_connection = weights["affine_skip_connection"].float().to(target_device)
        if tuple(self.w_enc.shape) != (d_in, d_sae) or tuple(self.w_dec.shape) != (d_sae, d_in):
            raise RealAdapterError("Gemma Scope 2 matrix dimensions do not match its frozen packet")
        if self.b_dec.numel() != d_in or self.b_enc.numel() != d_sae or self.threshold.numel() != d_sae:
            raise RealAdapterError("Gemma Scope 2 bias or threshold dimensions do not match its frozen packet")
        if tuple(self.affine_skip_connection.shape) != (d_in, d_in):
            raise RealAdapterError("Gemma Scope 2 affine skip dimensions do not match its frozen packet")

    def encode_tensor(self, activation: Any) -> Any:
        x = activation.float().to(self.device).reshape(-1)
        if x.numel() != self.b_dec.numel():
            raise RealAdapterError(f"activation width must be {self.b_dec.numel()}")
        # Gemma Scope 2 JumpReLU encoders use the stored affine encoder bias;
        # b_dec participates only in the decoder (the skip path is separate).
        pre = x @ self.w_enc + self.b_enc
        return (pre > self.threshold).to(pre.dtype) * self.torch.relu(pre)

    def encode(self, activation: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in self.encode_tensor(self.torch.as_tensor(tuple(activation))).tolist())

    def encode_batch_tensor(self, activations: Any) -> Any:
        x = activations.float().to(self.device)
        if x.ndim != 2 or x.shape[1] != self.b_dec.numel():
            raise RealAdapterError(f"activation batch must have shape [n, {self.b_dec.numel()}]")
        pre = x @ self.w_enc + self.b_enc
        return (pre > self.threshold).to(pre.dtype) * self.torch.relu(pre)

    def _skip(self, activation: Any) -> Any:
        if activation is None:
            raise RealAdapterError("Gemma Scope 2 affine decoding requires the original input activation")
        x = activation.float().to(self.device).reshape(-1)
        if x.numel() != self.b_dec.numel():
            raise RealAdapterError(f"activation width must be {self.b_dec.numel()}")
        return x @ self.affine_skip_connection

    def decode_tensor(self, features: Any, activation: Any | None = None) -> Any:
        f = features.float().to(self.device).reshape(-1)
        if f.numel() != self.w_dec.shape[0]:
            raise RealAdapterError(f"feature width must be {self.w_dec.shape[0]}")
        return f @ self.w_dec + self.b_dec + self._skip(activation)

    def decode(self, features: Sequence[float], activation: Sequence[float] | None = None) -> tuple[float, ...]:
        return tuple(float(v) for v in self.decode_tensor(self.torch.as_tensor(tuple(features)), activation=activation).tolist())

    def decode_batch_tensor(self, features: Any, activations: Any | None = None) -> Any:
        """Decode a two-dimensional feature batch without Python materialization."""
        f = features.float().to(self.device)
        if f.ndim != 2 or f.shape[1] != self.w_dec.shape[0]:
            raise RealAdapterError(f"feature batch must have shape [n, {self.w_dec.shape[0]}]")
        if activations is None:
            raise RealAdapterError("Gemma Scope 2 affine decoding requires the original input activations")
        x = activations.float().to(self.device)
        if x.ndim != 2 or x.shape[1] != self.b_dec.numel() or x.shape[0] != f.shape[0]:
            raise RealAdapterError(f"activation batch must have shape [n, {self.b_dec.numel()}]")
        return f @ self.w_dec + self.b_dec + x @ self.affine_skip_connection

    def reconstruct(self, activation: Sequence[float]) -> tuple[float, ...]:
        original = self.torch.as_tensor(tuple(activation), device=self.device).float().reshape(-1)
        return tuple(float(v) for v in self.decode_tensor(self.encode_tensor(original), activation=original).tolist())

    def reconstruct_batch(self, activations: Any) -> Any:
        """Reconstruct a two-dimensional activation batch on the frozen device."""
        x = activations.float().to(self.device)
        if x.ndim != 2 or x.shape[1] != self.b_dec.numel():
            raise RealAdapterError(f"activation batch must have shape [n, {self.b_dec.numel()}]")
        return self.decode_batch_tensor(self.encode_batch_tensor(x), activations=x)

    def intervene(self, activation: Sequence[float], feature_index: int, value: float) -> tuple[float, ...]:
        original = self.torch.as_tensor(tuple(activation), device=self.device).float().reshape(-1)
        features = self.encode_tensor(original)
        if not 0 <= int(feature_index) < features.numel():
            raise IndexError("feature_index out of range")
        features[int(feature_index)] = float(value)
        return tuple(float(v) for v in self.decode_tensor(features, activation=original).tolist())
