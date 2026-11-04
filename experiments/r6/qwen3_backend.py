"""Qwen3 adapter for the Astral V50-V60 backend chain (which was written for GPT-NeoX).

The chain hooks `layers_mod[l].mlp.dense_4h_to_h` pre-forward: the input of that projection is the MLP hidden
unit vector. For Qwen3 that is `mlp.down_proj`'s input (silu(gate) * up), the same object under another name. So
this adapter aliases names rather than changing any measurement logic:
  mlp.dense_4h_to_h -> mlp.down_proj;  m.gpt_neox -> m.model;  model.final_layer_norm -> model.norm (+ .eps).
Limit: unit_stats also computes direct-logit-attribution fields that assume LayerNorm centering. They are wrong
for RMSNorm and are never read by r6.py (only act_sing/act_plur are).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r6  # noqa: F401,E402  (sets sys.path for the imported tool modules)
import v60  # noqa: E402
from pythia_backend import PATH_DOCS  # noqa: E402


def make_backend_class():
    base = v60.make_backend_class()

    class Qwen3Backend(base):
        def __init__(self, model_dir, split_items, target_layers, device="mps"):
            if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
                raise RuntimeError("offline env required: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch, self.dev = torch, device
            self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
            if m.config.model_type != "qwen3":
                raise RuntimeError("not a qwen3 model")
            self.m = m.float().to(device).eval()  # bf16 checkpoint weights cast to float32 (protocol dtype)
            for p in self.m.parameters():
                p.requires_grad_(False)
            self.m.gpt_neox = self.m.model
            self.m.model.final_layer_norm = self.m.model.norm
            self.m.model.norm.eps = self.m.model.norm.variance_epsilon
            self.layers_mod = self.m.model.layers
            for layer in self.layers_mod:
                layer.mlp.dense_4h_to_h = layer.mlp.down_proj
            self.LY = tuple(target_layers)
            self.n_layers = self.m.config.num_hidden_layers
            self.I = self.m.config.intermediate_size
            self._items, self._split = {}, {}
            for part, items in split_items.items():
                for it in items:
                    self._items[it["id"]] = (it["text"], it["good"], it["bad"])
                    self._split.setdefault(part, []).append(it["id"])
            for _, g, b in self._items.values():
                for t in (g, b):
                    if len(self.tok.encode(t, add_special_tokens=False)) != 1:
                        raise RuntimeError("verb not a single token: %r" % t)
            self.mask = {k: torch.ones(self.I, device=device, dtype=torch.float32) for k in range(len(self.LY))}
            for k, l in enumerate(self.LY):
                self.layers_mod[l].mlp.dense_4h_to_h.register_forward_pre_hook(self._mask_hook(k))
            self.capture, self.acts = False, {}
            self.locked, self._base, self._enc = None, {}, {}

        def activation_values(self, selected, docs):
            """Last-position activation of the selected units at the measurement hook point (input of down_proj,
            silu(gate)*up). The Pythia version hooks dense_h_to_4h output (a pre-GELU value); SwiGLU has no single
            pre-activation, and silu(gate)*up is signed, so the gate's both-signs check is meaningful here.
            Addendum 2 (docs/protocols/R6-second-model-agreement-gate.md)."""
            want = {}
            for u in selected:
                want.setdefault(u // self.I, []).append(u % self.I)
            vals, hs = [], []
            for k, idx in want.items():
                def hook(mod, args, idx=idx):
                    vals.extend(args[0][:, -1][:, idx].float().flatten().tolist())
                hs.append(self.layers_mod[self.LY[k]].mlp.dense_4h_to_h.register_forward_pre_hook(hook))
            self._reset()
            try:
                self._mean(sorted(docs)[:PATH_DOCS])
            finally:
                for h in hs:
                    h.remove()
            return vals

    return Qwen3Backend
