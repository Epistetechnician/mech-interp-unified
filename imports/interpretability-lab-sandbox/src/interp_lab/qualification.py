"""Fixed-order synthetic qualification and aggregate-only report.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapters import InstrumentedAdapter, NativeAdapter, SyntheticFeatureAdapter, SyntheticModel
from .canonical import digest
from .metrics import cosine, max_abs_delta, nmse


def _rows() -> tuple[str, ...]:
    return ("alpha 01", "beta 02", "gamma 03", "delta 04", "epsilon 05", "zeta 06")


def _gate(name: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **values}


def run_qualification(*, seed: int = 20260914, thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Run parity, repeatability, no-op, reach, reconstruction, stability, and scrubbing."""
    del seed  # The fixed fixture has no stochastic branch; the field remains packet-visible.
    limits = {"parity_abs_tol": 0.0, "repeatability_abs_tol": 0.0, "no_op_abs_tol": 0.0, "reconstruction_nmse_max": 0.05, "intervention_min_abs_effect": 1e-9, "stability_min_cosine": 0.8, "scrub_min_abs_effect": 1e-9}
    if thresholds:
        limits.update({key: float(value) for key, value in thresholds.items()})
    rows = _rows()
    native = NativeAdapter(SyntheticModel())
    instrumented = InstrumentedAdapter(SyntheticModel())
    feature = SyntheticFeatureAdapter()
    native_observation = native.forward(rows, capture=True)
    instrumented_observation = instrumented.forward(rows, capture=True)
    parity_delta = max_abs_delta(native_observation.logits, instrumented_observation.logits)
    gates = [_gate("native_instrumented_parity", parity_delta <= limits["parity_abs_tol"], max_abs_logit_delta=parity_delta)]
    repeat = NativeAdapter(SyntheticModel()).forward(rows, capture=True)
    repeat_delta = max_abs_delta(native_observation.logits, repeat.logits)
    gates.append(_gate("deterministic_repeatability", repeat_delta <= limits["repeatability_abs_tol"], max_abs_logit_delta=repeat_delta))
    original = native_observation.activations[0]
    no_op = feature.intervene(original, 0, original[0])
    no_op_delta = max_abs_delta(original, no_op)
    gates.append(_gate("exact_no_op", no_op_delta <= limits["no_op_abs_tol"], max_abs_activation_delta=no_op_delta))
    changed = feature.intervene(original, 0, original[0] + 0.25)
    base_logit = sum(weight * value for weight, value in zip(native.model.weights, original))
    changed_logit = sum(weight * value for weight, value in zip(native.model.weights, changed))
    intervention_effect = abs(changed_logit - base_logit)
    gates.append(_gate("nonzero_intervention_reach", intervention_effect >= limits["intervention_min_abs_effect"], absolute_logit_effect=intervention_effect))
    reconstruction_errors = [nmse(activation, feature.decode(feature.encode(activation))) for activation in native_observation.activations]
    reconstruction_max = max(reconstruction_errors)
    gates.append(_gate("feature_reconstruction", reconstruction_max <= limits["reconstruction_nmse_max"], max_nmse=reconstruction_max))
    checkpoint_a = tuple(feature.encode(activation) for activation in native_observation.activations)
    checkpoint_b = tuple(feature.encode(tuple(value * 1.001 for value in activation)) for activation in native_observation.activations)
    stability = min(cosine(a, b) for a, b in zip(checkpoint_a, checkpoint_b))
    gates.append(_gate("feature_stability", stability >= limits["stability_min_cosine"], minimum_cosine=stability))
    scrubbed = feature.intervene(original, 0, 0.0)
    scrub_effect = abs(sum(weight * value for weight, value in zip(native.model.weights, scrubbed)) - base_logit)
    gates.append(_gate("causal_scrubbing", scrub_effect >= limits["scrub_min_abs_effect"], absolute_logit_effect=scrub_effect))
    event_chain = "0" * 64
    for event in instrumented_observation.events:
        event_chain = digest({"previous": event_chain, "event": event})
    report = {"schema":"interp-lab.qualification-report.v1","state_slice":"interpretability-lab-sandbox-v1","claim_status":"LOCAL_SYNTHETIC_ONLY","seed":20260914,"gate_order":tuple(gate["name"] for gate in gates),"gates":tuple(gates),"all_gates_pass":all(gate["passed"] for gate in gates),"event_chain_digest":event_chain,"nonclaims":("no real model evidence","no causal validity beyond the synthetic fixture","no alignment or continual-learning superiority claim","no production authorization")}
    return {**report, "report_digest": digest(report)}
