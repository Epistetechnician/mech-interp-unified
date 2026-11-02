"""Fail-closed real execution boundary.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .custody import CustodyError, require_owner_only, verify_packet_binding
from .review import ReviewVerificationError, verify_external_accept


class ExecutionClosed(RuntimeError):
    pass


def _configured_mode(config: Mapping[str, object], execution: Mapping[str, object]) -> object:
    """Read the canonical mode while tolerating the legacy top-level field."""
    return config.get("mode", execution.get("mode"))


def require_internal_execution(*, config: Mapping[str, object], packet: Mapping[str, object] | None, packet_digest: str | None, custody_root: Path | None, authorization: Mapping[str, object] | None, runtime_attestation: Mapping[str, object] | None) -> None:
    """Open only a packet-bound local exploratory run."""
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping) or _configured_mode(config, execution) != "internal-real" or not execution.get("real_model_enabled", False):
        raise ExecutionClosed("internal real execution is disabled")
    if execution.get("review_mode") != "internal":
        raise ExecutionClosed("internal review mode is not configured")
    if not isinstance(packet, Mapping) or not isinstance(packet_digest, str) or not packet_digest:
        raise ExecutionClosed("reviewed packet is required")
    if custody_root is None:
        raise ExecutionClosed("owner-only custody root is required")
    if not isinstance(authorization, Mapping):
        raise ExecutionClosed("internal authorization record is required")
    required = {"schema", "state_slice", "source", "decision", "purpose", "packet_digest", "operator_id", "claim_ceiling"}
    if set(authorization) != required:
        raise ExecutionClosed("internal authorization schema mismatch")
    if authorization.get("schema") != "interp-lab.internal-authorization.v1":
        raise ExecutionClosed("internal authorization schema mismatch")
    if authorization.get("state_slice") != "interpretability-lab-sandbox-v1":
        raise ExecutionClosed("internal authorization state slice mismatch")
    if authorization.get("source") != "internal" or authorization.get("decision") != "AUTHORIZE" or authorization.get("purpose") != "exploration":
        raise ExecutionClosed("internal authorization is not exploratory")
    if authorization.get("packet_digest") != packet_digest:
        raise ExecutionClosed("internal authorization packet mismatch")
    if not isinstance(authorization.get("operator_id"), str) or not authorization["operator_id"]:
        raise ExecutionClosed("internal authorization operator is incomplete")
    if authorization.get("claim_ceiling") != "LOCAL_INTERNAL_EXPLORATORY_ONLY":
        raise ExecutionClosed("internal claim ceiling is not bounded")
    if not isinstance(runtime_attestation, Mapping):
        raise ExecutionClosed("runtime attestation is required")
    attestation_required = {"schema", "state_slice", "source", "runtime", "device", "determinism"}
    if set(runtime_attestation) != attestation_required:
        raise ExecutionClosed("runtime attestation schema mismatch")
    if runtime_attestation.get("schema") != "interp-lab.runtime-attestation.v1" or runtime_attestation.get("state_slice") != "interpretability-lab-sandbox-v1" or runtime_attestation.get("source") != "internal":
        raise ExecutionClosed("runtime attestation identity mismatch")
    runtime = runtime_attestation.get("runtime")
    device = runtime_attestation.get("device")
    determinism = runtime_attestation.get("determinism")
    if not isinstance(runtime, Mapping) or not runtime.get("python") or not runtime.get("framework"):
        raise ExecutionClosed("runtime attestation runtime identity is incomplete")
    if not isinstance(device, Mapping) or not device.get("model") or not device.get("cuda"):
        raise ExecutionClosed("runtime attestation device identity is incomplete")
    if not isinstance(determinism, Mapping) or determinism.get("enabled") is not True:
        raise ExecutionClosed("deterministic execution is not attested")
    try:
        require_owner_only(custody_root)
        verify_packet_binding(packet, packet_digest)
    except CustodyError as error:
        raise ExecutionClosed(str(error)) from error


def require_real_execution(*, config: Mapping[str, object], packet: Mapping[str, object] | None, packet_digest: str | None, custody_root: Path | None, independent_receipt: Mapping[str, object] | None) -> None:
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping) or _configured_mode(config, execution) != "real" or not execution.get("real_model_enabled", False):
        raise ExecutionClosed("real model execution is disabled")
    if not isinstance(packet, Mapping) or not isinstance(packet_digest, str) or not packet_digest:
        raise ExecutionClosed("reviewed packet is required")
    if custody_root is None:
        raise ExecutionClosed("owner-only custody root is required")
    try:
        require_owner_only(custody_root)
        verify_packet_binding(packet, packet_digest)
    except CustodyError as error:
        raise ExecutionClosed(str(error)) from error
    if not isinstance(independent_receipt, Mapping):
        raise ExecutionClosed("independent packet-bound ACCEPT is required")
    try:
        verify_external_accept(independent_receipt, packet_digest)
    except ReviewVerificationError as error:
        raise ExecutionClosed(f"independent packet-bound ACCEPT is required: {error}") from error


def require_execution(*, config: Mapping[str, object], packet: Mapping[str, object] | None, packet_digest: str | None, custody_root: Path | None, authorization: Mapping[str, object] | None = None, independent_receipt: Mapping[str, object] | None = None, runtime_attestation: Mapping[str, object] | None = None) -> None:
    """Dispatch to the explicitly configured internal or external gate."""
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ExecutionClosed("execution policy is malformed")
    mode = execution.get("review_mode")
    if mode == "internal":
        require_internal_execution(
            config=config,
            packet=packet,
            packet_digest=packet_digest,
            custody_root=custody_root,
            authorization=authorization,
            runtime_attestation=runtime_attestation,
        )
        return
    if mode == "external":
        require_real_execution(
            config=config,
            packet=packet,
            packet_digest=packet_digest,
            custody_root=custody_root,
            independent_receipt=independent_receipt,
        )
        return
    raise ExecutionClosed("review mode must be internal or external")
