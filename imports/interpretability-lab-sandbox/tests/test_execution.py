"""Fail-closed execution tests.

State slice: interpretability-lab-sandbox-v1.
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from interp_lab.canonical import digest
from interp_lab.custody import CustodyError, file_manifest, require_internal_custody_binding
from interp_lab.execution import ExecutionClosed, require_execution, require_internal_execution, require_real_execution


def runtime_attestation() -> dict[str, object]:
    return {
        "schema": "interp-lab.runtime-attestation.v1",
        "state_slice": "interpretability-lab-sandbox-v1",
        "source": "internal",
        "runtime": {"python": "3.11.11", "framework": "synthetic-1"},
        "device": {"model": "synthetic-cpu", "cuda": "none"},
        "determinism": {"enabled": True},
    }


class ExecutionTests(unittest.TestCase):
    def test_synthetic_config_cannot_open_real_execution(self):
        with self.assertRaisesRegex(ExecutionClosed, "disabled"):
            require_real_execution(config={"mode":"synthetic", "execution":{"real_model_enabled":False}}, packet=None, packet_digest=None, custody_root=None, independent_receipt=None)

    def test_internal_execution_requires_runtime_attestation(self):
        packet = {"schema": "interp-lab.packet.v1", "state_slice": "interpretability-lab-sandbox-v1", "payload": "fixture"}
        packet["packet_digest"] = digest(packet)
        authorization = {
            "schema": "interp-lab.internal-authorization.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "source": "internal",
            "decision": "AUTHORIZE",
            "purpose": "exploration",
            "packet_digest": packet["packet_digest"],
            "operator_id": "operator-local",
            "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        }
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            with self.assertRaisesRegex(ExecutionClosed, "runtime attestation is required"):
                require_internal_execution(
                    config={"mode": "internal-real", "execution": {"real_model_enabled": True, "review_mode": "internal"}},
                    packet=packet,
                    packet_digest=packet["packet_digest"],
                    custody_root=custody,
                    authorization=authorization,
                    runtime_attestation=None,
                )

    def test_internal_authorization_opens_only_exact_bound_packet(self):
        packet = {"schema": "interp-lab.packet.v1", "state_slice": "interpretability-lab-sandbox-v1", "payload": "fixture"}
        packet["packet_digest"] = digest(packet)
        authorization = {
            "schema": "interp-lab.internal-authorization.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "source": "internal",
            "decision": "AUTHORIZE",
            "purpose": "exploration",
            "packet_digest": packet["packet_digest"],
            "operator_id": "operator-local",
            "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        }
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            (custody / "manifest.json").write_text(json.dumps({"state_slice": "interpretability-lab-sandbox-v1"}))
            require_internal_execution(
                config={"mode": "internal-real", "execution": {"real_model_enabled": True, "review_mode": "internal"}},
                packet=packet,
                packet_digest=packet["packet_digest"],
                custody_root=custody,
                authorization=authorization,
                runtime_attestation=runtime_attestation(),
            )

    def test_internal_packet_mode_may_be_nested_under_execution(self):
        packet = {"schema": "interp-lab.packet.v1", "state_slice": "interpretability-lab-sandbox-v1", "payload": "fixture"}
        packet["packet_digest"] = digest(packet)
        authorization = {
            "schema": "interp-lab.internal-authorization.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "source": "internal",
            "decision": "AUTHORIZE",
            "purpose": "exploration",
            "packet_digest": packet["packet_digest"],
            "operator_id": "operator-local",
            "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        }
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            require_internal_execution(
                config={"execution": {"mode": "internal-real", "real_model_enabled": True, "review_mode": "internal"}},
                packet=packet,
                packet_digest=packet["packet_digest"],
                custody_root=custody,
                authorization=authorization,
                runtime_attestation=runtime_attestation(),
            )

    def test_dispatcher_uses_internal_mode_explicitly(self):
        packet = {"schema": "interp-lab.packet.v1", "state_slice": "interpretability-lab-sandbox-v1", "payload": "fixture"}
        packet["packet_digest"] = digest(packet)
        authorization = {
            "schema": "interp-lab.internal-authorization.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "source": "internal",
            "decision": "AUTHORIZE",
            "purpose": "exploration",
            "packet_digest": packet["packet_digest"],
            "operator_id": "operator-local",
            "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        }
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            require_execution(
                config={"mode": "internal-real", "execution": {"real_model_enabled": True, "review_mode": "internal"}},
                packet=packet,
                packet_digest=packet["packet_digest"],
                custody_root=custody,
                authorization=authorization,
                runtime_attestation=runtime_attestation(),
            )

    def test_internal_authorization_rejects_packet_substitution(self):
        packet = {"schema": "interp-lab.packet.v1", "state_slice": "interpretability-lab-sandbox-v1", "payload": "fixture"}
        packet["packet_digest"] = digest(packet)
        authorization = {
            "schema": "interp-lab.internal-authorization.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "source": "internal",
            "decision": "AUTHORIZE",
            "purpose": "exploration",
            "packet_digest": "0" * 64,
            "operator_id": "operator-local",
            "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        }
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            with self.assertRaisesRegex(ExecutionClosed, "packet mismatch"):
                require_internal_execution(
                    config={"mode": "internal-real", "execution": {"real_model_enabled": True, "review_mode": "internal"}},
                    packet=packet,
                    packet_digest=packet["packet_digest"],
                    custody_root=custody,
                    authorization=authorization,
                    runtime_attestation=runtime_attestation(),
                )

    def test_external_accept_must_bind_packet_digest(self):
        packet = {"schema": "interp-lab.packet.v1", "state_slice": "interpretability-lab-sandbox-v1", "payload": "fixture"}
        packet["packet_digest"] = digest(packet)
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            with self.assertRaisesRegex(ExecutionClosed, "packet-bound ACCEPT"):
                require_real_execution(
                    config={"mode": "real", "execution": {"real_model_enabled": True}},
                    packet=packet,
                    packet_digest=packet["packet_digest"],
                    custody_root=custody,
                    independent_receipt={"source": "external", "decision": "ACCEPT", "packet_digest": "0" * 64},
                )

    def test_internal_custody_manifest_binds_all_real_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            model_root = root / "model"
            model_root.mkdir()
            os.chmod(model_root, 0o700)
            (model_root / "config.json").write_text("{}", encoding="utf-8")
            sae_path = root / "sae.pt"
            corpus_path = root / "corpus.json"
            authorization_path = root / "authorization.json"
            attestation_path = root / "runtime-attestation.json"
            for path, payload in (
                (sae_path, b"weights"),
                (corpus_path, b"corpus"),
                (authorization_path, b"authorization"),
                (attestation_path, b"attestation"),
            ):
                path.write_bytes(payload)

            def sha256(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            packet = {
                "schema": "interp-lab.packet.v1",
                "state_slice": "interpretability-lab-sandbox-v1",
                "model": {"checkpoints": {"base": {"sha256": file_manifest(model_root)["manifest_digest"]}}},
                "feature_asset": {"weights_sha256": sha256(sae_path)},
                "corpus": {"sha256": sha256(corpus_path)},
                "runtime": {"transformers": "4.57.3"},
            }
            packet["packet_digest"] = digest(packet)
            components = {
                "model": {"path": "model", "sha256": file_manifest(model_root)["manifest_digest"]},
                "sae": {"path": "sae.pt", "sha256": sha256(sae_path)},
                "corpus": {"path": "corpus.json", "sha256": sha256(corpus_path)},
                "authorization": {"path": "authorization.json", "sha256": sha256(authorization_path)},
                "runtime_attestation": {"path": "runtime-attestation.json", "sha256": sha256(attestation_path)},
            }
            manifest = {
                "schema": "interp-lab.internal-custody-manifest.v1",
                "state_slice": "interpretability-lab-sandbox-v1",
                "packet_digest": packet["packet_digest"],
                "components": components,
            }
            manifest["manifest_digest"] = digest(manifest)
            (root / "custody-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = require_internal_custody_binding(
                root=root,
                packet=packet,
                packet_digest=packet["packet_digest"],
                model_root=model_root,
                sae_weights=sae_path,
                corpus=corpus_path,
                authorization=authorization_path,
                runtime_attestation=attestation_path,
            )
            self.assertEqual(result["manifest"]["manifest_digest"], manifest["manifest_digest"])

    def test_internal_custody_rejects_unbound_packet_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            model_root = root / "model"
            model_root.mkdir()
            os.chmod(model_root, 0o700)
            (model_root / "weights").write_bytes(b"model")
            for name, payload in (("sae.pt", b"sae"), ("corpus.json", b"corpus"), ("authorization.json", b"auth"), ("runtime-attestation.json", b"attest")):
                (root / name).write_bytes(payload)
            packet = {
                "schema": "interp-lab.packet.v1",
                "state_slice": "interpretability-lab-sandbox-v1",
                "model": {"checkpoints": {"base": {"sha256": "0" * 64}}},
                "feature_asset": {"weights_sha256": "0" * 64},
                "corpus": {"sha256": "0" * 64},
                "runtime": {"transformers": "4.57.3"},
            }
            packet["packet_digest"] = digest(packet)
            components = {
                "model": {"path": "model", "sha256": file_manifest(model_root)["manifest_digest"]},
                "sae": {"path": "sae.pt", "sha256": hashlib.sha256((root / "sae.pt").read_bytes()).hexdigest()},
                "corpus": {"path": "corpus.json", "sha256": hashlib.sha256((root / "corpus.json").read_bytes()).hexdigest()},
                "authorization": {"path": "authorization.json", "sha256": hashlib.sha256((root / "authorization.json").read_bytes()).hexdigest()},
                "runtime_attestation": {"path": "runtime-attestation.json", "sha256": hashlib.sha256((root / "runtime-attestation.json").read_bytes()).hexdigest()},
            }
            manifest = {"schema": "interp-lab.internal-custody-manifest.v1", "state_slice": "interpretability-lab-sandbox-v1", "packet_digest": packet["packet_digest"], "components": components}
            manifest["manifest_digest"] = digest(manifest)
            (root / "custody-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CustodyError, "model digest is not custody-bound"):
                require_internal_custody_binding(root=root, packet=packet, packet_digest=packet["packet_digest"], model_root=model_root, sae_weights=root / "sae.pt", corpus=root / "corpus.json", authorization=root / "authorization.json", runtime_attestation=root / "runtime-attestation.json")
