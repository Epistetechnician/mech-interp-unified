"""Render markdown tables from audit/results/*.json (used to build REPORT.md; no new measurement)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
AUD = [("rules_gate", "A1 rules gate"), ("r5b_audit", "A2 R5b audit"), ("probe_battery", "A3 probe battery")]


def ci(r):
    return f"{r['rate']:.3f} [{r['boot_ci'][0]:.3f}, {r['boot_ci'][1]:.3f}]"


def main():
    d = json.loads((R / "f2_phase1.json").read_text())
    out = []
    for key, name in AUD:
        out.append(f"\n**{name}** (n per cell: 200 seeds x variants; bootstrap 95% CI, Wilson in JSON)\n")
        out.append("| class | TPR | FPR | label | TPR where claim is wrong | n/a |")
        out.append("|---|---|---|---|---|---|")
        for c, r in d["auditors"][key].items():
            if c.startswith("_"):
                continue
            if r["tpr"]["n"] == 0:
                out.append(f"| {c} | n/a | n/a | N/A | n/a | {r['n_na']} |")
                continue
            w = r["tpr_claim_wrong"]
            ws = f"{w['rate']:.2f} ({w['k']}/{w['n']})" if w["n"] else "n/a"
            out.append(f"| {c} | {ci(r['tpr'])} | {ci(r['fpr'])} | {r['label']} | {ws} | {r['n_na']} |")
    out.append("\n**Per-variant TPR (FPR)**\n")
    out.append("| auditor | class/variant | TPR | FPR |")
    out.append("|---|---|---|---|")
    for key, name in AUD:
        for c, r in d["auditors"][key].items():
            if c.startswith("_") or r["tpr"]["n"] == 0:
                continue
            for v, pv in r["per_variant"].items():
                out.append(f"| {key} | {c}/{v} | {pv['tpr']:.2f} | {pv['fpr']:.3f} |")
    p = R / "f2_goodhart_scripted.json"
    if p.exists():
        g = json.loads(p.read_text())
        out.append("\n**Scripted adversary (GOODHART C0)**\n")
        out.append("| target/adversary | successes | rate [boot 95% CI] | median queries | component | transfer flag rate |")
        out.append("|---|---|---|---|---|---|")
        for k, r in g["runs"].items():
            tf = [v for kk, v in r.items() if kk.startswith("transfer")]
            out.append(f"| {k} | {r['successes']}/{r['n']} | {ci(r)} | {r['median_queries_to_success']} | "
                       f"{r['decision_component']} | {tf[0] if tf else None} |")
        out.append(f"\nDecision: {g['decision']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
