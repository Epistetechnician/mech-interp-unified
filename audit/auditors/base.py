FLAG, PASS, NA = "FLAG", "PASS", "NA"


class Auditor:
    name = "base"

    def audit(self, case) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def explain(self, case) -> list:
        """Names of the checks that fired (for diagnostics; not used in metrics)."""
        return []
