import zlib

import numpy as np

from .base import FLAG, PASS, Auditor


class AlwaysPass(Auditor):
    name = "always_pass"

    def audit(self, case):
        return PASS


class AlwaysFlag(Auditor):
    name = "always_flag"

    def audit(self, case):
        return FLAG


class CoinFlip(Auditor):
    name = "coin_flip"

    def audit(self, case):
        rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(case.case_id.encode()), 424242]))
        return FLAG if rng.random() < 0.5 else PASS
