from . import (control_omission, device_dtype, label_independent, planted_superposition, post_hoc, repr_identity,
               saturated_metric, selection_leakage)

NET_CLASSES = [label_independent, selection_leakage, device_dtype, repr_identity, saturated_metric, post_hoc,
               control_omission]
ALL_CLASSES = [label_independent, selection_leakage, device_dtype, repr_identity, planted_superposition,
               saturated_metric, post_hoc, control_omission]

from . import best_of_m, underpowered_null, unmatched_controls  # noqa: E402

PHASE2_CLASSES = [unmatched_controls, underpowered_null, best_of_m]
