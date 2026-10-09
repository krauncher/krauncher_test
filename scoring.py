"""How a forecast field is compared with its expected value — shared by
run.py (one task) and report.py (all tasks)."""

PARAMS_TOL = 0.25                              # params_billions within +/-25 %
_NORM = {"precision": lambda v: v or "fp32"}   # no precision in the code is fp32

# Measured values and the ladder: scored by their error, not pass / fail.
MEASURED = ("reference_sec", "min_vram_gb", "compute_ratio")


def matches(field: str, want, got) -> bool:
    """Pass / fail of a classification / assay field."""
    if field == "params_billions":
        return got is not None and abs(got - want) <= PARAMS_TOL * want
    n = _NORM.get(field, lambda v: v)
    return n(want) == n(got)
