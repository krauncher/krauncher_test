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


def ladder_pairs(meas: dict, got: dict | None) -> dict[str, tuple[float, float | None]]:
    """gpu_id -> (measured ratio, forecast ratio) for a measurement set.

    A set's compute_ratio is relative to the reference card unless it names an
    `anchor_gpu` (a published benchmark seldom ran on the reference card): then
    the measured values are relative to that GPU, and the forecast is put on
    the same footing (forecast[g] / forecast[anchor]). Without the anchor in
    the forecast nothing is comparable.
    """
    exp, got = meas.get("compute_ratio") or {}, got or {}
    anchor = meas.get("anchor_gpu")
    if anchor:
        if not got.get(anchor):
            return {g: (r, None) for g, r in exp.items()}
        return {g: (r, got[g] / got[anchor] if g in got else None) for g, r in exp.items()}
    return {g: (r, got.get(g)) for g, r in exp.items()}
