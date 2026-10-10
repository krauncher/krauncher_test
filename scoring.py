"""How a forecast field is compared with its expected value — shared by
run.py (one task) and report.py (all tasks)."""

PARAMS_TOL = 0.25                              # params_billions within +/-25 %
_NORM = {"precision": lambda v: v or "fp32"}   # no precision in the code is fp32

# The reference card of the ladder (compute_ratio 1.0 by definition).
REFERENCE_GPU = "rtx_6000_blackwell"

# Measured values and the ladder: scored by their error, not pass / fail.
MEASURED = ("reference_sec", "min_vram_gb", "compute_ratio", "anchor_compute_sec")


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


def anchor_time(meas: dict, got: dict) -> tuple[float, float] | None:
    """(measured, forecast) compute time on the set's anchor GPU, seconds.

    Measured: `anchor_compute_sec` of the set (a published benchmark gives it
    as work / throughput). Forecast: the service's compute time on the
    reference card times its own ratio for the anchor GPU. None when either
    side is missing.
    """
    m, anchor = meas.get("anchor_compute_sec"), meas.get("anchor_gpu") or REFERENCE_GPU
    ref, ratio = got.get("compute_sec"), (got.get("compute_ratio") or {}).get(anchor)
    if not m or not ref or not ratio:
        return None
    return m, ref * ratio


# Status of a forecast value against what was measured: "good" (accurate),
# "warning" (a noticeable miss), "critical" (a problem). Thresholds on
# |log(forecast / measured)|; the report states them.
TIME_GOOD, TIME_WARN = 1.10, 1.25      # reference_sec
RATIO_GOOD, RATIO_WARN = 1.15, 1.50    # compute_ratio per GPU
VRAM_OVER_WARN = 1.30                  # forecast / measured peak above this: over-provisioning


def time_status(forecast: float, measured: float) -> str:
    """By the deviation alone. Whether the measurement falls inside the spread
    the service issued is reported beside it: that is the honesty of the
    interval, not the accuracy of the forecast."""
    r = forecast / measured
    e = max(r, 1 / r)
    return "good" if e <= TIME_GOOD else "warning" if e <= TIME_WARN else "critical"


def ratio_status(forecast: float, measured: float) -> str:
    r = forecast / measured
    e = max(r, 1 / r)
    return "good" if e <= RATIO_GOOD else "warning" if e <= RATIO_WARN else "critical"


def vram_status(forecast: float, measured: float) -> str:
    """Below the measured peak is an OOM risk; far above it, over-provisioning."""
    if forecast < measured:
        return "critical"
    return "good" if forecast / measured <= VRAM_OVER_WARN else "warning"
