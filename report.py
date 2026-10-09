"""Report over the frozen forecasts in results/, per service.

For every (service, task) the latest result is taken (optionally only those of
one service version, e.g. a Krauncher calibration_id) and compared with the
task's description (tasks/<task>.json). Levels:

0. forecast time — wall clock from request to the normalised forecast,
   measured by the stand for every service (plus the service's own parts);
1. classification / assay fields — share of tasks where the field matches;
2. reference-card time — forecast / reference per task: centre (geomean),
   typical deviation (median |log|), share of references inside the issued
   spread (reference / forecast within 1/spread_factor .. spread_factor);
3. VRAM — forecast / reference, and how many forecasts are below the measured
   peak (an OOM risk);
4. ladder — compute_ratio per GPU: error (median and p90 of |log|), card order
   (Spearman rank correlation), and time regret of the pick: how much slower
   the GPU the forecast calls fastest really is than the really fastest one.
   Regret in $/task needs prices, which no service answers yet.

Only fields a service answers (its result's `fields`) and the task describes
are scored. Levels 2-4 are reported per measurement set of the descriptions
(each set has an id, a source, a date and whether it is independent of the
service's calibration). Markdown to stdout.

    python report.py [--service krauncher] [--version c-d589e2f979e6] [--measurement <id>]
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from scoring import MEASURED, matches

ROOT = Path(__file__).parent


def latest(service: str | None, version: str | None) -> dict[tuple[str, str], dict]:
    """(service, task) -> its latest result (new format: results/<service>/)."""
    out: dict[tuple[str, str], dict] = {}
    for f in sorted((ROOT / "results").glob("*/*.json")):
        r = json.loads(f.read_text())
        if service and r["service"] != service:
            continue
        if version and version not in r["version"].values():
            continue
        key = (r["service"], r["task"])
        if key not in out or r["created_utc"] > out[key]["created_utc"]:
            out[key] = r
    return out


def _gm(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


def _q(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def report(results: dict, descs: dict, measurement: str | None = None) -> str:
    mids = [measurement] if measurement else list(dict.fromkeys(
        m["id"] for d in descs.values() for m in d.get("measurements", [])))
    lines = []
    for service in sorted({s for s, _ in results}):
        rs = {t: r for (s, t), r in results.items() if s == service and t in descs}
        versions = sorted({json.dumps(r["version"], sort_keys=True) for r in rs.values()})
        lines += [f"## {service}", "", f"{len(rs)} tasks; versions: " + "; ".join(versions), ""]

        # 0. forecast time: how long the service takes to answer
        timed = {t: r.get("timing") or {} for t, r in rs.items() if (r.get("timing") or {}).get("forecast_sec")}
        if timed:
            ts = [v["forecast_sec"] for v in timed.values()]
            keys = sorted({k for v in timed.values() for k in v} - {"forecast_sec"})
            lines += ["### Forecast time", "",
                      f"request -> normalised forecast over {len(ts)} tasks: median {statistics.median(ts):.1f} s, "
                      f"p90 {_q(ts, 0.9):.1f} s, max {max(ts):.1f} s"
                      + (f"; {len(rs) - len(timed)} results without timing" if len(timed) < len(rs) else ""), "",
                      "| task | forecast_sec | " + " | ".join(keys) + " |",
                      "|---|---|" + "---|" * len(keys)]
            lines += [f"| {t} | {v['forecast_sec']} | " + " | ".join(str(v.get(k, "-")) for k in keys) + " |"
                      for t, v in timed.items()]
            lines.append("")

        # 1. classification / assay fields
        per_field: dict[str, list[tuple[str, bool]]] = {}
        for t, r in rs.items():
            exp, got = descs[t]["fields"], r["fields"]
            for f, w in exp.items():
                if f in MEASURED or f not in got:
                    continue
                per_field.setdefault(f, []).append((t, matches(f, w, got[f])))
        lines += ["### Classification / assay fields", "", "| field | match | differs in |", "|---|---|---|"]
        for f, v in per_field.items():
            ok = sum(m for _, m in v)
            lines.append(f"| {f} | {ok}/{len(v)} | {', '.join(t for t, m in v if not m) or '-'} |")
        lines.append("")

        # 2-4. measured values, per measurement set of the descriptions
        for mid in mids:
            def meas(t, mid=mid):
                return next((m for m in descs[t].get("measurements", []) if m["id"] == mid), {})
            have = [t for t in rs if meas(t)]
            if not have:
                continue
            indep = sorted({str(meas(t).get("independent")) for t in have})
            lines += [f"### Measurements: {mid} (independent: {', '.join(indep)}; {len(have)} tasks)", ""]
            # 2. reference-card time, 3. VRAM
            for f, title in (("reference_sec", "Reference-card time"), ("min_vram_gb", "VRAM")):
                pairs = [(t, meas(t)[f], r["fields"][f]) for t, r in rs.items()
                         if meas(t).get(f) and r["fields"].get(f)]
                if not pairs:
                    continue
                ratios = [g / w for _, w, g in pairs]
                head = (f"forecast / reference over {len(pairs)} tasks: centre x{_gm(ratios):.2f}, "
                        f"typical deviation x{math.exp(statistics.median(abs(math.log(x)) for x in ratios)):.2f}")
                if f == "reference_sec":
                    inside = sum(1 for (t, w, g) in pairs
                                 if 1 / rs[t]["fields"].get("spread_factor", 1) <= w / g <= rs[t]["fields"].get("spread_factor", 1))
                    head += f"; reference inside the issued spread: {inside}/{len(pairs)}"
                else:
                    head += f"; forecast below the measured peak: {sum(1 for x in ratios if x < 1)}/{len(pairs)}"
                lines += [f"#### {title}", "", head]
                lines += ["", "| task | reference | forecast | forecast / reference |", "|---|---|---|---|"]
                lines += [f"| {t} | {w} | {g} | x{g / w:.2f} |" for t, w, g in pairs]
                lines.append("")

            # 4. ladder
            rows, all_err = [], []
            for t, r in rs.items():
                exp, got = meas(t).get("compute_ratio") or {}, r["fields"].get("compute_ratio") or {}
                gpus = [g for g in exp if g in got]
                if len(gpus) < 3:
                    continue
                err = [abs(math.log(got[g] / exp[g])) for g in gpus]
                all_err += err
                pick = min(gpus, key=lambda g: got[g])
                best = min(gpus, key=lambda g: exp[g])
                rows.append((t, len(gpus), math.exp(statistics.median(err)), math.exp(_q(err, 0.9)),
                             _spearman([exp[g] for g in gpus], [got[g] for g in gpus]),
                             exp[pick] / exp[best], pick, best))
            if rows:
                lines += ["#### Ladder (compute_ratio)", "",
                          f"over {len(all_err)} (task, GPU) pairs: typical error x{math.exp(statistics.median(all_err)):.2f}, "
                          f"p90 x{math.exp(_q(all_err, 0.9)):.2f}; median rank correlation "
                          f"{statistics.median(x[4] for x in rows):.2f}; pick slower than the fastest in "
                          f"{sum(1 for x in rows if x[5] > 1.0001)}/{len(rows)} tasks", "",
                          "| task | GPUs | typical error | p90 | rank corr | time regret of the pick | picked | fastest |",
                          "|---|---|---|---|---|---|---|---|"]
                lines += [f"| {t} | {n} | x{m:.2f} | x{p:.2f} | {rc:.2f} | x{rg:.2f} | {pk} | {bs} |"
                          for t, n, m, p, rc, rg, pk, bs in rows]
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service")
    ap.add_argument("--version", help="only results of this service version (e.g. a calibration_id)")
    ap.add_argument("--measurement", help="only this measurement set of the descriptions (default: each)")
    args = ap.parse_args()
    descs = {p.stem: json.loads(p.read_text()) for p in (ROOT / "tasks").glob("*.json")}
    print(report(latest(args.service, args.version), descs, args.measurement))


if __name__ == "__main__":
    main()
