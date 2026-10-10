"""Report over the frozen forecasts in results/, per service.

For every (service, task) the latest result is taken (optionally only those of
one service version, e.g. a Krauncher calibration_id) and compared with the
task's description (tasks/<source>/<task>.json). Levels:

0. forecast time — wall clock from request to the normalised forecast,
   measured by the stand for every service (plus the service's own parts);
1. classification / assay fields — match per field;
2. reference-card time — forecast / reference per task: centre (geomean),
   typical deviation (median |log|), share inside the issued spread;
3. VRAM — forecast / measured peak; forecasts below the peak (an OOM risk);
   forecasts above a measured upper bound (`vram_gb_at_most`);
4. ladder — compute_ratio per GPU: error (median and p90 of |log|), card order
   (Spearman), time regret of the GPU the forecast calls fastest; measured
   GPUs the service gave no forecast for.

Levels 2-4 are reported per measurement set of the descriptions. Every value
gets a status — good (accurate), warning, critical — by the thresholds in
scoring.py. Output: report.json (machine-readable), report.md, report.html
(for viewing) in --out.

    python report.py [--service krauncher] [--version <id>] [--measurement <id>] [--out DIR]
"""

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import scoring
from scoring import (MEASURED, REFERENCE_GPU, TIME_GOOD, anchor_time, ladder_pairs, matches, ratio_status,
                     time_status, vram_status)

ROOT = Path(__file__).parent


# --------------------------------------------------------------------- data

def latest(service: str | None, version: str | None) -> dict[tuple[str, str], dict]:
    """(service, task) -> its latest result (results/<service>/...)."""
    out: dict[tuple[str, str], dict] = {}
    for f in sorted((ROOT / "results").glob("*/**/*.json")):
        r = json.loads(f.read_text())
        if "service" not in r or (service and r["service"] != service):
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


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def _typ(ratios):
    return _r(math.exp(statistics.median(abs(math.log(x)) for x in ratios)))


def analyze(results: dict, descs: dict, measurement: str | None = None) -> dict:
    """Everything the report shows, as data."""
    mids = [measurement] if measurement else list(dict.fromkeys(
        m["id"] for d in descs.values() for m in d.get("measurements", [])))
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {
            "time_good": scoring.TIME_GOOD, "time_warning": scoring.TIME_WARN,
            "ratio_good": scoring.RATIO_GOOD, "ratio_warning": scoring.RATIO_WARN,
            "vram_over_warning": scoring.VRAM_OVER_WARN, "params_tolerance": scoring.PARAMS_TOL,
        },
        "services": [],
    }
    for service in sorted({s for s, _ in results}):
        rs = {t: r for (s, t), r in results.items() if s == service and t in descs}
        svc = {"service": service, "tasks": sorted(rs),
               "versions": sorted({json.dumps(r["version"], sort_keys=True) for r in rs.values()}),
               "result_created_utc": {t: rs[t]["created_utc"] for t in sorted(rs)}}

        # 0. forecast time
        timed = {t: rs[t]["timing"] for t in sorted(rs) if (rs[t].get("timing") or {}).get("forecast_sec")}
        ts = [v["forecast_sec"] for v in timed.values()]
        svc["forecast_time"] = {
            "median_sec": _r(statistics.median(ts), 2) if ts else None,
            "p90_sec": _r(_q(ts, 0.9), 2) if ts else None,
            "max_sec": _r(max(ts), 2) if ts else None,
            "tasks": timed,
        }

        # 1. classification / assay fields
        fields: dict[str, dict] = {}
        for t in sorted(rs):
            exp, got = descs[t]["fields"], rs[t]["fields"]
            for f, w in exp.items():
                if f in MEASURED or f not in got:
                    continue
                ok = matches(f, w, got[f])
                fields.setdefault(f, {"tasks": {}})["tasks"][t] = {
                    "expected": w, "forecast": got[f], "status": "good" if ok else "critical"}
        for v in fields.values():
            v["match"] = sum(1 for x in v["tasks"].values() if x["status"] == "good")
            v["total"] = len(v["tasks"])
        order = ("workload_type", "mode", "framework", "precision", "params_billions", "batch_size",
                 "epochs", "dataset_samples", "seq_len", "cpu_only")
        svc["fields"] = {f: fields[f] for f in sorted(fields, key=lambda f: (order.index(f) if f in order else len(order), f))}

        # 2-4. per measurement set
        svc["measurements"] = []
        for mid in mids:
            def meas(t, mid=mid):
                return next((m for m in descs[t].get("measurements", []) if m["id"] == mid), {})
            have = [t for t in sorted(rs) if meas(t)]
            if not have:
                continue
            ms = {"id": mid, "independent": sorted({bool(meas(t).get("independent")) for t in have}),
                  "tasks": have}

            # 2. reference-card time
            rows = []
            for t in have:
                m, g = meas(t).get("reference_sec"), rs[t]["fields"].get("reference_sec")
                if m and g:
                    sp = rs[t]["fields"].get("spread_factor") or 1.0
                    rows.append({"task": t, "measured": m, "forecast": g, "ratio": _r(g / m),
                                 "spread_factor": sp, "inside_spread": 1 / sp <= m / g <= sp,
                                 "status": time_status(g, m)})
            if rows:
                ms["reference_time"] = {"centre": _r(_gm([x["ratio"] for x in rows])),
                                        "typical_deviation": _typ([x["ratio"] for x in rows]),
                                        "inside_spread": sum(x["inside_spread"] for x in rows),
                                        "n": len(rows), "tasks": rows}

            # 2b. compute time on the anchor GPU (work / published throughput)
            rows = []
            for t in have:
                at = anchor_time(meas(t), rs[t]["fields"])
                if at:
                    rows.append({"task": t, "gpu": meas(t).get("anchor_gpu") or REFERENCE_GPU,
                                 "measured": _r(at[0], 1), "forecast": _r(at[1], 1),
                                 "ratio": _r(at[1] / at[0]), "status": time_status(at[1], at[0])})
            if rows:
                ms["anchor_time"] = {"centre": _r(_gm([x["ratio"] for x in rows])),
                                     "typical_deviation": _typ([x["ratio"] for x in rows]),
                                     "good": sum(1 for x in rows if x["status"] == "good"),
                                     "n": len(rows), "tasks": rows}

            # 3. VRAM: measured peak, and measured upper bounds
            rows = []
            for t in have:
                m, g = meas(t).get("min_vram_gb"), rs[t]["fields"].get("min_vram_gb")
                if m and g:
                    rows.append({"task": t, "measured": m, "forecast": g, "ratio": _r(g / m),
                                 "status": vram_status(g, m)})
            if rows:
                ms["vram"] = {"centre": _r(_gm([x["ratio"] for x in rows])),
                              "typical_deviation": _typ([x["ratio"] for x in rows]),
                              "below_peak": sum(1 for x in rows if x["forecast"] < x["measured"]),
                              "n": len(rows), "tasks": rows}
            bounds = []
            for t in have:
                b, g = meas(t).get("vram_gb_at_most"), rs[t]["fields"].get("min_vram_gb")
                if b and g is not None:
                    bounds.append({"task": t, "at_most": b, "forecast": g,
                                   "status": "critical" if g > b else "good"})
            if bounds:
                ms["vram_bound"] = {"above": sum(1 for x in bounds if x["status"] == "critical"),
                                    "n": len(bounds), "tasks": bounds}

            # 4. ladder
            lrows, cells, all_err, uncovered, anchors = [], {}, [], [], {}
            for t in have:
                full = ladder_pairs(meas(t), rs[t]["fields"].get("compute_ratio"))
                if not full:
                    continue
                if meas(t).get("anchor_gpu"):
                    anchors[t] = meas(t)["anchor_gpu"]
                # the anchor (or the reference card) is 1.0 on both sides by
                # definition: kept for the card order and the pick, not scored
                anchor = meas(t).get("anchor_gpu") or REFERENCE_GPU
                cells[t] = {g: {"measured": e, "forecast": _r(f),
                                "status": "anchor" if g == anchor else
                                          ratio_status(f, e) if f is not None else "none"}
                            for g, (e, f) in full.items()}
                scored = [g for g in full if g != anchor]
                missing = [g for g in scored if full[g][1] is None]
                if missing:
                    uncovered.append({"task": t, "missing": missing, "measured": len(scored)})
                lp = {g: v for g, v in full.items() if v[1] is not None}
                if len(lp) < 2:
                    continue
                exp, got = {g: v[0] for g, v in lp.items()}, {g: v[1] for g, v in lp.items()}
                err = [abs(math.log(got[g] / exp[g])) for g in lp if g != anchor]
                all_err += err
                pick, best = min(lp, key=lambda g: got[g]), min(lp, key=lambda g: exp[g])
                regret = exp[pick] / exp[best]
                rank = _spearman(list(exp.values()), list(got.values())) if len(lp) >= 3 else None
                lrows.append({"task": t, "gpus": len(lp), "scored_gpus": len(err),
                              "typical_error": _r(math.exp(statistics.median(err))),
                              "p90_error": _r(math.exp(_q(err, 0.9))),
                              "rank_corr": _r(rank, 2) if rank is not None else None, "pick": pick, "fastest": best,
                              "time_regret": _r(regret),
                              "status": "critical" if regret > 1.0001 else
                                        "warning" if rank is not None and rank < 0.7 else "good"})
            if cells:
                ms["ladder"] = {
                    "pairs": len(all_err),
                    "typical_error": _r(math.exp(statistics.median(all_err))) if all_err else None,
                    "p90_error": _r(math.exp(_q(all_err, 0.9))) if all_err else None,
                    "median_rank_corr": _r(statistics.median(ranks), 2) if (ranks := [x["rank_corr"] for x in lrows
                                                                                  if x["rank_corr"] is not None]) else None,
                    "pick_slower": sum(1 for x in lrows if x["time_regret"] > 1.0001),
                    "n": len(lrows), "tasks": lrows, "cells": cells, "uncovered": uncovered,
                    "anchors": anchors}
            svc["measurements"].append(ms)
        out["services"].append(svc)
    return out


# --------------------------------------------------------------------- markdown

def render_md(rep: dict) -> str:
    L = []
    for s in rep["services"]:
        L += [f"## {s['service']}", "", f"{len(s['tasks'])} tasks; versions: " + "; ".join(s["versions"]), ""]
        ft = s["forecast_time"]
        if ft["median_sec"] is not None:
            L += ["### Forecast time", "", f"median {ft['median_sec']} s, p90 {ft['p90_sec']} s, max {ft['max_sec']} s", ""]
        L += ["### Classification / assay fields", "", "| field | match | differs in |", "|---|---|---|"]
        for f, v in s["fields"].items():
            bad = [t for t, x in v["tasks"].items() if x["status"] != "good"]
            L.append(f"| {f} | {v['match']}/{v['total']} | {', '.join(bad) or '-'} |")
        L.append("")
        for m in s["measurements"]:
            L += [f"### Measurements: {m['id']} (independent: {', '.join(map(str, m['independent']))}; "
                  f"{len(m['tasks'])} tasks)", ""]
            if "anchor_time" in m:
                x = m["anchor_time"]
                L += ["#### Compute time on the anchor GPU", "",
                      f"centre x{x['centre']}, typical deviation x{x['typical_deviation']}, "
                      f"within x{TIME_GOOD}: {x['good']}/{x['n']}", "",
                      "| task | GPU | measured, s | forecast, s | forecast / measured | status |", "|---|---|---|---|---|---|"]
                L += [f"| {r['task']} | {r['gpu']} | {r['measured']} | {r['forecast']} | x{r['ratio']} | {r['status']} |"
                      for r in x["tasks"]]
                L.append("")
            for key, title, unit in (("reference_time", "Reference-card time", "s"), ("vram", "VRAM", "GB")):
                if key not in m:
                    continue
                x = m[key]
                extra = (f"inside the issued spread {x['inside_spread']}/{x['n']}" if key == "reference_time"
                         else f"below the measured peak {x['below_peak']}/{x['n']}")
                L += [f"#### {title}", "", f"centre x{x['centre']}, typical deviation x{x['typical_deviation']}, {extra}", "",
                      f"| task | measured, {unit} | forecast, {unit} | forecast / measured | status |", "|---|---|---|---|---|"]
                L += [f"| {r['task']} | {r['measured']} | {r['forecast']} | x{r['ratio']} | {r['status']} |" for r in x["tasks"]]
                L.append("")
            if "vram_bound" in m:
                vb = m["vram_bound"]
                L += ["#### VRAM upper bound", "", f"forecast above the measured bound in {vb['above']}/{vb['n']} tasks"
                      + "".join(f"; {x['task']}: {x['forecast']} GB > {x['at_most']} GB"
                                for x in vb["tasks"] if x["status"] == "critical"), ""]
            if "ladder" in m:
                ld = m["ladder"]
                if ld["tasks"]:
                    L += ["#### Ladder (compute_ratio)", "",
                          f"{ld['pairs']} (task, GPU) pairs: typical error x{ld['typical_error']}, p90 x{ld['p90_error']}; "
                          f"median rank correlation {ld['median_rank_corr']}; "
                          f"pick slower than the fastest in {ld['pick_slower']}/{ld['n']} tasks", "",
                          "| task | GPUs | typical error | p90 | rank corr | picked | fastest | time regret | status |",
                          "|---|---|---|---|---|---|---|---|---|"]
                    L += [f"| {x['task']} | {x['gpus']} | x{x['typical_error']} | x{x['p90_error']} | {x['rank_corr']} | "
                          f"{x['pick']} | {x['fastest']} | x{x['time_regret']} | {x['status']} |" for x in ld["tasks"]]
                    L.append("")
                if ld["uncovered"]:
                    L += ["#### Ladder coverage", "", "measured GPUs without a forecast: " + "; ".join(
                        f"{x['task']} {len(x['missing'])}/{x['measured']}" for x in ld["uncovered"]), ""]
    return "\n".join(L)


# --------------------------------------------------------------------- html

_ICON = {"good": "✓", "warning": "!", "critical": "✗", "none": "–", "anchor": "◦"}
_WORD = {"good": "accurate", "warning": "miss", "critical": "problem", "none": "no forecast",
         "anchor": "anchor (1.00 by definition, not scored)"}

# Status colours: the reference palette's fixed status set (good / warning /
# critical), always paired with an icon and a word — never colour alone.
_CSS = """
.viz-root{color-scheme:light;--surface-0:#f3f3f1;--surface-1:#fcfcfb;--border:#dedcd6;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#6f6d67;
--good:#0ca30c;--warning-ink:#8a5d00;--critical:#d03b3b;
--good-bg:rgba(12,163,12,.13);--warning-bg:rgba(250,178,25,.24);--critical-bg:rgba(208,59,59,.15);--none-bg:rgba(119,117,111,.10)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9a988f;
--warning-ink:#fab219;--good-bg:rgba(12,163,12,.24);--warning-bg:rgba(250,178,25,.20);--critical-bg:rgba(208,59,59,.28);--none-bg:rgba(195,194,183,.10)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#34332f;
--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9a988f;
--warning-ink:#fab219;--good-bg:rgba(12,163,12,.24);--warning-bg:rgba(250,178,25,.20);--critical-bg:rgba(208,59,59,.28);--none-bg:rgba(195,194,183,.10)}
html,body{margin:0;background:#f3f3f1}
@media (prefers-color-scheme:dark){html:where(:not([data-theme="light"])),html:where(:not([data-theme="light"])) body{background:#121211}}
html[data-theme="dark"],html[data-theme="dark"] body{background:#121211}
.viz-root{background:var(--surface-0);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:24px 16px 48px;min-height:100vh;box-sizing:border-box}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px} h3{font-size:15px;margin:28px 0 6px}
.sub{color:var(--text-secondary);margin:0 0 12px} .muted{color:var(--text-muted);font-weight:400}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin:12px 0 4px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--text-secondary);font-size:12px} .tile .v{font-size:24px;font-weight:600;margin:2px 0}
.tile .n{color:var(--text-muted);font-size:12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.panel{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--text-secondary);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--surface-1)}
tr:last-child td{border-bottom:0} td.num{text-align:right}
.st{display:inline-flex;align-items:center;gap:5px;padding:1px 8px;border-radius:999px;font-size:12px;color:var(--text-primary)}
.st.good{background:var(--good-bg)} .st.warning{background:var(--warning-bg)} .st.critical{background:var(--critical-bg)} .st.none{background:var(--none-bg)}
.st b{font-weight:700} .st.good b{color:var(--good)} .st.warning b{color:var(--warning-ink)} .st.critical b{color:var(--critical)} .st.none b{color:var(--text-muted)}
td.cell{text-align:center;font-size:12px;min-width:56px;padding:7px 8px}
td.cell.good{background:var(--good-bg)} td.cell.warning{background:var(--warning-bg)}
td.cell.critical{background:var(--critical-bg)} td.cell.none{background:var(--none-bg);color:var(--text-muted)}
td.cell.anchor{color:var(--text-muted)} .st.anchor{background:var(--none-bg)} .st.anchor b{color:var(--text-muted)}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 0}
.note{color:var(--text-secondary);font-size:13px;margin:6px 0}
.toggle{float:right;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);border-radius:8px;padding:4px 10px;cursor:pointer;font:inherit}
"""


def _st(status: str, text: str | None = None) -> str:
    return f'<span class="st {status}"><b>{_ICON[status]}</b>{html.escape(text or _WORD[status])}</span>'


def _tile(k: str, v: str, n: str = "", status: str | None = None) -> str:
    badge = _st(status) if status else ""
    return (f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
            f'<div class="n">{badge}<span>{html.escape(n)}</span></div></div>')


def _tbl(head: list[str], rows: list[list[str]], num: frozenset = frozenset()) -> str:
    h = "".join(f'<th class="{"num" if i in num else ""}">{html.escape(c)}</th>' for i, c in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f'<div class="panel"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _x(v) -> str:
    return "–" if v is None else f"x{v:.2f}"


def _band(v: float, good: float, warn: float) -> str:
    return "good" if v <= good else "warning" if v <= warn else "critical"


def _ind(flags: list[bool]) -> str:
    return "independent" if flags == [True] else "not independent — service calibration data" if flags == [False] else "mixed"


def render_html(rep: dict) -> str:
    th, e = rep["thresholds"], html.escape
    P = ["<h1>Forecast quality report</h1>",
         f'<p class="sub">Generated {e(rep["created_utc"])}. Each value is marked '
         f'{_st("good")} time within x{th["time_good"]}, ladder ratio within '
         f'x{th["ratio_good"]}, VRAM up to x{th["vram_over_warning"]} above the measured peak · '
         f'{_st("warning")} time within x{th["time_warning"]}, ratio within x{th["ratio_warning"]}, VRAM over-provisioned · '
         f'{_st("critical")} beyond that, VRAM below the peak (OOM risk) or above a measured bound, '
         f'a field that differs, a slower GPU picked as the fastest.</p>']
    for s in rep["services"]:
        P.append(f"<h2>{e(s['service'])}</h2>")
        vers = [" · ".join(f"{k} {v}" for k, v in json.loads(v).items()) for v in s["versions"]]
        P.append(f'<p class="sub">{len(s["tasks"])} tasks · ' + " | ".join(e(v) for v in vers) + "</p>")
        ft = s["forecast_time"]
        fm = sum(v["match"] for v in s["fields"].values())
        ftot = sum(v["total"] for v in s["fields"].values())
        tiles = []
        if ft["median_sec"] is not None:
            tiles.append(_tile("Forecast time, median", f"{ft['median_sec']} s", f"p90 {ft['p90_sec']} s · max {ft['max_sec']} s"))
        if ftot:
            tiles.append(_tile("Classification fields", f"{fm}/{ftot}", "match the description",
                               "good" if fm == ftot else "critical"))
        for m in s["measurements"]:
            tag = f"{m['id']}"
            if "reference_time" in m:
                rt = m["reference_time"]
                tiles.append(_tile("Time, typical deviation", _x(rt["typical_deviation"]),
                                   f"{tag} · in spread {rt['inside_spread']}/{rt['n']}",
                                   _band(rt["typical_deviation"], th["time_good"], th["time_warning"])))
            if "anchor_time" in m:
                at = m["anchor_time"]
                tiles.append(_tile("Compute time, typical deviation", _x(at["typical_deviation"]),
                                   f"{tag} · centre {_x(at['centre'])} · {at['n']} tasks",
                                   _band(at["typical_deviation"], th["time_good"], th["time_warning"])))
            if "vram" in m:
                v = m["vram"]
                tiles.append(_tile("VRAM below the measured peak", f"{v['below_peak']}/{v['n']}",
                                   f"{tag} · centre {_x(v['centre'])}", "good" if v["below_peak"] == 0 else "critical"))
            if "vram_bound" in m:
                vb = m["vram_bound"]
                tiles.append(_tile("VRAM above a measured bound", f"{vb['above']}/{vb['n']}", tag,
                                   "good" if vb["above"] == 0 else "critical"))
            if "ladder" in m:
                ld = m["ladder"]
                if ld["n"]:
                    tiles.append(_tile("Ladder, typical error", _x(ld["typical_error"]),
                                       f"{tag} · p90 {_x(ld['p90_error'])} · {ld['pairs']} pairs",
                                       _band(ld["typical_error"], th["ratio_good"], th["ratio_warning"])))
                    tiles.append(_tile("Fastest GPU picked right", f"{ld['n'] - ld['pick_slower']}/{ld['n']}",
                                       f"{tag} · rank corr {ld['median_rank_corr']}",
                                       "good" if ld["pick_slower"] == 0 else "critical"))
                if ld["uncovered"]:
                    miss = sum(len(x["missing"]) for x in ld["uncovered"])
                    tot = sum(sum(1 for x in c.values() if x["status"] != "anchor") for c in ld["cells"].values())
                    tiles.append(_tile("Measured GPUs with no forecast", f"{miss}/{tot}", tag, "critical"))
        P.append('<div class="tiles">' + "".join(tiles) + "</div>")

        # classification: tasks x fields
        fnames = list(s["fields"])
        rows = []
        for t in s["tasks"]:
            row = [e(t)]
            for f in fnames:
                x = s["fields"][f]["tasks"].get(t)
                if x is None:
                    row.append('<span class="muted">–</span>')
                    continue
                tip = f"expected {x['expected']} · forecast {x['forecast']}"
                shown = ("fp32" if f == "precision" else "not read") if x["forecast"] is None else x["forecast"]
                txt = str(shown) if x["status"] == "good" else f"{shown} ≠ {x['expected']}"
                row.append(f'<span title="{e(tip)}">{_st(x["status"], txt)}</span>')
            rows.append(row)
        P.append("<h3>Classification / assay fields</h3>")
        P.append('<p class="note">Expected values are read from the task code and stored in the task description. '
                 'Hover a cell for expected / forecast.</p>')
        P.append(_tbl(["task"] + fnames, rows))

        for m in s["measurements"]:
            P.append(f"<h3>Measurements: {e(m['id'])} <span class='muted'>· {_ind(m['independent'])} · "
                     f"{len(m['tasks'])} tasks</span></h3>")
            if "reference_time" in m:
                rt = m["reference_time"]
                P.append(f'<p class="note">Time on the reference card: centre {_x(rt["centre"])}, typical deviation '
                         f'{_x(rt["typical_deviation"])}, inside the issued spread {rt["inside_spread"]}/{rt["n"]}.</p>')
                P.append(_tbl(["task", "measured, s", "forecast, s", "forecast / measured", "status",
                               "issued spread", "measured inside it"],
                              [[e(x["task"]), str(x["measured"]), str(x["forecast"]), _x(x["ratio"]), _st(x["status"]),
                                f"x{x['spread_factor']:.3g}",
                                _st("good", "yes") if x["inside_spread"] else _st("critical", "no")]
                               for x in rt["tasks"]], frozenset({1, 2, 3, 5})))
            if "anchor_time" in m:
                at = m["anchor_time"]
                P.append(f'<p class="note">Compute time on the anchor GPU (measured: the work of the task / the '
                         f'published throughput; forecast: compute time on the reference card x the forecast ratio '
                         f'of the anchor): centre {_x(at["centre"])}, typical deviation '
                         f'{_x(at["typical_deviation"])}, within x{th["time_good"]}: {at["good"]}/{at["n"]}.</p>')
                P.append(_tbl(["task", "GPU", "measured, s", "forecast, s", "forecast / measured", "status"],
                              [[e(x["task"]), e(x["gpu"]), str(x["measured"]), str(x["forecast"]), _x(x["ratio"]),
                                _st(x["status"])] for x in at["tasks"]], frozenset({2, 3, 4})))
            if "vram" in m or "vram_bound" in m:
                rows = [[e(x["task"]), str(x["measured"]), str(x["forecast"]), _x(x["ratio"]),
                         _st(x["status"], "below the peak" if x["status"] == "critical"
                             else "over-provisioned" if x["status"] == "warning" else None)]
                        for x in m.get("vram", {}).get("tasks", [])]
                rows += [[e(x["task"]), f"≤ {x['at_most']}", str(x["forecast"]), "–",
                          _st(x["status"], "above the bound" if x["status"] == "critical" else "within the bound")]
                         for x in m.get("vram_bound", {}).get("tasks", [])]
                P.append('<p class="note">VRAM, GB: the measured peak, or an upper bound (the run completed on a card of '
                         'that size), against the forecast requirement.</p>')
                P.append(_tbl(["task", "measured", "forecast", "forecast / measured", "status"], rows, frozenset({1, 2, 3})))
            if "ladder" in m:
                ld = m["ladder"]
                if ld["tasks"]:
                    P.append(f'<p class="note">Ladder: {ld["pairs"]} (task, GPU) pairs, typical error {_x(ld["typical_error"])}, '
                             f'p90 {_x(ld["p90_error"])}, median rank correlation {ld["median_rank_corr"]}.</p>')
                    P.append(_tbl(["task", "GPUs", "typical error", "p90", "rank corr", "picked as fastest",
                                   "really fastest", "time regret", "status"],
                                  [[e(x["task"]), str(x["gpus"]), _x(x["typical_error"]), _x(x["p90_error"]),
                                    str(x["rank_corr"]), e(x["pick"]), e(x["fastest"]), _x(x["time_regret"]),
                                    _st(x["status"], "slower pick" if x["status"] == "critical"
                                        else "order off" if x["status"] == "warning" else "right pick")]
                                   for x in ld["tasks"]], frozenset({1, 2, 3, 4, 7})))
                gpus = sorted({g for c in ld["cells"].values() for g in c})
                body = ""
                for t, c in ld["cells"].items():
                    tds = []
                    for g in gpus:
                        x = c.get(g)
                        if x is None:
                            tds.append("<td></td>")
                        elif x["status"] == "anchor":
                            tds.append(f'<td class="cell anchor" title="{e(g)}: anchor, 1.00 by definition">◦ anchor</td>')
                        elif x["forecast"] is None:
                            tip = f"{g}: measured {x['measured']} · no forecast"
                            tds.append(f'<td class="cell none" title="{e(tip)}">– none</td>')
                        else:
                            r = x["forecast"] / x["measured"]
                            tip = f"{g}: measured {x['measured']} · forecast {x['forecast']}"
                            tds.append(f'<td class="cell {x["status"]}" title="{e(tip)}">{_ICON[x["status"]]} {r:.2f}</td>')
                    anchor = ld["anchors"].get(t)
                    label = e(t) + (f' <span class="muted">vs {e(anchor)}</span>' if anchor else "")
                    body += f"<tr><td>{label}</td>{''.join(tds)}</tr>"
                head = "".join(f"<th>{e(g)}</th>" for g in gpus)
                P.append('<p class="note">Ladder per GPU: forecast / measured compute_ratio (1.00 = exact; hover for both '
                         'values). An empty cell is a GPU not measured for that task; "vs" names the anchor GPU a '
                         'published measurement is relative to.</p>')
                P.append(f'<div class="panel"><table><thead><tr><th>task</th>{head}</tr></thead><tbody>{body}</tbody></table></div>')
                P.append('<div class="legend">' + " ".join(_st(k) for k in ("good", "warning", "critical", "none", "anchor")) + "</div>")
                if ld["uncovered"]:
                    P.append('<p class="note">' + _st("critical", "no forecast")
                             + " measured GPUs the service listed no row for (e.g. filtered out by its VRAM requirement): "
                             + "; ".join(f"{e(x['task'])} {len(x['missing'])}/{x['measured']}" for x in ld["uncovered"]) + "</p>")
    toggle = ('<button class="toggle" type="button" onclick="var r=document.documentElement;'
              "var d=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;"
              "r.dataset.theme=d?'light':'dark'\">Light / dark</button>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Forecast quality report</title><style>{_CSS}</style></head>"
            f'<body><div class="viz-root"><div class="wrap">{toggle}' + "\n".join(P) + "</div></div></body></html>\n")


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service")
    ap.add_argument("--version", help="only results of this service version (e.g. a calibration_id)")
    ap.add_argument("--measurement", help="only this measurement set of the descriptions (default: each)")
    ap.add_argument("--out", default=str(ROOT / "results" / "report"), help="output directory")
    args = ap.parse_args()
    tasks = ROOT / "tasks"
    descs = {str(p.relative_to(tasks).with_suffix("")): json.loads(p.read_text()) for p in tasks.rglob("*.json")}
    rep = analyze(latest(args.service, args.version), descs, args.measurement)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(render_md(rep) + "\n")
    (out / "report.html").write_text(render_html(rep))
    print(f"{out}: report.json, report.md, report.html")


if __name__ == "__main__":
    main()
