#!/usr/bin/env python3
"""Freeze the agents' winning kernels into self-contained candidate sets for e2e eval.

Each set is a directory usable directly as ``FASTKERNELS_CANDIDATE_DIR``
(``L1/ L2/ L3/`` with ``<stem>.py`` plus CUDA/C++ sources). Sets:

* ``<agent>-ind``: winners of the independent campaign that are still correct in the
  composed bench of exactly that winner set (``comp/bench_results/bench.json``);
* ``<agent>-seq``: winners of the sequential campaign that are correct in its final
  composed bench (``seq/bench_results/bench.json``);
* ``selftest``: one trivial subclass of the baseline per target class (identical
  numerics; exercises only the candidate swap).

Build artifacts (compiled extensions, ninja files, __pycache__) are dropped so every
kernel JIT-compiles fresh on the machine that runs it. Each set gets a
``manifest.json`` (kernels, dropped kernels, file checksums, per-scenario kernel
usage, static-check findings); ``index.json`` summarizes all sets.

Usage::

    python3 scripts/freeze_candidate_sets.py            # writes agent-candidates/
    python3 scripts/freeze_candidate_sets.py --no-filter  # keep winners that broke when composed
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent
FK_ROOT = RESULTS_DIR.parent

# (set prefix, display name, run directory relative to FK_ROOT)
AGENTS = [
    ("drkernel", "Dr. Kernel", "drkernel-b200-20260919/campaign"),
    ("claude", "Claude Code", "claude-fk-runs"),
    ("kda", "KDA", "kda-fk-runs"),
    ("ako", "AKO", "ako4x-fk-runs"),
]
# set suffix -> (mode name, winners dir, composed bench of exactly those winners)
MODES = {
    "ind": ("independent", "ind/win_candidates", "comp/bench_results/bench.json"),
    "seq": ("sequential", "seq/win_candidates", "seq/bench_results/bench.json"),
}
SOURCES = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
EXCLUDED_STEMS = {(3, "oasis_rollout")}  # harness always skips it (not a kernel)
QUAL_RE = re.compile(r"\.L([1-4])\.([^:]+):(\w+)$")
ABS_PATH_RE = re.compile(r"(/checkpoint/|/code/users/|/home/\w+/|/Users/\w+/|/root/)")
# Third-party top-level modules provided by fastkernels' pinned runtime dependencies.
ALLOWED_THIRD_PARTY = {
    "torch", "triton", "numpy", "flashinfer", "vllm", "vllm_omni", "cutlass", "cuda",
    "deep_gemm", "flash_attn", "transformers", "quack", "safetensors", "einops",
    "fastkernels", "fla",
}
# Source-file names that kernels reference but that are not meant to ship with the set
# (manually verified): generated at runtime or located in an installed package.
EXTERNAL_SOURCES = {
    "kernel.cu": "written at runtime into the JIT build directory",
    "cutlass.h": "probed in the installed CUTLASS include directory",
    "packed_stride.hpp": "probed in the installed CUTLASS include directory",
    "trtllm_fused_moe_kernel_launcher.cu": "staged from the installed flashinfer package",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_targets(path: Path) -> dict[tuple[int, str], list[str]]:
    targets = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            level, stem, classes = line.split()[:3]
            if (int(level), stem) not in EXCLUDED_STEMS:
                targets[(int(level), stem)] = classes.split(",")
    return targets


def stem_results(bench: Path) -> dict[tuple[int, str], dict]:
    """Correct iff every non-skipped scenario PASSED; speedup = geomean over them."""
    rows = defaultdict(list)
    for row in json.loads(bench.read_text()).get("scenarios") or []:
        m = QUAL_RE.search(row.get("op") or "")
        if m:
            rows[(int(m.group(1)), m.group(2))].append(row)
    out = {}
    for key, rs in rows.items():
        timed = [r for r in rs if r.get("status") != "SKIPPED"]
        bad = sorted({r.get("status") for r in timed if r.get("status") != "PASSED"})
        ok = bool(timed) and not bad and all((r.get("speedup") or 0) > 0 for r in timed)
        speedup = None
        if ok:
            logs = [__import__("math").log(r["speedup"]) for r in timed]
            speedup = __import__("math").exp(sum(logs) / len(logs))
        out[key] = {"correct": ok, "speedup": speedup, "failed_statuses": bad}
    return out


def scenario_usage(captures: Path) -> dict[str, set[tuple[int, str]]]:
    """Model -> L1-L3 stems its captured forward pass instantiates."""
    usage: dict[str, set] = {}
    summary = json.loads((captures / "summary.json").read_text())
    for sc in summary["scenarios"]:
        model = sc["model"]
        stems = usage.setdefault(model, set())
        for rep in captures.glob(f"{sc['index']:02d}_*/report_*.json"):
            for qual in json.loads(rep.read_text()).get("operators", {}):
                m = QUAL_RE.search(qual)
                if m and int(m.group(1)) <= 3:
                    stems.add((int(m.group(1)), m.group(2)))
    return usage


def module_exists(fk_repo: Path, dotted: str) -> bool:
    base = fk_repo.joinpath(*dotted.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def static_checks(set_dir: Path, fk_repo: Path, targets: dict) -> tuple[list[str], list[str]]:
    """Findings that would break the set on another machine (no GPU needed), plus
    optional imports (guarded by try/except, so the kernel falls back if missing)."""
    findings, optional = [], []
    stdlib = set(sys.stdlib_module_names)
    for py in sorted(set_dir.glob("L[1-3]/*.py")):
        rel = py.relative_to(set_dir)
        level = int(py.parent.name[1])
        src = py.read_text(errors="replace")
        if ABS_PATH_RE.search(src):
            findings.append(f"{rel}: absolute path {ABS_PATH_RE.search(src).group(0)!r}")
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            findings.append(f"{rel}: syntax error on local Python {sys.version_info[:2]}: {exc}")
            continue
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        classes |= {t.id for n in tree.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}  # e.g. RMSNorm = ModelNew
        guarded = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try)
                   for b in t.body for n in ast.walk(b)}
        for cls in targets.get((level, py.stem), []):
            if cls not in classes:
                findings.append(f"{rel}: does not define baseline class {cls}")
        pkg = ["fastkernels", "tasks", "candidate", py.parent.name]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                parts = pkg[: len(pkg) - (node.level - 1)] + (node.module or "").split(".")
                dotted = ".".join(p for p in parts if p)
                if dotted.startswith("fastkernels.tasks.candidate."):
                    lvl_stem = dotted.split(".")[3:]
                    stems = ([lvl_stem] if len(lvl_stem) == 2 else
                             [[lvl_stem[0], a.name] for a in node.names] if len(lvl_stem) == 1 else [])
                    ok = bool(stems) and all(
                        (set_dir / lv / f"{st}.py").is_file()
                        or module_exists(fk_repo, f"fastkernels.tasks.baseline.{lv}.{st}")
                        for lv, st in stems)
                else:
                    ok = module_exists(fk_repo, dotted) or (fk_repo.joinpath(*dotted.split(".")) / "__init__.py").is_file()
                if not ok:
                    findings.append(f"{rel}:{node.lineno}: unresolved relative import {dotted}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    top = name.split(".")[0]
                    if top == "fastkernels" and not module_exists(fk_repo, name):
                        findings.append(f"{rel}:{node.lineno}: missing module {name}")
                    elif top and top not in stdlib and top not in ALLOWED_THIRD_PARTY:
                        if id(node) in guarded:
                            optional.append(f"{rel}:{node.lineno}: {name}")
                        else:
                            findings.append(f"{rel}:{node.lineno}: import outside pinned deps: {name}")
        for const in ast.walk(tree):
            if (isinstance(const, ast.Constant) and isinstance(const.value, str)
                    and Path(const.value).suffix in SOURCES and "\n" not in const.value
                    and "/" not in const.value and " " not in const.value):
                if not (py.parent / const.value).is_file() and const.value not in EXTERNAL_SOURCES:
                    findings.append(f"{rel}: references missing source file {const.value}")
    return findings, optional


def copy_set_files(src_root: Path, dest: Path, keep: set[tuple[int, str]]) -> None:
    for level_dir in sorted(src_root.glob("L[1-3]")):
        level = int(level_dir.name[1])
        for path in sorted(level_dir.iterdir()):  # top level only: subdirs are build dirs
            if not path.is_file():
                continue
            if path.suffix == ".py" and (level, path.stem) in keep:
                pass
            elif path.suffix.lower() not in SOURCES:
                continue
            out = dest / level_dir.name / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def unreferenced_sources(set_dir: Path) -> list[str]:
    texts = {lvl: " ".join(p.read_text(errors="replace") for p in set_dir.glob(f"{lvl}/*"))
             for lvl in ("L1", "L2", "L3")}
    return sorted(str(p.relative_to(set_dir)) for p in set_dir.glob("L[1-3]/*")
                  if p.suffix.lower() in SOURCES and p.name not in texts[p.parent.name])


def write_selftest(dest: Path, targets: dict) -> None:
    for (level, stem), classes in sorted(targets.items()):
        mod = f"fastkernels.tasks.baseline.L{level}.{stem}"
        lines = ['"""Self-test candidate: subclasses the baseline unchanged (swap only)."""', ""]
        lines += [f"from {mod} import {c} as _Base{c}" for c in classes] + [""]
        for c in classes:
            lines += ["", f"class {c}(_Base{c}):", "    pass", ""]
        out = dest / f"L{level}" / f"{stem}.py"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))


def finalize(set_dir: Path, meta: dict, usage: dict, fk_repo: Path, targets: dict) -> dict:
    kernels = {(k["level"], k["stem"]) for k in meta["kernels"]}
    meta["files"] = {str(p.relative_to(set_dir)): sha256(p)
                     for p in sorted(set_dir.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    meta["unreferenced_sources"] = unreferenced_sources(set_dir)
    meta["scenarios"] = {model: sorted(f"L{l}:{s}" for l, s in stems & kernels)
                         for model, stems in usage.items()}
    meta["static_check_findings"], meta["optional_imports"] = static_checks(set_dir, fk_repo, targets)
    (set_dir / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n")
    return {
        "kernels_per_level": {f"L{l}": sum(1 for k in kernels if k[0] == l) for l in (1, 2, 3)},
        "dropped": len(meta.get("dropped", [])),
        "static_check_findings": len(meta["static_check_findings"]),
        "manifest_sha256": sha256(set_dir / "manifest.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "agent-candidates")
    ap.add_argument("--runs-root", type=Path, default=FK_ROOT)
    ap.add_argument("--fastkernels", type=Path, default=FK_ROOT / "fastkernels")
    ap.add_argument("--captures", type=Path, default=RESULTS_DIR / "captures" / "default" / "b200")
    ap.add_argument("--targets", type=Path,
                    default=FK_ROOT / "fastkernels-iclr" / "figures" / "data" / "kernel_targets_default_b200.txt")
    ap.add_argument("--no-filter", action="store_true",
                    help="keep winners that failed their set's composed bench")
    args = ap.parse_args()

    targets = load_targets(args.targets)
    usage = scenario_usage(args.captures)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    shutil.copy2(args.targets, args.out / "targets_default_b200.txt")
    fk_commit = subprocess.run(["git", "-C", str(args.fastkernels), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    index = {"fastkernels_commit": fk_commit, "captures": "captures/default/b200",
             "filtered_by_composed_bench": not args.no_filter, "sets": {}}

    for prefix, agent, run_dir in AGENTS:
        for suffix, (mode, win_rel, bench_rel) in MODES.items():
            name = f"{prefix}-{suffix}"
            src = args.runs_root / run_dir / win_rel
            bench = stem_results(args.runs_root / run_dir / bench_rel)
            winners = {(int(p.parent.name[1]), p.stem) for p in src.glob("L[1-3]/*.py")}
            winners -= EXCLUDED_STEMS
            keep, dropped = set(), []
            for key in sorted(winners):
                res = bench.get(key)
                if res and res["correct"]:
                    keep.add(key)
                elif args.no_filter:
                    keep.add(key)
                else:
                    why = (f"failed composed bench: {', '.join(res['failed_statuses'])}"
                           if res else "not in composed bench")
                    dropped.append({"level": key[0], "stem": key[1], "reason": why})
            set_dir = args.out / name
            copy_set_files(src, set_dir, keep)
            meta = {
                "set": name, "agent": agent, "mode": mode,
                "source": {"winners": f"{run_dir}/{win_rel}", "composed_bench": f"{run_dir}/{bench_rel}"},
                "selection": ("winners correct in the composed bench of this set"
                              if not args.no_filter else "all winners (unfiltered)"),
                "kernels": [{"level": l, "stem": s, "classes": targets.get((l, s), []),
                             "composed_speedup": (bench.get((l, s)) or {}).get("speedup")}
                            for l, s in sorted(keep)],
                "dropped": dropped,
            }
            index["sets"][name] = finalize(set_dir, meta, usage, args.fastkernels, targets)

    set_dir = args.out / "selftest"
    write_selftest(set_dir, targets)
    meta = {"set": "selftest", "agent": None, "mode": "selftest",
            "selection": "trivial subclass of every target baseline class",
            "kernels": [{"level": l, "stem": s, "classes": c} for (l, s), c in sorted(targets.items())]}
    index["sets"]["selftest"] = finalize(set_dir, meta, usage, args.fastkernels, targets)
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")

    print(f"{'set':14} {'L1':>4} {'L2':>4} {'L3':>4} {'dropped':>8} {'findings':>9}")
    for name, s in index["sets"].items():
        k = s["kernels_per_level"]
        print(f"{name:14} {k['L1']:>4} {k['L2']:>4} {k['L3']:>4} {s['dropped']:>8} "
              f"{s['static_check_findings']:>9}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
