"""One e2e probe on Llama-3.1-8B (runs inside a Modal container, see fk_modal.py::probe2).

Builds the compiled baseline engine, then either a candidate engine (the set in
FASTKERNELS_CANDIDATE_DIR swapped in) or -- for the "noise" reference -- the baseline
again after clearing every compile cache. Prints one JSON line prefixed with RESULT.
"""

import json
import os
import time
import traceback
import types

import torch
from transformers import AutoTokenizer

from fastkernels import eval as E
from fastkernels.infra import compilation as C
from fastkernels.list import apply_candidates, discover_candidate_impls
from fastkernels.workloads import resolve_benchmark

MODE = os.environ["PROBE_MODE"]  # "candidate" or "noise"
sc = resolve_benchmark("/tmp/probe.yaml")[0]
args = types.SimpleNamespace(max_requests=int(os.environ.get("PROBE_MAX_REQUESTS", "32")), seed=42,
                             enforce_eager=os.environ.get("PROBE_EAGER") == "1", max_layers=None,
                             temperature=0.0, self_test=MODE == "noise")
tok = AutoTokenizer.from_pretrained(sc.hf_name)
tr, lr, msl, _ = E._load_scenario_runs(sc, args, tok)
t0 = time.time()
base = E._run_impl(sc, args, tr, lr, msl)
result = {"mode": MODE, "eager": args.enforce_eager, "t_baseline_s": round(time.time() - t0)}
try:
    if MODE == "noise":
        for name, obj in vars(C).items():
            if isinstance(obj, type) and isinstance(getattr(obj, "_loaded_artifacts", None), dict):
                obj._loaded_artifacts.clear()
        torch._dynamo.reset()
    else:
        t0 = time.time()
        pairs = discover_candidate_impls()  # imports (and JIT-builds) the candidates
        result["t_import_s"] = round(time.time() - t0)
        result["swapped"] = [f"L{t.level}:{t.name}" for t, _, _ in pairs]
        apply_candidates(pairs)
    t0 = time.time()
    cand = E._run_impl(sc, args, tr, lr, msl)
    result["t_candidate_s"] = round(time.time() - t0)
except Exception:  # noqa: BLE001 -- a crash is a result
    result["crash"] = traceback.format_exc()[-3000:]
    print("RESULT " + json.dumps(result))
    raise SystemExit(0)

a_all = base["throughput"][0]["token_ids"]
b_all = cand["throughput"][0]["token_ids"]


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


b_tp = base["throughput"][0]["total_output_tokens"] / base["throughput"][0]["elapsed"]
c_tp = cand["throughput"][0]["total_output_tokens"] / cand["throughput"][0]["elapsed"]
b_lat = E._median(base["latency"][0]["latencies"]) if base["latency"] else None
c_lat = E._median(cand["latency"][0]["latencies"]) if cand["latency"] else None
result.update({
    "throughput_speedup": c_tp / b_tp,
    "latency_speedup": (b_lat / c_lat) if b_lat and c_lat else None,
    "exact_match": sum(a == b for a, b in zip(a_all, b_all)) / len(a_all),
    "first_token_agreement": sum(bool(a) and bool(b) and a[0] == b[0] for a, b in zip(a_all, b_all)) / len(a_all),
    "mean_prefix_frac": sum(prefix(a, b) / max(1, len(a)) for a, b in zip(a_all, b_all)) / len(a_all),
    "position_match": E._alignment(a_all, b_all)["matched_tokens"] / max(1, sum(map(len, a_all))),
})
print("RESULT " + json.dumps(result))
