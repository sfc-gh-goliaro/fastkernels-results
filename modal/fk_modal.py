"""Modal environment for FastKernels paper experiments (B200).

Image: CUDA 13 devel (nvcc for candidate JIT builds) + the pinned fastkernels stack,
installed from the local ``fastkernels`` checkout next to this repo. Volumes cache
Hugging Face weights/datasets and JIT build outputs across runs.

    modal run modal/fk_modal.py::versions                   # build image, check imports (CPU)
    modal run modal/fk_modal.py::download --repo meta-llama/Llama-3.1-8B-Instruct   (CPU)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Fixed snapshot of fastkernels (9acebaa) for the image's dependency install, so editing the
# working checkout never rebuilds the heavy layers; dev runs mount the live code on top.
FK_LOCAL = Path(os.environ.get("FK_IMAGE_BASE", HERE.parent.parent / "fk-wt" / "image-base"))
CANDIDATES_LOCAL = HERE.parent / "agent-candidates"

DEEPGEMM = "deep-gemm @ git+https://github.com/deepseek-ai/DeepGEMM.git@a6b593d2826719dcf4892609af7b84ee23aaf32a"
JIT = "/jit"      # JIT build caches (volume)
HF = "/hf"        # Hugging Face cache (volume)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 "
        "ninja packaging wheel setuptools numpy")
    # add_python's interpreter defaults extension builds to clang; use the system gcc.
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0",
          "CC": "gcc", "CXX": "g++"})
    .run_commands(f"uv pip install --system --no-build-isolation '{DEEPGEMM}'")
    .add_local_dir(FK_LOCAL, "/opt/fastkernels", copy=True,
                   ignore=[".git", "**/__pycache__", "**/*.pyc"])
    .run_commands("cd /opt/fastkernels && uv pip install --system --no-build-isolation-package fastkernels "
                  "--no-build-isolation-package flash-attn --no-build-isolation-package deep-gemm -e .")
    # fastkernels leaves `datasets` unpinned and uv's full resolve lands on 1.1.1, which
    # is incompatible with pyarrow>=21; a modern release resolves cleanly on its own.
    .run_commands("uv pip install --system 'datasets>=3'")
    .env({
        "HF_HOME": HF,
        "TORCH_EXTENSIONS_DIR": f"{JIT}/torch_extensions",
        "TRITON_CACHE_DIR": f"{JIT}/triton",
        "FLASHINFER_WORKSPACE_BASE": f"{JIT}/flashinfer",
        "VLLM_CACHE_ROOT": f"{JIT}/vllm",
        "TORCHINDUCTOR_CACHE_DIR": f"{JIT}/inductor",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(CANDIDATES_LOCAL, "/root/agent-candidates")
)

hf_vol = modal.Volume.from_name("fk-hf-cache", create_if_missing=True)
jit_vol = modal.Volume.from_name("fk-jit-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
app = modal.App("fk-paper", image=image)
VOLUMES = {HF: hf_vol, JIT: jit_vol}


@app.function(volumes=VOLUMES, timeout=1200)
def versions() -> str:
    """CPU-only sanity check of the image: key imports and toolchain."""
    out = [subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]]
    for mod in ("torch", "triton", "vllm", "flashinfer", "deep_gemm", "flash_attn", "fla",
                "cutlass", "transformers", "fastkernels"):
        try:
            m = __import__(mod)
            out.append(f"{mod:12} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"{mod:12} IMPORT FAILED: {type(exc).__name__}: {str(exc)[:160]}")
    import torch
    out.append(f"torch.version.cuda {torch.version.cuda}")
    return "\n".join(out)


@app.function(volumes=VOLUMES, secrets=[hf_secret], timeout=3600, cpu=4)
def download(repo: str, datasets: str = "", all_files: bool = False) -> str:
    """Download a model snapshot (and optional comma-separated HF datasets) into the
    HF cache volume, on CPU."""
    from huggingface_hub import snapshot_download
    patterns = None if all_files else ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
    path = snapshot_download(repo, token=os.environ["HF_TOKEN"], allow_patterns=patterns)
    if datasets:
        from datasets import load_dataset
        for name in datasets.split(","):
            load_dataset(name, split="train")
    hf_vol.commit()
    return path


@app.function(volumes=VOLUMES, timeout=1200)
def sh_remote(cmd: str) -> str:
    """Run a shell command in the image (CPU) and return its output."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/tmp")
    return r.stdout + r.stderr


@app.local_entrypoint()
def sh(cmd: str):
    """modal run modal/fk_modal.py::sh --cmd '...'"""
    print(sh_remote.remote(cmd))


@app.function(gpu="B200", volumes=VOLUMES, secrets=[hf_secret], timeout=2400)
def gpu_py_remote(code: str, env: dict | None = None) -> str:
    """Run a Python snippet on a B200 (for one-off experiments); returns its output."""
    Path("/tmp/snippet.py").write_text(code)
    Path("/tmp/probe.yaml").write_text(PROBE_YAML)
    r = subprocess.run(["python", "/tmp/snippet.py"], capture_output=True, text=True, cwd="/tmp",
                       env={**os.environ, **(env or {})})
    jit_vol.commit()
    return (r.stdout + r.stderr)[-20_000:]


@app.local_entrypoint()
def gpu_py(file: str, env: str = ""):
    """modal run modal/fk_modal.py::gpu_py --file snippet.py [--env K=V,K2=V2]"""
    kv = dict(x.split("=", 1) for x in env.split(",") if x)
    print(gpu_py_remote.remote(Path(file).read_text(), kv))


PROBE_YAML = """\
scenarios:
  - model: meta-llama/Llama-3.1-8B-Instruct
    tp: 1
    dtype: bfloat16
    workloads: [LLM.mixed, LLM.single_request]
"""


@app.function(gpu="B200", volumes=VOLUMES, secrets=[hf_secret], timeout=2400)
def probe(cand_set: str, max_requests: int = 32, eager: bool = False, extra: str = "",
          only: str = "") -> dict:
    """Unmodified `fastkernels eval` on Llama-3.1-8B with one candidate set swapped in."""
    import json
    import shutil
    import time

    # Candidates JIT-build next to their own files, so run from a writable copy on the
    # JIT volume (builds are then reused by later runs of the same set).
    # ``only`` (e.g. "L1:rms_norm,L2:attention") keeps just those kernels, for bisection.
    src = Path("/root/agent-candidates") / cand_set
    dst = Path(JIT) / "cands" / (cand_set + (f"__{abs(hash(only)) % 10**8}" if only else ""))
    if not dst.exists():
        if only:
            keep = {tuple(k.split(":")) for k in only.split(",")}
            shutil.copytree(src, dst, ignore=lambda d, names: [
                n for n in names if n.endswith(".py") and (Path(d).name, n[:-3]) not in keep])
        else:
            shutil.copytree(src, dst)
    yaml = Path("/tmp/probe.yaml")
    yaml.write_text(PROBE_YAML)
    out = Path(f"/tmp/probe_{cand_set}.json")
    cmd = ["python", "-m", "fastkernels", "eval", str(yaml), "--max-requests", str(max_requests),
           "--output", str(out), "--gpus", "0"] + (["--enforce-eager"] if eager else []) + extra.split()
    t0 = time.time()
    # The fastkernels source lives in /opt (not on sys.path, unlike Modal's /root) so the
    # editable install is what `import fastkernels` resolves to.
    proc = subprocess.run(cmd, env={**os.environ, "FASTKERNELS_CANDIDATE_DIR": str(dst)},
                          capture_output=True, text=True, cwd="/tmp")
    jit_vol.commit()
    return {"set": cand_set, "only": only, "eager": eager, "extra": extra, "rc": proc.returncode,
            "elapsed_s": round(time.time() - t0),
            "report": json.loads(out.read_text()) if out.exists() else None,
            "log": (proc.stdout + proc.stderr)[-200_000:]}


@app.local_entrypoint()
def run_probe(sets: str = "selftest", max_requests: int = 32, eager: bool = False, extra: str = "",
              only: str = "", out: str = "probe_results.json"):
    """``only`` takes ";"-separated kernel groups; each group runs as its own container."""
    """modal run modal/fk_modal.py::run_probe --sets selftest,claude-seq,..."""
    import json
    groups = only.split(";") if only else [""]
    results = list(probe.starmap([(s, max_requests, eager, extra, g)
                                  for s in sets.split(",") for g in groups]))
    Path(out).write_text(json.dumps(results, indent=1))
    for r in results:
        rep = r["report"] or {}
        tp = [(w["name"], round(w["speedup"], 3), w["alignment"]["exact_matches"],
               w["alignment"]["total_requests"],
               round(w["alignment"]["matched_tokens"] / max(1, w["alignment"]["total_tokens"]), 3))
              for w in rep.get("throughput", [])]
        lat = [(w["name"], round(w["speedup"], 3)) for w in rep.get("latency", [])]
        print(f"{r['set']:14} only={r['only'][:60]!r} eager={r['eager']} extra={r['extra']!r} rc={r['rc']} {r['elapsed_s']}s thr={tp} lat={lat}")
    print(f"full results -> {out}")


PROBE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def model_subset(set_dir: Path, model: str, dst: Path) -> list[str]:
    """Copy into *dst* only the kernels *model* uses (per the set's manifest) plus any
    kernel files they import from the set, with all CUDA/C++ sources. Importing a
    candidate JIT-compiles it, so this keeps compile work to what the model runs."""
    import ast
    import json
    import shutil
    manifest = json.loads((set_dir / "manifest.json").read_text())
    todo = [tuple(k.split(":")) for k in manifest["scenarios"].get(model, [])]
    keep: set[tuple[str, str]] = set()
    while todo:
        level, stem = todo.pop()
        f = set_dir / level / f"{stem}.py"
        if (level, stem) in keep or not f.is_file():
            continue
        keep.add((level, stem))
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level in (1, 2):
                mod = (node.module or "").split(".")
                if node.level == 1:  # from .x import y / from . import x
                    todo += [(level, mod[0])] if node.module else [(level, a.name) for a in node.names]
                elif mod[0].startswith("L"):  # from ..L2.x import y / from ..L2 import x
                    todo += [(mod[0], mod[1])] if len(mod) > 1 else [(mod[0], a.name) for a in node.names]
    for src in set_dir.glob("L[1-3]/*"):
        if src.suffix != ".py" or (src.parent.name, src.stem) in keep:
            out = dst / src.parent.name / src.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    return sorted(f"{l}:{s}" for l, s in keep)


@app.function(gpu="B200", volumes=VOLUMES, secrets=[hf_secret], timeout=1200)
def probe2(cand_set: str, worker: str, eager: bool = False, max_requests: int = 32,
           run_id: str = "probe") -> dict:
    """Llama-3.1-8B e2e probe via e2e_probe_worker.py; cand_set "noise" = recompiled baseline.
    The result is also written to the JIT volume (results/<run_id>/<set>.json) right away."""
    import json
    import shutil
    env = {**os.environ, "PROBE_MODE": "noise" if cand_set == "noise" else "candidate",
           "PROBE_EAGER": "1" if eager else "0", "PROBE_MAX_REQUESTS": str(max_requests)}
    res: dict = {"set": cand_set, "eager": eager}
    if cand_set != "noise":
        dst = Path("/tmp/cands") / cand_set
        res["kernels"] = model_subset(Path("/root/agent-candidates") / cand_set, PROBE_MODEL, dst)
        env["FASTKERNELS_CANDIDATE_DIR"] = str(dst)
        # Per-set build dir: sets can reuse extension names (e.g. silu_and_mul) with
        # different sources, so a shared dir would make parallel sets clobber each other.
        env["TORCH_EXTENSIONS_DIR"] = f"{JIT}/torch_extensions/{cand_set}"
    else:
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    Path("/tmp/probe.yaml").write_text(PROBE_YAML)
    Path("/tmp/worker.py").write_text(worker)
    try:  # stay under the function timeout so a stuck set still reports
        r = subprocess.run(["python", "/tmp/worker.py"], capture_output=True, text=True, cwd="/tmp",
                           env=env, timeout=1080)
        log, res["rc"] = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as exc:
        log, res["rc"] = f"TIMEOUT after 1080s\n{(exc.stdout or b'')[-2000:]!r}", "timeout"
    line = next((l for l in log.splitlines() if l.startswith("RESULT ")), None)
    res.update(json.loads(line[len("RESULT "):]) if line else {"crash": log[-3000:]})
    out = Path(JIT) / "results" / run_id / f"{cand_set}{'-eager' if eager else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    jit_vol.commit()
    return res


@app.local_entrypoint()
def run_probe2(sets: str, eager: bool = False, max_requests: int = 32, out: str = "probe2.json",
               run_id: str = "probe"):
    """modal run modal/fk_modal.py::run_probe2 --sets noise,claude-seq,..."""
    import json
    worker = (HERE / "e2e_probe_worker.py").read_text()
    results = [r if isinstance(r, dict) else {"set": "?", "crash": repr(r)}
               for r in probe2.starmap([(s, worker, eager, max_requests, run_id) for s in sets.split(",")],
                                       return_exceptions=True)]
    Path(out).write_text(json.dumps(results, indent=1))
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"{'set':14} {'thr':>6} {'lat':>6} {'exact':>6} {'1st-tok':>7} {'prefix':>7} {'pos':>6} {'t_imp':>5}  crash")
    for r in results:
        print(f"{r['set']:14} {f(r.get('throughput_speedup')):>6} {f(r.get('latency_speedup')):>6} "
              f"{f(r.get('exact_match')):>6} {f(r.get('first_token_agreement')):>7} "
              f"{f(r.get('mean_prefix_frac')):>7} {f(r.get('position_match')):>6} "
              f"{str(r.get('t_import_s', '-')):>5}  "
              f"{(r.get('crash') or '').strip().splitlines()[-1][:90] if r.get('crash') else ''}")
    print(f"full results -> {out}")


@app.local_entrypoint()
def download_many(repos: str, all_files: bool = True):
    """modal run modal/fk_modal.py::download_many --repos a/b,c/d  (parallel, one app)"""
    names = [r for r in repos.split(",") if r]
    for repo, res in zip(names, download.starmap([(r, "", all_files) for r in names],
                                                  return_exceptions=True)):
        print(f"{repo}: {'OK ' + res if isinstance(res, str) else 'FAILED ' + repr(res)[:200]}")


# ---------------------------------------------------------------------------
# Development: run a shell command against a LOCAL fastkernels checkout/worktree,
# mounted at /opt/fk-dev at container start (no image rebuild per code change).
#   FK_DEV_DIR=/path/to/worktree modal run modal/fk_modal.py::dev --cmd "..." --name mytest [--gpus 2]
# $FK_OUT (= /jit/dev/<name>) persists on the fk-jit-cache volume.
# ---------------------------------------------------------------------------
FK_DEV = os.environ.get("FK_DEV_DIR", "")
dev_image = (image.add_local_dir(FK_DEV, "/opt/fk-dev", ignore=[".git", "**/__pycache__", "**/*.pyc"])
             if FK_DEV else image)


def _dev_sh(cmd: str, name: str) -> str:
    out = Path(JIT) / "dev" / name
    out.mkdir(parents=True, exist_ok=True)
    dev = Path("/opt/fk-dev")
    # Persist fastkernels' home-dir state (datasets, third_party checkouts) on the volume.
    home_fk = Path.home() / ".fastkernels"
    if not home_fk.is_symlink():
        Path(JIT, "fkhome").mkdir(parents=True, exist_ok=True)
        if home_fk.exists():
            subprocess.run(["rm", "-rf", str(home_fk)])
        home_fk.symlink_to(Path(JIT, "fkhome"))
    env = {**os.environ, "FK_OUT": str(out)}
    if dev.exists():  # the mounted checkout shadows the installed fastkernels
        env["PYTHONPATH"] = f"{dev}:{env.get('PYTHONPATH', '')}".rstrip(":")
    r = subprocess.run(["bash", "-lc", cmd], cwd=str(dev if dev.exists() else "/tmp"), env=env,
                       capture_output=True, text=True)
    jit_vol.commit()
    hf_vol.commit()
    return f"[exit {r.returncode}]\n" + (r.stdout + r.stderr)[-40_000:]


@app.function(image=dev_image, volumes=VOLUMES, secrets=[hf_secret], timeout=1800, gpu="B200")
def dev_gpu(cmd: str, name: str) -> str:
    return _dev_sh(cmd, name)


@app.function(image=dev_image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600, cpu=8, memory=32768)
def dev_cpu(cmd: str, name: str) -> str:
    return _dev_sh(cmd, name)


@app.local_entrypoint()
def dev(cmd: str, name: str = "dev", gpus: int = 1, timeout: int = 1800, cpu_only: bool = False):
    """Run `cmd` (bash) in /opt/fk-dev on B200:<gpus> (or CPU). Prints exit code + output tail."""
    fn = dev_cpu if cpu_only else dev_gpu.with_options(gpu=f"B200:{gpus}", timeout=timeout)
    print(fn.remote(cmd, name))


@app.local_entrypoint()
def main():
    print(versions.remote())
