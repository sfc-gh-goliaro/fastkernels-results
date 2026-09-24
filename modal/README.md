# Modal tooling for the FastKernels paper experiments

`fk_modal.py` builds a B200 image with fastkernels' pinned stack (CUDA 13 devel, torch 2.11,
vllm 0.26, ...) from the local `../fastkernels` checkout, plus two volumes: `fk-hf-cache`
(HF weights/datasets) and `fk-jit-cache` (JIT builds, dev outputs, ~/.fastkernels).

    modal run modal/fk_modal.py::download_many --repos org/a,org/b          # CPU, parallel
    FK_DEV_DIR=/path/to/fastkernels modal run modal/fk_modal.py::dev \
        --name NAME --gpus N --cmd 'python -m fastkernels e2e ... --out $FK_OUT/x'

`dev` mounts the given checkout at /opt/fk-dev (first on PYTHONPATH), so code changes need no
image rebuild. `$FK_OUT` (= /jit/dev/NAME) persists on the `fk-jit-cache` volume.
`e2e_probe_worker.py` and the `probe*` functions are the earlier Llama-only probes.

Notes: Modal's /root is on sys.path, so the fastkernels source is installed from /opt;
`datasets>=3` is pinned because uv otherwise resolves datasets 1.1.1 (broken on pyarrow>=21).
