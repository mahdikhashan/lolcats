# 3. Infrastructure

## Requirements set for the training setup

1. Training runs in Docker.
2. No evaluation during training; it is a separate step.
3. Every checkpoint and results file is pushed to a Hugging Face repository as soon as it is saved.
4. Use flash-attention as in the code (not SDPA), with the required library installed.
5. The environment matches `environment.yaml` exactly (conda).
6. Keep everything simple.

## First approach (replaced)

The first HF Jobs setup (commit `40d275c`, never merged) ran on the stock
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image. `hf_jobs/run.sh` cloned the repo at the
current commit, pip-installed pinned requirements, ran `make lizard`, and uploaded `checkpoints/`
and `results/` every 30 minutes and on exit. It switched the model to SDPA to avoid flash-attn. It
was replaced by the Docker image below to meet the requirements above.

## Docker image (PR #2, commits `f890c21`, `96e2f79`)

- **Base:** `condaforge/miniforge3:26.7.2-0`, plus `make`.
- **Environment:** `conda env create -f environment.yaml`, with `CONDA_OVERRIDE_CUDA=12.4`
  because there is no GPU at build time. The env's `bin` is put first on `PATH`.
- **Code:** the repo is copied into `/workspace/lolcats`; the default command is `make lizard`.
- **Size:** about 4.7 GB compressed, about 15 GB unpacked.
- **Platform:** always built for `linux/amd64` (commit `96e2f79`). On an Apple-silicon Mac Docker
  builds for ARM by default, but HF Jobs machines are x86-64, and the CUDA PyTorch and flash-attn
  builds only exist for x86-64.

### Changes to `environment.yaml`

| Change | Reason |
|---|---|
| `pytorch::pytorch=2.5.1` pinned from the `pytorch` channel | Unpinned, the solver picked a CPU-only PyTorch (2.13) from conda-forge |
| flash-attn 2.7.4.post1 prebuilt wheel (`cu12torch2.5cxx11abiFALSE-cp311`) added under `pip:` | Building flash-attn from source takes hours; a local `conda env create` and the image stay identical |

Pinned versions that matter: `transformers=4.43.1`, `peft=0.9.0`, `datasets=2.15.0`,
`pytorch-cuda=12.4`, Python 3.11. Unpinned packages resolve to the newest compatible version at
build time.

A later direct commit (`312616c`) added conda download settings to the Dockerfile:
`remote_connect_timeout_secs 60`, `remote_read_timeout_secs 600`, `remote_max_retries 10`.

### Pushing outputs to the Hub

`src/utils/hub.py` has `push_to_hub(path)`: if `HF_REPO` is set, it creates the private repo if
needed and uploads the file to the same relative path (for example
`checkpoints/<model_config>/<run_name>_distill.pt`). A failed upload is logged and training
continues. It is called after every best-checkpoint save, every periodic save
(`_<step>.pt`, every 1,000 steps), and every results CSV write.

### No evaluation

The final evaluation and the sample generations were removed. `distill_llama.py` needed a fix so it
doesn't crash at the end when no eval config is given. The validation loss every 100 steps still
runs, because it selects the best checkpoint.

## Building the image

1. **Apple-silicon Mac on a home connection: failed.** Conda's downloads of the large CUDA
   packages (libcublas, libcufft, libcusparse, mkl, 100–230 MB each) hit conda's short read
   timeout and gave up after 3 retries. The environment itself solved correctly.
2. **University server `student06`: not possible.** Installing Docker needs `sudo`, which the
   account doesn't have.
3. **GitHub Actions: works (PR #3, commit `7f2d9cd`).** `.github/workflows/docker-image.yml` runs
   on demand (`workflow_dispatch`), frees disk space on the runner, logs in to Docker Hub with the
   repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, and runs
   `make docker-build docker-push`. To rebuild after a code change: merge into `main`, then
   Actions → "Docker image" → Run workflow.

The job always runs the code baked into the image, so every code change needs a rebuild before it
takes effect on HF Jobs.

## Hugging Face Jobs

### CLI version

An old `huggingface_hub` CLI rejected `--flavor h200`, because it validates against a built-in list
that predates the H200, and it had no `hf jobs hardware` command. The Mac's conda and Homebrew
Python installs were broken, so the new CLI was installed with `uv tool install huggingface_hub`.

### Make targets

| Target | Command it launches |
|---|---|
| `make docker-build docker-push IMAGE=<user>/lolcats` | Build for linux/amd64 and push |
| `make hf-job IMAGE=... HF_REPO=... [TARGET=lizard] [HF_FLAVOR=...] [HF_TIMEOUT=...]` | `hf jobs run --flavor $(HF_FLAVOR) --timeout $(HF_TIMEOUT) --detach --secrets HF_TOKEN [--secrets WANDB_API_KEY] --env HF_REPO=... $(IMAGE) make $(TARGET) ARGS=...`; adds `--no_wandb` when no W&B key is set. Defaults: `a100-large`, `12h` |
| `make hf-job-finetune IMAGE=... HF_REPO=...` (PR #6, commit `47be168`) | Stage 2 only, on `h200` by default: downloads the stage 1 checkpoint from `HF_REPO` with `huggingface-cli download`, then runs `make lizard ARGS="--load_distill_checkpoint default"` |

Notes on `hf-job-finetune`:

- The job command is a make variable (`FINETUNE_JOB`), because backslash-newlines inside a
  single-quoted `bash -c` string in a recipe would reach the job as literal backslashes.
- `DISTILL_CONFIG`, `FINETUNE_CONFIG`, `SEED` and `REPLICATE` are passed to the inner `make lizard`,
  so the downloaded file and the path `--load_distill_checkpoint default` looks for always match.
- `HF_FLAVOR = h200` is a target-specific default. A command-line `HF_FLAVOR=...` overrides it; an
  exported environment variable does not. `HF_TIMEOUT` (default `12h`) can be set either way.
- `huggingface-cli` is used rather than `hf`, because `transformers=4.43.1` keeps `huggingface_hub`
  below 1.0, where `huggingface-cli download` always exists (`hf` only exists from 0.34).

## W&B (PR #4, commit `75d9408`)

The code hard-coded the W&B entity `mahdikhashan1`. The API key's account has the default team
`nano-apps`, and W&B no longer lets such accounts log to a personal entity, so `wandb.init` failed
with "the provided API key cannot access this resource". The entity argument was removed, so runs
go to the key's default entity (`nano-apps`, project `lolcats-personal`).

**Known W&B issue (not fixed):** both stages log into the same W&B run with `step=grad_step`.
Stage 1 ends at step 1,178 and stage 2 restarts at 0, and W&B drops any step lower than the last
one ("Tried to log to step 1 that is less than the current step 1178"). So **no stage 2 metrics
reach W&B**. The stage 2 evaluation losses are still in the results CSV uploaded to the Hub.

## Checkpoint names

Checkpoints are named after the run, built in `src/utils/setup.py`:

1. `get_run_name_from_args` builds
   `dl-d=<distill_config>-m=<model_config>-f=<finetune_config>-s=<seed>`.
2. `update_config_from_args` on the distill config appends `-se=<seed>-re=<replicate>`.
3. For stage 2, `prepare_finetune_configs` calls `update_config_from_args` again. It appends every
   trainer argument that is set at that point. In a full run the distill stage has copied its
   trainer settings into the arguments, so this adds `-bs=1-gas=8-nte=2-ms=-1-se=0-re=0`. In a
   finetune-only run (`--load_distill_checkpoint default`) the distill stage is skipped, so it only
   adds `-se=0-re=0`.

As a result, a finetune-only run writes a **differently named** stage 2 checkpoint and does not
overwrite a full run's. (This was first predicted wrongly, then corrected by replaying the naming
code with the exact arguments; the Hub listing confirmed the corrected names.)

Files in `nanoman1/lolcats-lizard-llama-3.2-1b` (prefix `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0`,
under `checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/`):

| Suffix | Size | What it is |
|---|---|---|
| `-ms=10-mfs=10-se=0-re=0_distill.pt` | 639 kB | Smoke test, stage 1 |
| `-ms=10-mfs=10-se=0-re=0-bs=1-gas=8-nte=2-ms=10-mfs=10-se=0-re=0_ft.pt` | 3.5 MB | Smoke test, stage 2 |
| `-se=0-re=0_distill.pt` | 638 kB | **First full run, stage 1 (best validation loss); used for everything after** |
| `-se=0-re=0_distill_1000.pt` | 638 kB | First full run, stage 1 at step 1,000 |
| `-se=0-re=0-bs=1-gas=8-nte=2-ms=-1-se=0-re=0_ft.pt` | 3.5 MB | First full run, stage 2 best before the crash (step ≤ 386) |
| `-se=0-re=0-se=0-re=0_ft.pt` | 3.49 MB | **Finetune-only rerun, stage 2 (the evaluated model)** |

Results CSVs with the same names (`_ft.csv`) are under `results/`. The sizes fit the parameter
counts: stage 1 files hold 294,992 Lizard parameters in bf16 (~590 kB), stage 2 files hold
1,703,936 LoRA parameters in bf16 (~3.4 MB).
