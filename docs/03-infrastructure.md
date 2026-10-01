# 3. Infrastructure

## Requirements for the training configuration

1. Training runs in Docker.
2. Training does no evaluation. Evaluation is a separate step.
3. Each time the training saves a checkpoint or a results file, it pushes the file to a Hugging Face repository immediately.
4. The model uses flash-attention, as in the code, not SDPA. The environment contains the necessary library.
5. The conda environment is exactly the same as `environment.yaml`.
6. All parts stay simple.

## First approach (replaced)

The first configuration for HF Jobs (commit `40d275c`) did not merge. It used the standard image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. The script `hf_jobs/run.sh` did these steps:

1. It cloned the repository at the current commit.
2. It installed pinned requirements with pip.
3. It ran `make lizard`.
4. It uploaded `checkpoints/` and `results/` every 30 minutes and at exit.

This approach changed the model to SDPA, so that it did not need flash-attn. The Docker image below replaced it, because the image satisfies the requirements above.

## Docker image (PR #2, commits `f890c21`, `96e2f79`)

- **Base:** `condaforge/miniforge3:26.7.2-0`, plus `make`.
- **Environment:** `conda env create -f environment.yaml`, with `CONDA_OVERRIDE_CUDA=12.4`. This variable is necessary because no GPU is available during the build. The `bin` directory of the environment is first on `PATH`.
- **Code:** The build copies the repository into `/workspace/lolcats`. The default command is `make lizard`.
- **Size:** approximately 4.7 GB compressed and approximately 15 GB unpacked.
- **Platform:** The build is always for `linux/amd64` (commit `96e2f79`). On an Apple-silicon Mac, Docker builds for ARM by default. But HF Jobs machines are x86-64, and the CUDA builds of PyTorch and flash-attn exist only for x86-64.

### Changes to `environment.yaml`

| Change | Reason |
|---|---|
| `pytorch::pytorch=2.5.1`, pinned, from the `pytorch` channel | Without the pin, the solver selected a CPU-only PyTorch (2.13) from conda-forge. |
| The prebuilt wheel of flash-attn 2.7.4.post1 (`cu12torch2.5cxx11abiFALSE-cp311`), added under `pip:` | A build of flash-attn from source takes hours. With the wheel, a local `conda env create` and the image stay identical. |

These pinned versions are important: `transformers=4.43.1`, `peft=0.9.0`, `datasets=2.15.0`, `pytorch-cuda=12.4`, Python 3.11. For each package without a pin, the build installs the newest compatible version available at build time.

A later direct commit (`312616c`) added conda download settings to the Dockerfile: `remote_connect_timeout_secs 60`, `remote_read_timeout_secs 600`, `remote_max_retries 10`.

### Pushes of outputs to the Hub

`src/utils/hub.py` contains `push_to_hub(path)`. If `HF_REPO` has a value, the function creates the private repository when necessary. It then uploads the file to the same relative path, for example `checkpoints/<model_config>/<run_name>_distill.pt`. If an upload fails, the function writes the error to the log and the training continues. The training calls the function at these points:

- after each save of a best checkpoint,
- after each periodic save (`_<step>.pt`, every 1,000 steps),
- after each write of the results CSV.

### No evaluation

PR #2 removed the final evaluation and the sample generations. `distill_llama.py` needed a fix, so that it does not stop with an error at the end when the command has no evaluation config. The validation loss every 100 steps stays, because it selects the best checkpoint.

## Builds of the image

1. **Apple-silicon Mac on a home connection: failed.** The large CUDA packages (libcublas, libcufft, libcusparse, mkl) are 100–230 MB each. Their downloads exceeded the short read timeout of conda, and conda stopped after 3 retries. The solve of the environment itself was successful.
2. **University server `student06`: not possible.** The Docker installation needs `sudo`, and the account does not have it.
3. **GitHub Actions: works (PR #3, commit `7f2d9cd`).** The workflow `.github/workflows/docker-image.yml` runs on request (`workflow_dispatch`). It does these steps:
   1. It makes disk space free on the runner.
   2. It authenticates with Docker Hub with the repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.
   3. It runs `make docker-build docker-push`.

To build the image again after a code change, merge the change into `main`. Then select Actions → "Docker image" → Run workflow.

The job always runs the code inside the image. Thus a code change has no effect on HF Jobs until a new build of the image is available.

## Hugging Face Jobs

### CLI version

An old `huggingface_hub` CLI did not accept `--flavor h200`. It checks the flavor against an internal list that is older than the H200. It also did not have the command `hf jobs hardware`. The conda and Homebrew Python installations on the Mac did not work. Thus the installation of the new CLI used `uv tool install huggingface_hub`.

### Make targets

| Target | Command that the target starts |
|---|---|
| `make docker-build docker-push IMAGE=<user>/lolcats` | Builds for linux/amd64 and pushes the image. |
| `make hf-job IMAGE=... HF_REPO=... [TARGET=lizard] [HF_FLAVOR=...] [HF_TIMEOUT=...]` | `hf jobs run --flavor $(HF_FLAVOR) --timeout $(HF_TIMEOUT) --detach --secrets HF_TOKEN [--secrets WANDB_API_KEY] --env HF_REPO=... $(IMAGE) make $(TARGET) ARGS=...`. It adds `--no_wandb` when no W&B key is available. Defaults: `a100-large`, `12h`. |
| `make hf-job-finetune IMAGE=... HF_REPO=...` (PR #6, commit `47be168`) | Stage 2 only, on `h200` by default. It downloads the stage 1 checkpoint from `HF_REPO` with `huggingface-cli download`. Then it runs `make lizard ARGS="--load_distill_checkpoint default"`. |

Notes about `hf-job-finetune`:

- The job command is in a make variable (`FINETUNE_JOB`). In a recipe, a backslash-newline inside a single-quoted `bash -c` string goes to the job as a literal backslash.
- The target gives `DISTILL_CONFIG`, `FINETUNE_CONFIG`, `SEED` and `REPLICATE` to the inner `make lizard`. Thus the downloaded file always has the path that `--load_distill_checkpoint default` looks for.
- `HF_FLAVOR = h200` is a default for this target only. A value of `HF_FLAVOR=...` on the command line replaces it. An exported environment variable does not replace it. `HF_TIMEOUT` (default `12h`) accepts a value from the command line or from the environment.
- The target uses `huggingface-cli`, not `hf`. With `transformers=4.43.1`, `huggingface_hub` stays below version 1.0. In these versions, `huggingface-cli download` always exists, but `hf` exists only from version 0.34.

## W&B (PR #4, commit `75d9408`)

The code had the fixed W&B entity `mahdikhashan1`. The account of the API key has the default team `nano-apps`. W&B does not let such accounts log to a personal entity. Thus `wandb.init` failed with the message "the provided API key cannot access this resource". PR #4 removed the entity argument. Now runs go to the default entity of the key (`nano-apps`, project `lolcats-personal`).

**Known W&B issue (no fix yet):** Both stages log into the same W&B run with `step=grad_step`. Stage 1 ends at step 1,178, and stage 2 starts again at step 0. W&B ignores each step that is lower than the last step ("Tried to log to step 1 that is less than the current step 1178"). Thus **W&B receives no stage 2 metrics**. The results CSV on the Hub contains the stage 2 evaluation losses.

## Checkpoint names

The name of a checkpoint comes from the run name. `src/utils/setup.py` builds the run name in three steps:

1. `get_run_name_from_args` builds `dl-d=<distill_config>-m=<model_config>-f=<finetune_config>-s=<seed>`.
2. `update_config_from_args` on the distill config adds `-se=<seed>-re=<replicate>`.
3. For stage 2, `prepare_finetune_configs` calls `update_config_from_args` again. This call adds each trainer argument that has a value at that point.
   - In a full run, stage 1 copies its trainer settings into the arguments. Thus the call adds `-bs=1-gas=8-nte=2-ms=-1-se=0-re=0`.
   - In a run of stage 2 only (`--load_distill_checkpoint default`), stage 1 does not run. Thus the call adds only `-se=0-re=0`.

Thus a run of stage 2 only writes a stage 2 checkpoint with a **different name**. It does not overwrite the checkpoint of a full run. The first prediction of this behavior was wrong. A replay of the naming code with the exact arguments gave the right names, and the Hub listing showed the same names.

The files below are in `nanoman1/lolcats-lizard-llama-3.2-1b`, under `checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/`. Each name starts with the prefix `dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0`.

| Suffix | Size | Content |
|---|---|---|
| `-ms=10-mfs=10-se=0-re=0_distill.pt` | 639 kB | Run 0, stage 1 |
| `-ms=10-mfs=10-se=0-re=0-bs=1-gas=8-nte=2-ms=10-mfs=10-se=0-re=0_ft.pt` | 3.5 MB | Run 0, stage 2 |
| `-se=0-re=0_distill.pt` | 638 kB | **Run 1, stage 1 (best validation loss). All later steps used this file.** |
| `-se=0-re=0_distill_1000.pt` | 638 kB | Run 1, stage 1 at step 1,000 |
| `-se=0-re=0-bs=1-gas=8-nte=2-ms=-1-se=0-re=0_ft.pt` | 3.5 MB | Run 1, best stage 2 checkpoint before the crash (step ≤ 386) |
| `-se=0-re=0-se=0-re=0_ft.pt` | 3.49 MB | **Run 2, stage 2. The evaluation used this file.** |

The results CSVs (`_ft.csv`) have the same names and are under `results/`. The sizes agree with the parameter counts. The stage 1 files contain 294,992 Lizard parameters in bf16 (~590 kB). The stage 2 files contain 1,703,936 LoRA parameters in bf16 (~3.4 MB).
