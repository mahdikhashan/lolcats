# Distill and finetune Llama 3.2 1B with subquadratic attentions (evaluation is a separate step)
# -> make lolcats  # LoLCATs attention (Hedgehog linear attention + sliding window)
# -> make lizard   # Lizard attention (gated linear attention + sliding window with sink tokens)
# Pass extra distill_llama.py flags with ARGS, e.g., make lizard ARGS="--no_wandb"
# Set HF_REPO=<hf-user>/<repo> to push checkpoints and results to the Hub as they are saved
#
# Docker + Hugging Face Jobs:
# -> make docker-build docker-push IMAGE=<dockerhub-user>/lolcats
# -> make hf-job IMAGE=<dockerhub-user>/lolcats HF_REPO=<hf-user>/<repo> [TARGET=lizard] [HF_FLAVOR=h200]
# -> make hf-job-finetune IMAGE=<dockerhub-user>/lolcats HF_REPO=<hf-user>/<repo>  # Lizard finetune only, from the distill checkpoint in HF_REPO (H200)
# -> make hf-job IMAGE=... HF_REPO=... HF_FLAVOR=h200 ARGS="--model_config <model_config> --no_finetune"  # distill only, with another model config
# -> make distill-local  # distill only (stage 1) on this machine's GPU, in the lolcats-env conda env (no Docker, no HF Jobs)

PYTHON ?= python

DISTILL_CONFIG  ?= distill_alpaca_clean_xent0_mse1000_lr1e-2_1b
FINETUNE_CONFIG ?= finetune_lora_qkvo_alpaca_clean_1b
SEED            ?= 0
REPLICATE       ?= 0
ARGS            ?=

IMAGE      ?= lolcats
TARGET     ?= lizard
HF_FLAVOR  ?= a100-large
HF_TIMEOUT ?= 12h
HF_REPO    ?=
# distill-local: stage 1 only, on a local GPU (GPU is the index in nvidia-smi order)
LOCAL_MODEL_CONFIG ?= distill_llama3_2_1b_lizard_w128_fd32_m4
GPU                ?= 0
CACHE_DIR          ?= $(HOME)/.cache/huggingface/hub
# Lizard distill checkpoint in HF_REPO, at the path `make lizard` pushes it to
DISTILL_CKPT ?= checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/dl-d=$(DISTILL_CONFIG)-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=$(FINETUNE_CONFIG)-s=$(SEED)-se=$(SEED)-re=$(REPLICATE)_distill.pt

TRAIN_ARGS = --distill_config $(DISTILL_CONFIG) \
	--finetune_config $(FINETUNE_CONFIG) \
	--no_init_eval --verbose --seed $(SEED) --replicate $(REPLICATE) $(ARGS)

.PHONY: help lolcats lizard distill-local docker-build docker-push hf-job hf-job-finetune

help:
	@echo "make lolcats   # distill + finetune with LoLCATs attention"
	@echo "make lizard    # distill + finetune with Lizard attention"
	@echo "make distill-local [LOCAL_MODEL_CONFIG=...] [GPU=0]  # distill only, on a local GPU in the conda env"
	@echo "make docker-build docker-push IMAGE=<dockerhub-user>/lolcats"
	@echo "make hf-job IMAGE=<dockerhub-user>/lolcats HF_REPO=<hf-user>/<repo>"
	@echo "make hf-job-finetune IMAGE=<dockerhub-user>/lolcats HF_REPO=<hf-user>/<repo>"

lolcats:
	$(PYTHON) distill_llama.py --model_config distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 \
	--lk_zero_init $(TRAIN_ARGS)

lizard:
	$(PYTHON) distill_llama.py --model_config distill_llama3_2_1b_lizard_w128_fd128_m4 \
	$(TRAIN_ARGS)

# Stage 1 only, on this machine (conda activate lolcats-env; export HF_TOKEN=... for Llama and Alpaca)
# -> make distill-local ARGS="--max_steps 10"  # a short run first, to check that the memory is sufficient
# Checkpoints go to checkpoints/$(LOCAL_MODEL_CONFIG)/; set HF_REPO=<hf-user>/<repo> to also push them to the Hub
distill-local:
	@$(PYTHON) -c 'import torch; assert torch.cuda.is_available(), "PyTorch does not see a GPU"'
	CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(GPU) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	HF_REPO=$(HF_REPO) $(PYTHON) distill_llama.py --model_config $(LOCAL_MODEL_CONFIG) \
	--cache_dir $(CACHE_DIR) --no_finetune $(if $(WANDB_API_KEY),,--no_wandb) $(TRAIN_ARGS)

# HF Jobs run on x86-64, so build for it even on Apple silicon
docker-build:
	docker build --platform linux/amd64 -t $(IMAGE) .

docker-push:
	docker push $(IMAGE)

hf-job:
	@test -n "$(HF_REPO)" || (echo "Set HF_REPO=<hf-user>/<repo>; the job's disk is deleted when it ends"; exit 1)
	hf jobs run --flavor $(HF_FLAVOR) --timeout $(HF_TIMEOUT) --detach \
	--secrets HF_TOKEN $(if $(WANDB_API_KEY),--secrets WANDB_API_KEY) --env HF_REPO=$(HF_REPO) \
	$(IMAGE) make $(TARGET) ARGS="$(if $(WANDB_API_KEY),,--no_wandb) $(ARGS)"

# Skip distillation: download the distill checkpoint from HF_REPO, then finetune from it with `make lizard`
FINETUNE_JOB = huggingface-cli download $$HF_REPO "$(DISTILL_CKPT)" --local-dir . && \
	make lizard DISTILL_CONFIG=$(DISTILL_CONFIG) FINETUNE_CONFIG=$(FINETUNE_CONFIG) SEED=$(SEED) REPLICATE=$(REPLICATE) \
	ARGS="--load_distill_checkpoint default $(if $(WANDB_API_KEY),,--no_wandb) $(ARGS)"

hf-job-finetune: HF_FLAVOR = h200
hf-job-finetune:
	@test -n "$(HF_REPO)" || (echo "Set HF_REPO=<hf-user>/<repo>, the repo with the distill checkpoint"; exit 1)
	hf jobs run --flavor $(HF_FLAVOR) --timeout $(HF_TIMEOUT) --detach \
	--secrets HF_TOKEN $(if $(WANDB_API_KEY),--secrets WANDB_API_KEY) --env HF_REPO=$(HF_REPO) \
	$(IMAGE) bash -c '$(FINETUNE_JOB)'
