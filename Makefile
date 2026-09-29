# Distill and finetune Llama 3.2 1B with subquadratic attentions (evaluation is a separate step)
# -> make lolcats  # LoLCATs attention (Hedgehog linear attention + sliding window)
# -> make lizard   # Lizard attention (gated linear attention + sliding window with sink tokens)
# Pass extra distill_llama.py flags with ARGS, e.g., make lizard ARGS="--no_wandb"
# Set HF_REPO=<hf-user>/<repo> to push checkpoints and results to the Hub as they are saved
#
# Docker + Hugging Face Jobs:
# -> make docker-build docker-push IMAGE=<dockerhub-user>/lolcats
# -> make hf-job IMAGE=<dockerhub-user>/lolcats HF_REPO=<hf-user>/<repo> [TARGET=lizard] [HF_FLAVOR=h200]

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

TRAIN_ARGS = --distill_config $(DISTILL_CONFIG) \
	--finetune_config $(FINETUNE_CONFIG) \
	--no_init_eval --verbose --seed $(SEED) --replicate $(REPLICATE) $(ARGS)

.PHONY: help lolcats lizard docker-build docker-push hf-job

help:
	@echo "make lolcats   # distill + finetune with LoLCATs attention"
	@echo "make lizard    # distill + finetune with Lizard attention"
	@echo "make docker-build docker-push IMAGE=<dockerhub-user>/lolcats"
	@echo "make hf-job IMAGE=<dockerhub-user>/lolcats HF_REPO=<hf-user>/<repo>"

lolcats:
	$(PYTHON) distill_llama.py --model_config distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 \
	--lk_zero_init $(TRAIN_ARGS)

lizard:
	$(PYTHON) distill_llama.py --model_config distill_llama3_2_1b_lizard_w128_fd128_m4 \
	$(TRAIN_ARGS)

docker-build:
	docker build -t $(IMAGE) .

docker-push:
	docker push $(IMAGE)

hf-job:
	@test -n "$(HF_REPO)" || (echo "Set HF_REPO=<hf-user>/<repo>; the job's disk is deleted when it ends"; exit 1)
	hf jobs run --flavor $(HF_FLAVOR) --timeout $(HF_TIMEOUT) --detach \
	--secrets HF_TOKEN $(if $(WANDB_API_KEY),--secrets WANDB_API_KEY) --env HF_REPO=$(HF_REPO) \
	$(IMAGE) make $(TARGET) ARGS="$(if $(WANDB_API_KEY),,--no_wandb) $(ARGS)"
