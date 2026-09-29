# Distill, finetune, and evaluate Llama 3.2 1B with subquadratic attentions
# -> make lolcats  # LoLCATs attention (Hedgehog linear attention + sliding window)
# -> make lizard   # Lizard attention (gated linear attention + sliding window with sink tokens)
# Pass extra distill_llama.py flags with ARGS, e.g., make lizard ARGS="--no_wandb"

PYTHON ?= python

DISTILL_CONFIG  ?= distill_alpaca_clean_xent0_mse1000_lr1e-2_1b
FINETUNE_CONFIG ?= finetune_lora_qkvo_alpaca_clean_1b
EVAL_CONFIG     ?= eval_alpaca_clean
SEED            ?= 0
REPLICATE       ?= 0
ARGS            ?=

TRAIN_ARGS = --distill_config $(DISTILL_CONFIG) \
	--finetune_config $(FINETUNE_CONFIG) \
	--eval_config $(EVAL_CONFIG) \
	--verbose --seed $(SEED) --replicate $(REPLICATE) $(ARGS)

.PHONY: help lolcats lizard

help:
	@echo "make lolcats  # distill + finetune + eval with LoLCATs attention"
	@echo "make lizard   # distill + finetune + eval with Lizard attention"

lolcats:
	$(PYTHON) distill_llama.py --model_config distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 \
	--lk_zero_init $(TRAIN_ARGS)

lizard:
	$(PYTHON) distill_llama.py --model_config distill_llama3_2_1b_lizard_w128_fd128_m4 \
	$(TRAIN_ARGS)
