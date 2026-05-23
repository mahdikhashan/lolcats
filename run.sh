python distill_llama.py --model_config distill_llama3_1_1b_lk_smd_wtk64_fd64_w01 \
  --distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2_1b \
  --finetune_config finetune_lora_qkvo_alpaca_clean_1b \
  --eval_config eval_alpaca_clean \
  --lk_zero_init --verbose --seed 0 --replicate 0