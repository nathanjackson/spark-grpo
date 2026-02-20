import logging
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import transformers

import blackjack
from grpo import train_grpo

if "__main__" == __name__:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"grpo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_filename = os.path.join(run_dir, "grpo.log")
    log_handlers = [
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )
    logger = logging.getLogger("grpo")

    model_id = "ibm-granite/granite-4.0-350m"
    #model_id = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False  # Freeze all parameters
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

    base_lr = 5e-6
    total_steps = 10000
    warmup_steps = 20
    optim = torch.optim.AdamW(policy_model.parameters(), lr=base_lr, fused=True)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1.0 / max(1, warmup_steps),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=max(1, total_steps - warmup_steps),
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    eval_every = 100
    eval_games = 1000
    eval_seed = 1337
    train_temperature = 1.6
    max_seq_len = 384
    groups_per_batch = 4
    rollouts_per_group = 12
    kl_coef = 0.2
    entropy_coef = 0.03
    ppo_epochs = 2

    train_grpo(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optim=optim,
        sched=sched,
        logger=logger,
        run_dir=run_dir,
        total_steps=total_steps,
        eval_every=eval_every,
        eval_games=eval_games,
        eval_seed=eval_seed,
        train_temperature=train_temperature,
        max_seq_len=max_seq_len,
        groups_per_batch=groups_per_batch,
        rollouts_per_group=rollouts_per_group,
        kl_coef=kl_coef,
        entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs,
        generate_trajectory=blackjack.generate_trajectory,
        game_cls=blackjack.Blackjack,
    )
