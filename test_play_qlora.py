import argparse
import json
import os
import random

import torch
import transformers
from peft import PeftModel
from transformers import BitsAndBytesConfig

import blackjack
from blackjack import (
    eval_probe,
    generate_trajectory,
)


def _visible_state(game):
    state = game.get_state()
    return {
        "your_hand": state["player_hand"],
        "dealer_hand": ["Hidden"] + state["dealer_hand"][1:],
    }


def _outcome_from_reward(reward):
    if reward == -2.0:
        return "Invalid action."
    if reward == 1.0:
        return "Push (tie)."
    if reward == 2.0:
        return "Player wins."
    return "Dealer wins."


def play_game(game_id, tokenizer, model, temperature, verbose=False):
    game = blackjack.Blackjack()
    game.start_round()
    start_state = _visible_state(game)
    probe_stats = eval_probe(game, tokenizer, model)
    trajectory = generate_trajectory(
        game,
        tokenizer,
        model,
        temperature=temperature,
    )
    reward = trajectory["reward"]
    outcome = _outcome_from_reward(reward)

    if verbose:
        print(f"Game {game_id}")
        print(
            "  Probe: "
            f"p_hit={probe_stats['p_hit']:.4f} "
            f"p_stay={probe_stats['p_stay']:.4f}"
        )
        print(f"  Start state: {start_state}")
        step = 0
        for idx in range(0, len(trajectory["messages"]), 2):
            user_msg = trajectory["messages"][idx]["content"]
            action = ""
            if idx + 1 < len(trajectory["messages"]):
                action = trajectory["messages"][idx + 1]["content"]
            print(
                f"  Step {step}: "
                f"user={user_msg!r} "
                f"action={action}"
            )
            step += 1

    if verbose:
        print(f"  Final state: {game.get_state()}")
        print(f"  Result: {outcome} reward={reward}")
        print("")

    return reward, outcome, probe_stats


def _read_adapter_base_model(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return None, None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("base_model_name_or_path"), config.get("revision")


def _resolve_model_sources(model_path, model_id, adapter_path):
    resolved_adapter_path = adapter_path
    resolved_model_source = model_path or model_id
    resolved_revision = None

    if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        resolved_adapter_path = model_path
        resolved_model_source, resolved_revision = _read_adapter_base_model(model_path)

    if resolved_adapter_path and resolved_model_source is None:
        resolved_model_source, resolved_revision = _read_adapter_base_model(resolved_adapter_path)

    if resolved_model_source is None:
        raise ValueError(
            "Could not determine the base model. "
            "Pass --model-id or point --model-path/--adapter-path at a valid adapter checkpoint."
        )

    return resolved_model_source, resolved_adapter_path, resolved_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter-path")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Quantized loading requires CUDA with bitsandbytes.")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_source, adapter_path, revision = _resolve_model_sources(
        args.model_path,
        args.model_id,
        args.adapter_path,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_source,
        revision=revision,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    wins = 0
    pushes = 0
    losses = 0
    invalid = 0
    hit_prob_sum = 0.0
    stay_prob_sum = 0.0

    for i in range(1, args.games + 1):
        reward, _, probe_stats = play_game(
            i,
            tokenizer,
            model,
            args.temperature,
            verbose=args.verbose,
        )
        hit_prob_sum += probe_stats["p_hit"]
        stay_prob_sum += probe_stats["p_stay"]
        if reward == 2.0:
            wins += 1
        elif reward == 1.0:
            pushes += 1
        elif reward == -1.0:
            losses += 1
        else:
            invalid += 1

    total = wins + pushes + losses + invalid
    if total > 0:
        win_pct = 100.0 * wins / total
        push_pct = 100.0 * pushes / total
        loss_pct = 100.0 * losses / total
        invalid_pct = 100.0 * invalid / total
    else:
        win_pct = push_pct = loss_pct = invalid_pct = 0.0
    print(
        f"Summary: games={total} wins={wins} pushes={pushes} "
        f"losses={losses} invalid={invalid}"
    )
    print(
        f"Percentages: wins={win_pct:.2f}% pushes={push_pct:.2f}% "
        f"losses={loss_pct:.2f}% invalid={invalid_pct:.2f}%"
    )
    if total > 0:
        print(
            f"Probe means: p_hit={hit_prob_sum / total:.4f} "
            f"p_stay={stay_prob_sum / total:.4f}"
        )


if __name__ == "__main__":
    main()
