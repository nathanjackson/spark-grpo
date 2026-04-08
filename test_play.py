import argparse
import os
import random

import torch
import transformers

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


def _is_adapter_checkpoint(path):
    return path is not None and os.path.exists(os.path.join(path, "adapter_config.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if _is_adapter_checkpoint(args.model_path):
        raise ValueError(
            f"{args.model_path} looks like a LoRA adapter checkpoint. "
            "Use test_play_qlora.py for adapter-based checkpoints."
        )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_source = args.model_path or args.model_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_source).to(device)
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
