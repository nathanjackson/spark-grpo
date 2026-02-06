import argparse
import random

import torch
import transformers

import blackjack
from grpo import _build_action_token_sequences, _make_prefix_allowed_tokens_fn


def _format_visible_state(game):
    state = game.get_state()
    return {
        "your_hand": state["player_hand"],
        "dealer_hand": ["Hidden"] + state["dealer_hand"][1:],
    }


def _model_action(messages, tokenizer, model, temperature):
    device = model.device
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = encoding.input_ids.shape[1]

    action_seqs = _build_action_token_sequences(tokenizer)
    prefix_allowed_tokens_fn = _make_prefix_allowed_tokens_fn(
        prompt_len, action_seqs, tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=8,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    response_ids = outputs.sequences[0, prompt_len:]
    action = "".join(tokenizer.batch_decode(response_ids, skip_special_tokens=True))
    return action.strip().upper()


def play_game(game_id, tokenizer, model, temperature):
    system_msg = {
        "role": "system",
        "content": (
            "You are playing blackjack. At each step you'll be given the state "
            "of the game. Respond with HIT or STAY. Do not add commentary."
        ),
    }
    messages = [system_msg]
    game = blackjack.Blackjack()
    game.start_round()

    print(f"Game {game_id}")
    step = 0
    invalid_action = False
    observation = ""

    while not game.get_state()["game_over"]:
        visible_state = _format_visible_state(game)
        messages.append({"role": "user", "content": str(visible_state)})

        action = _model_action(messages, tokenizer, model, temperature)
        messages.append({"role": "assistant", "content": action})

        print(f"  Step {step}: state={visible_state} action={action}")

        if action == "HIT":
            observation = game.hit()
        elif action == "STAY":
            observation = game.stand()
        else:
            invalid_action = True
            observation = f"Invalid action: {action}"
            break

        step += 1

    if invalid_action:
        reward = -2.0
        outcome = observation
    elif "push" in observation.lower():
        reward = 1.0
        outcome = "Push (tie)."
    elif "player wins" in observation.lower():
        reward = 2.0
        outcome = "Player wins."
    else:
        reward = -1.0
        outcome = "Dealer wins."

    final_state = game.get_state()
    print(f"  Final state: {final_state}")
    print(f"  Result: {outcome} reward={reward}")
    print("")
    return reward, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--model-id", default="ibm-granite/granite-4.0-350m")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-350m")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_source = args.model_path or args.model_id
    model = transformers.AutoModelForCausalLM.from_pretrained(model_source).to(device)
    model.eval()

    wins = 0
    pushes = 0
    losses = 0
    invalid = 0

    for i in range(1, args.games + 1):
        reward, _ = play_game(i, tokenizer, model, args.temperature)
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


if __name__ == "__main__":
    main()
