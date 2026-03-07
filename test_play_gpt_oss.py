import argparse
import re

import torch
import transformers
import openai_harmony as H
from peft import PeftModel

import blackjack


harmony_encoding = H.load_harmony_encoding(H.HarmonyEncodingName.HARMONY_GPT_OSS)


def _format_visible_state(game):
    state = game.get_state()
    return {
        "your_hand": state["player_hand"],
        "dealer_hand": ["Hidden"] + state["dealer_hand"][1:],
    }


def _parse_harmony_response(token_ids, tokenizer):
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    thinking = ""
    action = ""
    try:
        parsed = harmony_encoding.parse_messages_from_completion_tokens(
            token_ids, role=H.Role.ASSISTANT
        )
        if parsed and parsed[0].content:
            thinking = parsed[0].content[0].text.strip()
        if len(parsed) > 1 and parsed[1].content:
            action = parsed[1].content[0].text.strip()
    except (IndexError, H.HarmonyError):
        pass

    if not action:
        match = re.search(r"\b(HIT|STAY)\b", raw_text.upper())
        if match:
            action = match.group(1)
        else:
            action = raw_text

    if not thinking:
        thinking = raw_text

    return action.strip().upper(), thinking


def _model_action(messages, tokenizer, model, temperature):
    device = model.device
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0.0

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

    response_ids = outputs.sequences[0, encoding.input_ids.shape[1]:]
    action, thinking = _parse_harmony_response(response_ids, tokenizer)
    return action, thinking


def play_game(game_id, tokenizer, model, temperature):
    system_msg = {
        "role": "system",
        "content": (
            "You are playing blackjack. At each step you'll be given the state "
            "of the game. Provide your reasoning in the thinking channel, then "
            "respond with HIT or STAY only."
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

        action, thinking = _model_action(messages, tokenizer, model, temperature)
        messages.append({"role": "assistant", "content": action, "thinking": thinking})

        print(f"  Step {step}: state={visible_state} action={action}")
        print(f"    Thinking: {thinking}")

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
    parser.add_argument(
        "--model-id", default="axolotl-ai-co/gpt-oss-20b-dequantized"
    )
    parser.add_argument("--adapter-path")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    model_source = args.model_path or args.model_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("gpt-oss-20b test requires CUDA with bitsandbytes.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
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
