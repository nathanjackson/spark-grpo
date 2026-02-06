import argparse

import torch
import transformers


SYSTEM_PROMPT = "You are a helpful assistant."


def generate_reply(messages, tokenizer, model, temperature):
    device = model.device
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
        )

    response_ids = outputs[0, encoding.input_ids.shape[1]:]
    response = "".join(tokenizer.batch_decode(response_ids, skip_special_tokens=True))
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--model-id", default="ibm-granite/granite-4.0-350m")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "ibm-granite/granite-4.0-350m"
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_source = args.model_path or args.model_id
    model = transformers.AutoModelForCausalLM.from_pretrained(model_source).to(device)
    model.eval()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Interactive chat. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print("")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(messages, tokenizer, model, args.temperature)
        messages.append({"role": "assistant", "content": reply})
        print(f"Model: {reply}")


if __name__ == "__main__":
    main()
