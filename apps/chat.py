import transformers


def to_chatml(prompt):
    return f"<human>: {prompt}<|endoftext|><assistant>:"


def chat(model_path: str, system_prompt: str):
    print("[fmengine] loading model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    model.to("cuda")
    print("[fmengine] models loaded")
    dialogs = ""
    if system_prompt:
        dialogs = "<system>: " + system_prompt
    while True:
        user_input = input("User: ")
        if user_input == "\exit":
            break
        if user_input == "\reset":
            dialogs = ""
            continue
        model_input = dialogs + to_chatml(user_input)
        input_ids = tokenizer.encode(model_input, return_tensors="pt")
        input_ids = input_ids.to("cuda")
        output_ids = model.generate(
            input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        dialogs += (
            output.replace(model_input, "").split("<|endoftext|>")[0]
            + "<|endoftext|>\n"
        )
        printed_output = output.replace(model_input, "").split("<|endoftext|>")[0]
        print(f"System: {printed_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Location of model")
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt",
        default="You are a friendly and helpful chatbot built by the fmengine, and you are here to help the human.",
    )
    args = parser.parse_args()
    chat(args.model_path, args.system_prompt)
