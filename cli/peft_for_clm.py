import argparse
import transformers
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from fmengine.dataloader.jsonl_loader import get_jsonl_dataset

def train(args):
    import os
    os.environ["WANDB_PROJECT"] = args.project_name
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        # just use default parameters
        # r=args.lora_rank,
        # lora_alpha=args.lora_alpha,
        # lora_dropout=args.lora_dropout,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_len,
        use_fast=True,
    )
    """
    handle special tokens
    """
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.bos_token

    tokenizer.padding_side = "left"
    model.print_trainable_parameters()

    train_dataset = get_jsonl_dataset(
        args.data_path,
        tokenizer=tokenizer,
        args={
            "seq_length": args.max_seq_len,
            "batch_size": args.micro_batch_size,
        },
    )
    train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
    print(model)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            save_strategy='epoch',
            logging_steps=1,
            output_dir=args.output_dir,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default=".cache/peft")
    parser.add_argument("--wandb-run-name", type=str, required=True)
    parser.add_argument("--project-name", type=str, default="peft")
    args = parser.parse_args()
    train(args)
