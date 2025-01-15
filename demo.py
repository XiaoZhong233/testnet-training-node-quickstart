import os
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=training_args.lora_rank,  # LoRA rank, reduce to save memory
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Training configuration
    training_config = SFTConfig(
        per_device_train_batch_size=1,  # Reduce batch size to 1
        gradient_accumulation_steps=16,  # Increase accumulation steps
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,  # Use bf16 for mixed precision
        optim="paged_adamw_8bit",  # Lightweight optimizer
        logging_steps=20,
        output_dir="outputs",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=min(context_length, 1024),  # Reduce sequence length if necessary
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        trust_remote_code=True,
        device_map="auto",  # Auto map devices for distributed training
        load_in_8bit=True,  # Enable 8-bit quantization
        token=os.environ.get("HF_TOKEN"),
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Create dataset
    dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=min(context_length, 1024),
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=min(context_length, 1024)),
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model("outputs")

    # Remove checkpoint folders to save space
    os.system("rm -rf outputs/checkpoint-*")

    print("Training Completed.")


def print_model_structure(model_id: str):
    """Print model structure to help identify target_modules for LoRA"""
    from transformers import AutoModelForCausalLM

    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\nModel structure:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Focus on linear layers
            print(f"Linear Layer: {name} -> Shape: {module.weight.shape}")


if __name__ == "__main__":
    # Example LoRA training arguments
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.1,
    )

    # Set model ID and context length
    model_id = "microsoft/Phi-3-small-8k-instruct"
    context_length = 2048

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, context_length=context_length, training_args=training_args
    )
