import os
from dataclasses import dataclass
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    lora_dropout: int


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 训练配置
    training_config = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 强制设置模型为 bfloat16 类型
        attn_implementation="flash_attention_2",
    )

    # 确保模型参数是浮点类型并启用梯度计算
    for param in model.parameters():
        if param.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(f"Parameter {param} has unsupported dtype {param.dtype}")
        param.requires_grad = True  # 确保启用梯度计算

    # 创建数据集
    dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")



def print_model_structure(model_id: str):
    """打印模型结构，帮助找到可用的 target_modules"""
    from transformers import AutoModelForCausalLM
    import torch

    # 加载模型
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 打印所有命名模块
    print("\nModel structure:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # 我们通常对线性层感兴趣
            print(f"Linear Layer: {name} -> Shape: {module.weight.shape}")


if __name__ == "__main__":
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    # Set model ID and context length
    model_id = "Qwen/Qwen1.5-0.5B"
    context_length = 2048

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, context_length=context_length, training_args=training_args
    )

    # You can call this to print the model structure
    model_id = "microsoft/Phi-3-small-8k-instruct"  # 或其他模型ID
    print_model_structure(model_id)
