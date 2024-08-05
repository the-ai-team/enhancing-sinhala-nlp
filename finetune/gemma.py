base_model_name = "google/gemma-2-2b" # model that we're fine-tuning
model_name = "sihala-gemma-2b" # name of the fine-tuned model
dataset_name = "0xAIT/sinhala-flan" # dataset to fine-tune on
subset_name = "cot_fsopt"

## Training Arguments - mostly copied from https://github.com/TeluguLLMLabs/Indic-gemma-7b-Navarasa/blob/main/training.ipynb
output_dir = f"models/{model_name}"


# Number of training epochs
num_train_epochs = 4

max_seq_length = 2048

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 64

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 1.0

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 10000

# Log every X updates steps
logging_steps = 10

# Pack multiple short examples in the same input sequence to increase efficiency and make training 5x faster for short sequences.
packing = False 

# text field in dataset
dataset_text_field = "text"

# dataset para,
dataset_num_proc = 2

# Load the entire model on the GPU 0
# device_map = {"": 0}
device_map = "auto"

# monitoring
report_to = "wandb"

logging_strategy = "steps"
save_strategy = "steps"

prompt_template = """
### Input:
{}

### Output:
{}"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login
import torch
import wandb
import os

login(token=os.environ['HF_TOKEN'])
wandb.login(key=os.environ['WANDB_KEY'])

os.environ["WANDB_PROJECT"] = "sinhala-aya"
os.environ["WANDB_LOG_MODEL"] = "false"  # don't log model checkpoints

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

def formatting_prompts_func(examples):
    inputs       = examples["Translated Input"]
    outputs      = examples["Translated Target"]
    texts = []
    for input, output in zip(inputs, outputs):
        if input is None:
            input = ""
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt_template.format(input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

def format_dataset():
    dataset = load_dataset(dataset_name, subset_name)[subset_name]
    formatted_dataset = dataset.map(formatting_prompts_func)
    return formatted_dataset

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
dataset = format_dataset()

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to,
)

trainer = SFTTrainer(
    model = base_model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = dataset_text_field,
    max_seq_length = max_seq_length,
    dataset_num_proc = dataset_num_proc,
    packing = packing,
    args = training_arguments,
)

trainer.train()