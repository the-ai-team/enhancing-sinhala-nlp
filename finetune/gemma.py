base_model_name = "google/gemma-2-2b" # model that we're fine-tuning
model_name = "sihala-gemma-2b" # name of the fine-tuned model
dataset_name = "0xAIT/sinhala-flan" # dataset to fine-tune on
subset_name = "cot_fsopt"

## Training Arguments - mostly copied from https://github.com/TeluguLLMLabs/Indic-gemma-7b-Navarasa/blob/main/training.ipynb
output_dir = f"models/{model_name}"


# Number of training epochs
num_train_epochs = 4

max_seq_length =1024 

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False 

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

def format_instructions(sample):
  outputs = []

  for i in range(len(sample['Translated Target'])):
    outputs.append(prompt_template.format(sample['Translated Input'][i], sample['Translated Target'][i])+tokenizer.eos_token)
    
  return outputs

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
dataset = load_dataset(dataset_name, subset_name)[subset_name]

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
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to,
)

trainer = SFTTrainer(
    model = base_model,
    train_dataset = dataset,
    max_seq_length = max_seq_length,
    packing = packing,
    args = training_arguments,
    formatting_func=format_instructions,
)

trainer.train()