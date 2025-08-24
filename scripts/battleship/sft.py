import verifiers as vf
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer  # type: ignore

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 8 examples/sft.py
"""

############## Training Config ##############

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "ljt019/battleship-multiturn-1200"

RUN_NAME = "battleship-sft-220825"

HUB_MODEL_ID = "ljt019/Qwen3-4B-Instruct-bs-sft-0825"
PUSH_TO_HUB = True

OUTPUT_DIR = "data/checkpoints"

# Trainer args

MAX_LENGTH = 11264  # Reduce sequence length to save memory
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Keep at 1 since we're already at minimum
GRADIENT_ACCUMULATION_STEPS = 4  # Increase to maintain effective batch size
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.1

#############################################


# convenience function for FA2 initialization
model, tokenizer = vf.get_model_and_tokenizer(
    MODEL_NAME,
    use_liger=False,
)
dataset = load_dataset(DATASET_NAME, split="train")

tok_counts = []
for row in dataset:
    # count tokens in messages (which contains the full conversation)
    messages = row["prompt"] + row["completion"]  # type: ignore
    toks = tokenizer.apply_chat_template(messages, tokenize=True)
    tok_counts.append(len(toks))

# tok count stats
print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=MAX_LENGTH,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    run_name=RUN_NAME,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,  # type: ignore
)

trainer.train()
