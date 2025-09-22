import verifiers as vf
from environments.rust_cargo.rust_cargo import load_environment

############## Training Config ##############

MODEL_NAME = "ljt019/Qwen2.5-Coder-1.5B-Instruct-rust"

RUN_NAME = "rust-review-grpo-230825"

#############################################

env = load_environment()

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
