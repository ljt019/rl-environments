import verifiers as vf
from environments.rust_review.rust_review import load_environment

############## Training Config ##############

MODEL_NAME = "ljt019/Qwen2.5-Coder-1.5B-Instruct-rust"
CODER_MODEL = "qwen/qwen3-coder-30b-a3b-instruct"

RUN_NAME = "rust-review-grpo-230825"

#############################################

env = load_environment(coder_model=CODER_MODEL)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
