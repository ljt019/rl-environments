import verifiers as vf
from environments.rust_review.rust_review import load_environment

############## Training Config ##############

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVIEW_APPLICATOR_MODEL = "openai/gpt-4.1-nano"

RUN_NAME = "rust-review-grpo-250925"

#############################################

env = load_environment(review_applicator_model=REVIEW_APPLICATOR_MODEL)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
