from datetime import datetime

import verifiers as vf
from environments.rust_review.rust_review import load_environment

############## Training Config ##############

MODEL_NAME = "ljt019/Qwen2.5-Coder-7B-rust-review"
REVIEW_APPLICATOR_MODEL = "qwen/qwen3-coder-30b-a3b-instruct"

RUN_NAME = "rust-review-grpo-7b"

#############################################

env = load_environment(review_applicator_model=REVIEW_APPLICATOR_MODEL)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=f"{RUN_NAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}")
args.max_steps = 2500
args.save_steps = 25
args.save_only_model = False
args.push_to_hub = True
args.save_total_limit = 2
args.hub_model_id = "ljt019/Qwen2.5-Coder-7B-rust-review"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train(resume_from_checkpoint=False)
