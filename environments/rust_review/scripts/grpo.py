import verifiers as vf

vf.setup_logging(level="DEBUG")
from transformers import TrainerCallback

from environments.rust_review.rust_review import load_environment


class PushToHubOnSaveCallback(TrainerCallback):
    def __init__(self, trainer: vf.GRPOTrainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.trainer.push_to_hub(
                commit_message=f"checkpoint-{state.global_step}",
                blocking=True,
                max_shard_size="15GB",
            )
        return control


############## Training Config ##############

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
REVIEW_APPLICATOR_MODEL = "openai/gpt-4.1-nano"

RUN_NAME = "rust-review-grpo-270925"

#############################################

env = load_environment(review_applicator_model=REVIEW_APPLICATOR_MODEL)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)
args.max_steps = 2500
args.save_steps = 25
args.save_only_model = False
args.push_to_hub = True
args.hub_model_id = "ljt019/Qwen2.5-Coder-7B-rust-review"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.add_callback(PushToHubOnSaveCallback(trainer))

trainer.train()
