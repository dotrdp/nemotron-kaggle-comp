import polars as pl

# train = pl.read_csv('/train.csv')
import site

cutlass_pkg_path = "/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)
import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


# import subprocess
#
# subprocess.run("uv pip freeze > /kaggle/working/requirements.txt",
#                shell=True,
#                check=True)
#
# Configuration
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working/output"
LORA_RANK = 2  # Can be set to a maximum of 32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # device_map="auto",
    dtype=torch.bfloat16)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model loaded successfully.")

# Initialize LoRA Adapter
print(f"Initializing LoRA adapter with rank={LORA_RANK}...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# train model
from datasets import Dataset
from trl.experimental.gfpo import GFPOConfig, GFPOTrainer

# dummy group filter to scores the completions based on its indice in group


train_file = pl.read_csv("/datasets/train.csv")
dataset = Dataset.from_polars(train_file)


def proccess_sample(example):
    prompt = [{"role": "user", "content": f"{example["prompt"]}"+'\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answ}`'}]
    # prompt = [
    #     {
    #         "role": "system",
    #         "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": f"{example["prompt"]}"+'\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answ}`'},
    #         ],
    #     },
    # ]
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {"id": example['id'], "prompt": prompt, "answer": example['answer']}

dataset = dataset.map(proccess_sample)

train, evaldat = dataset.train_test_split()

import re


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return 'NOT_FOUND'

    # Search for boxed answer
    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def reward_func(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer, strict=False):
        raw = completion[-1]["content"]

        # detect form *yes* or *no*
        guess = extract_final_answer(raw)

        reward = 0.0

        if guess is None:
            reward -= 0.5  # invalid format
        elif guess == ans:
            reward += 0.6  # correct under required format
        else:
            reward -= 1.0  # wrong answer

        rewards.append(reward)

    return rewards

config = GFPOConfig(
    # num_generations=8,
    max_completion_length=7680,
    chat_template_kwargs={"enable_thinking": True},

    # loss_type="vespo",

    top_p=1.0,
    # temperature=0.0,


    gradient_checkpointing=False,

    # per_device_train_batch_size=4,
    # num_remains_in_group=2,
    bf16=True,
)

class GroupFilter:
    def __call__(self, group_completions, group_rewards, **kwargs):
        group_scores = []
        for completions, rewards in zip(group_completions, group_rewards):
            scores = [float(i) for i in range(len(completions))]
            group_scores.append(scores)
        return group_scores


trainer = GFPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=config,
    train_dataset=train,
    eval_dataset=evaldat,
    group_filter_func=GroupFilter(),
    # peft_config=lora_config
)

trainer.train()
trainer.save_model(OUTPUT_DIR)




# Save Adapter
# print(f"Saving adapter to {OUTPUT_DIR}...")
# model.save_pretrained(OUTPUT_DIR)

import subprocess

subprocess.run("zip -m submission.zip *",
               shell=True,
               check=True,
               cwd=OUTPUT_DIR)

print('Done.')
