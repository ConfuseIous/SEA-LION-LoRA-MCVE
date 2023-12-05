import os

# hf cache dir
os.environ["TRANSFORMERS_CACHE"] = "/home/karan/projects/hsd2/sealion/cache"

import torch
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# List all GPUs and their names
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

# -- PREPARE MODEL -- #
base_model_id = "aisingapore/sealion7b"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="right",
    add_eos_token=True,
    trust_remote_code=True,
    )

tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)


# -- PREPARE DATASET -- #
dataset_input_ids = []

def generate_and_tokenize_prompt(data_point):
    full_prompt =  "this is an example prompt"

    input_ids = tokenizer(full_prompt, padding="max_length", truncation=True, max_length=128)
    dataset_input_ids.append(input_ids)

for i in range(1000):
    generate_and_tokenize_prompt(i)

# -- TRAIN -- #
project = "example"
base_model_name = "sealion-7b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset_input_ids,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=1000,
        learning_rate=2.5e-5,
        logging_steps=100,
        bf16=True,
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
