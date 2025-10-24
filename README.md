# idefics2-lora-finetune

# Idefics2 LoRA & QLoRA Fine-Tuning

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains code for **fine-tuning the multimodal model [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b)** using **LoRA** and **QLoRA** approaches.  
It supports text + image inputs and leverages memory-efficient training with **BitsAndBytes** quantization.

---

## 🚀 Features

- Fine-tuning **Idefics2** using **LoRA adapters**.
- Memory-efficient **QLoRA 4-bit quantization** training.
- Custom **DataCollator** for multimodal data (text + images).
- Handles conversational prompts with images using **processor.apply_chat_template**.
- Ready to integrate with **Hugging Face Trainer**.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/bassemdahmani/idefics2-lora-qlora-finetune.git
cd idefics2-lora-qlora-finetune

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 🛠 Requirements
```bash
Python 3.10+
PyTorch 2.1+
Transformers (Hugging Face)
PEFT
BitsAndBytes
Datasets, Accelerate (optional for Trainer)
```
## 📝 Usage
1. Your dataset should contain:
```bash
{
  "image": "<image_path_or_PIL_object>",
  "query": {"en": "Your question about the image"},
  "answers": ["Answer1", "Answer2", ...]
}
```
2. Initialize Processor and DataCollator:
```bash
from transformers import AutoProcessor
from data_collator import MyDataCollator

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
data_collator = MyDataCollator(processor)
```
3. Load model with LoRA/QLoRA:
```bash
from transformers import Idefics2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA config
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```
4. Train with Hugging Face Trainer:
```bash
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()
```

## 📌 Notes:

QLoRA allows training large models on a single GPU by loading in 4-bit precision.

LoRA only trains small adapters, keeping the base model frozen.

The DataCollator ensures text + image sequences are correctly tokenized and aligned for training.




