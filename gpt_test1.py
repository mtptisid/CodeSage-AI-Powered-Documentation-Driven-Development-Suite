
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from pdfminer.high_level import extract_text
from docx import Document

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding issues

# Function to extract text from different file formats
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        return extract_text(file_path)  # Extract text from PDF
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])  # Extract text from DOCX
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()  # Extract text from TXT
    else:
        return ""

# Directory in Google Drive where your documents are stored
data_dir = "/content/drive/MyDrive/my_train_data"
all_texts = []
for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    text = extract_text_from_file(file_path)
    if text:
        all_texts.append(text)

# Convert text into dataset
dataset = Dataset.from_dict({"text": all_texts})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for better training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We are fine-tuning for causal LM (GPT-2)
)
