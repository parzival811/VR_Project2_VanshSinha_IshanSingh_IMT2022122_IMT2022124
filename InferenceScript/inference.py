import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import os
import zipfile
import gdown
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from huggingface_hub import login

def download_and_extract_adapter(gdrive_id, output_dir, zip_name="adapter.zip"):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)
    if not os.path.exists(zip_path):
        print("Downloading LoRA adapter from Google Drive...")
        gdown.download(id=gdrive_id, output=zip_path, quiet=False)
    
    extracted_path = os.path.join(output_dir, "extracted_adapter")
    if not os.path.exists(extracted_path):
        print("Extracting adapter...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_path)
    return extracted_path

# Function to load the PaliGemma model with LoRA adapter
def paligemma_load_with_lora(adapter_dir):
    # Login to Hugging Face Hub
    login('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')  

    base_model_name = "google/paligemma-3b-pt-224"

    # Load base model
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        revision="float16",
    ).eval()

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(base_model_name, use_fast=True)

    n = model.num_parameters() / 1e9
    print(f"Model Parameters: {n:.1f}B" if n >= 1 else f"Model Parameters: {n*1000:.0f}M")
    
    return model, processor

# Function to perform inference using PaliGemma
def paligemma_inference(img_path, question_text, model, processor):
    image = Image.open(img_path).convert("RGB")
    text = f"<image> Answer the question in exactly one word: {question_text}"
    model_inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    
    input_len = model_inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Download and extract LoRA adapter
    adapter_dir = download_and_extract_adapter('1qClgmAX3nrtHcAqa3oGgmUFabWhrQbdp', output_dir="adapter_data")

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor
    model, processor = paligemma_load_with_lora(adapter_dir)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            answer = paligemma_inference(image_path, question, model, processor)
        except Exception as e:
            answer = "error"
        # Post-process to get only one word in lowercase
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()

