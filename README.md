# Multimodal VQA with ABO Dataset

This repository contains all scripts, notebooks and datasets used for the AIM825 Mini Project 2 on Multimodal Visual Question Answering (VQA) with Amazon Berkeley Objects (ABO) dataset.

## 👥 Team Members

* **Vansh Sinha** – IMT2022122
* **Ishan Singh** – IMT2022124

## 📄 Report
- `Report.pdf` – Final project report detailing data curation, model selection, fine-tuning, evaluation, and results.

## 📂 CuratedDatasets

### └── CuratedDatasetForEvaluations
- `SQID_test_generated_vqa.csv` – VQA dataset generated from Shopping Queries Image Dataset (SQID) for evaluating baseline and fine-tuned model performance.

### └── CuratedDatasetForFineTuning
- `RO_generated_vqa.csv` – VQA dataset curated from ABO dataset using Random Oversampling strategy for fine-tuning.
- `S_generated_vqa.csv` – VQA dataset curated from ABO dataset using Custom Sampling strategy for fine-tuning.

## 📂 InferenceScript
- `inference.py` – Script to load final fine-tuned model and generate predictions on new input.
- `requirements.txt` – Python dependencies needed to run the inference script.

## 📂 Notebooks

### 🛠 Dataset Curation
- `vrmp2-dataset-curation-1.ipynb` – Initial ABO dataset preprocessing and sampling images based on 2 different sampling strategies.
- `vrmp2-dataset-curation-2a.ipynb` – VQA dataset generation using Random Oversampling.
- `vrmp2-dataset-curation-2b.ipynb` – VQA dataset generation using Custom Sampling.
- `vrmp2-sqid-test-dataset-preprocessing.ipynb` – SQID test dataset download and preprocessing.
- `vrmp2-sqid-test-dataset-curation.ipynb` – VQA dataset generation using SQID test dataset.

### 🧠 Baseline Evaluation
- `vrmp2-blip-baseline.ipynb` – BLIP model baseline evaluation.
- `vrmp2-paligemma-baseline.ipynb` – PaliGemma model baseline evaluation.
- `vrmp2-vilt-baseline.ipynb` – VILT model baseline evaluation.
- `vrmp2-baseline-evaluations.ipynb` – Summarizes and compares metrics across all baseline models.

### 🧪 Fine-Tuning and Evaluations

#### 🔁 Random Oversampled VQA Dataset
- `vrmp2-ro-blip-lora-finetuning.ipynb` – BLIP fine-tuning using LoRA.
- `vrmp2-ro-blip-finetuned-predictions.ipynb` – Predictions from fine-tuned BLIP.
- `vrmp2-ro-paligemma-lora-finetuning.ipynb` – PaliGemma fine-tuning using LoRA.
- `vrmp2-ro-paligemma-finetuned-predictions.ipynb` – Predictions from fine-tuned PaliGemma.
- `vrmp2-ro-finetuned-evaluations.ipynb` – Evaluation of models fine-tuned on Random Oversampled dataset.

#### 🔁 Custom Sampled VQA Dataset
- `vrmp2-s-blip-lora-finetuning.ipynb` – BLIP fine-tuning using LoRA.
- `vrmp2-s-blip-finetuned-predictions.ipynb` – Predictions from fine-tuned BLIP.
- `vrmp2-s-paligemma-lora-finetuning.ipynb` – PaliGemma fine-tuning using LoRA.
- `vrmp2-s-paligemma-finetuned-predictions.ipynb` – Predictions from fine-tuned PaliGemma.
- `vrmp2-s-finetuned-evaluations.ipynb` – Evaluation of models fine-tuned on Custom Sampled dataset.

