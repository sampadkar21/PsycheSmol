# 🧠 PsycheSmol-135M-DPO

[![Model](https://img.shields.io/badge/Model-SmolLM2--135M-blue)](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![Library](https://img.shields.io/badge/Library-TRL-orange)](https://github.com/huggingface/trl)

**PsycheSmol-135M-DPO** is a specialized, lightweight language model aligned for **therapeutic conversation and mental health support**. By leveraging Direct Preference Optimization (DPO) on top of a Supervised Fine-Tuning (SFT) foundation, this model provides empathetic, non-judgmental, and safe responses while avoiding the "robotic" or overly prescriptive tone of general-purpose LLMs.

---

A highly empathetic, safe, and conversational AI mental health assistant. This project demonstrates an end-to-end Generative AI alignment pipeline, taking a compact foundational model (`HuggingFaceTB/SmolLM2-135M-Instruct`) and aligning it to complex therapeutic guidelines using SFT and DPO via the **Hugging Face TRL** library.

🔗 **[Access the Merged Model Weights Here](https://drive.google.com/drive/folders/1sXwU9dh9Yrc5pw_0OLm1v7VzrRepM7kc?usp=sharing)**

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Training Pipeline & Notebooks](#training-pipeline--notebooks)
- [DPO Alignment Metrics](#dpo-alignment-metrics)
- [Output Comparison: Why DPO Wins](#output-comparison-why-dpo-wins)
- [Usage & Inference](#usage--inference)
- [Citations](#citations)

## 🔍 Project Overview
General-purpose LLMs often fail in mental health contexts by offering unsolicited life advice, acting dismissive, or attempting medical diagnoses. The goal of this project is to align an LLM to behave like a compassionate counselor. 

Using the `Psychotherapy-LLM/PsychoCounsel-Preference` dataset, the model was trained to prioritize 7 key therapeutic metrics: **Empathy, Relevance, Clarity, Safety, Exploration, Autonomy, and Staging**. 

## 🛠️ Training Pipeline & Notebooks
The repository contains three main Jupyter Notebooks documenting the lifecycle:

1. **`sft-finetuning.ipynb`**: Applies LoRA (Low-Rank Adaptation) to fine-tune the base model on high-quality therapeutic responses.
2. **`dpo-ft.ipynb`**: Utilizes the `trl` **DPOTrainer** to penalize prescriptive advice and reward exploratory, validating responses using a custom weighted preference score.
3. **`inference.ipynb`**: Runs a comparative qualitative analysis across the Base, SFT, and DPO versions.

---

## 📊 DPO Alignment Metrics
The following metrics demonstrate the model's progress during the Direct Preference Optimization phase. By training on preference pairs, the model successfully learns to distinguish high-quality therapeutic responses from subpar ones.

<p align="center">
  <img src="image_520dc4.png" alt="DPO Training Reward Metrics" width="800">
</p>

### **Key Takeaways:**
* **Reward Margin:** The increasing gap between the "Chosen" and "Rejected" rewards indicates the model is successfully learning the preference boundary.
* **Reward Accuracy:** Reaching nearly 100% accuracy early on shows that the preference signal in the dataset was strong and the model successfully captured the desired therapeutic tone.
* **Chosen Rewards:** The stability of the "Chosen" rewards around the -2.0 mark (relative to the reference model) shows controlled alignment without catastrophic forgetting.

---

## 🏆 Output Comparison: Why DPO Wins

To understand the impact of the alignment pipeline, we evaluated the models on a complex scenario involving parental guilt and burnout.

**User Query:**
> *"I feel like I'm failing as a parent because I don't have enough time or energy to spend with my kids... I want to be present and happy with them, but I'm just so drained all the time."*

### 🔍 Analysis of Versions:
* **Base Model (`SmolLM2-135M-Instruct`)**: Responds like a generic life coach. It immediately jumps to unsolicited time-management advice, which lacks emotional validation and can feel dismissive.
* **SFT Model**: Shows improved structure but struggles with nuance. It attempts empathy but uses clunky phrasing (e.g., *"remember that you're not doing a perfect job"*) which can inadvertently amplify the user's guilt.
* **DPO Model (Final)**: Exhibits superior therapeutic alignment. It strongly validates the user's guilt without judgment and uses open-ended questions to grant the user autonomy.

---

## 🚀 Usage & Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "your-username/PsycheSmol-135M-DPO" 

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

SYSTEM_PROMPT = "You are a compassionate AI mental health assistant. Respond with empathy and supportive guidance."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I'm feeling really overwhelmed with everything lately."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```
```bibtex
@misc{allal2025smollm2,
      title={SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Leandro von Werra and Thomas Wolf},
      year={2025},
      eprint={2502.02737},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{psychocounsel2024,
  author = {Psychotherapy-LLM},
  title = {PsychoCounsel-Preference Dataset},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{[https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference](https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference)}}
}
```
