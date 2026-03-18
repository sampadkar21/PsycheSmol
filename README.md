# 🧠 PsycheSmol-135M-DPO

[![Model](https://img.shields.io/badge/Model-SmolLM2--135M-blue)](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

**PsycheSmol-135M-DPO** is a specialized, lightweight language model aligned for **therapeutic conversation and mental health support**. By leveraging Direct Preference Optimization (DPO) on top of a Supervised Fine-Tuning (SFT) foundation, this model is designed to provide empathetic, non-judgmental, and safe responses while avoiding the "robotic" or overly prescriptive tone of general-purpose LLMs.

---

A highly empathetic, safe, and conversational AI mental health assistant. This project demonstrates an end-to-end Generative AI alignment pipeline, taking a compact foundational model (`HuggingFaceTB/SmolLM2-135M-Instruct`) and aligning it to complex therapeutic guidelines using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

🔗 **[Access the Merged Model Weights Here](#)** *(Replace with your Hugging Face or Drive link)*

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Training Pipeline & Notebooks](#training-pipeline--notebooks)
- [Output Comparison: Why DPO Wins](#output-comparison-why-dpo-wins)
- [Usage & Inference](#usage--inference)
- [Citations](#citations)

## 🔍 Project Overview
General-purpose LLMs often fail in mental health contexts by offering unsolicited life advice, acting dismissive, or attempting medical diagnoses. The goal of this project is to align an LLM to behave like a compassionate counselor. 

Using the `Psychotherapy-LLM/PsychoCounsel-Preference` dataset, the model was trained to prioritize 7 key therapeutic metrics: **Empathy, Relevance, Clarity, Safety, Exploration, Autonomy, and Staging**. 

## 🛠️ Training Pipeline & Notebooks
The repository contains three main Jupyter Notebooks that document the entire training and evaluation lifecycle:

1. **`sft-finetuning.ipynb` (Supervised Fine-Tuning)**
   - Formats the dataset and applies LoRA (Low-Rank Adaptation) to fine-tune the base SmolLM2 model purely on the "Chosen" (high-quality) therapeutic responses. 
   - Establishes the foundational tone and structure for empathetic conversation.
2. **`dpo-ft.ipynb` (Direct Preference Optimization)**
   - Calculates a custom, weighted `chosen_score` and `rejected_score` across the 7 therapeutic metrics to define preference margins.
   - Utilizes the `trl` DPOTrainer to heavily penalize prescriptive/unsafe advice and reward exploratory, validating responses. 
   - Merges the final adapters into the base model.
3. **`inference.ipynb` (Evaluation & Comparison)**
   - Loads the Base, SFT, and DPO models side-by-side.
   - Runs a comparative qualitative analysis across complex psychological prompts to evaluate alignment success.

## 📊 Output Comparison: Why DPO Wins

To understand the impact of the alignment pipeline, we evaluated the models on a complex scenario involving parental guilt and burnout. 

**User Query:**
> *"I feel like I'm failing as a parent because I don't have enough time or energy to spend with my kids. Between my job, the commute, and household chores, by the time I can actually sit down with them, I have zero patience left. I see other parents doing all these fun, engaging activities, and it just makes me feel so guilty—like I'm depriving them of a good childhood. I want to be present and happy with them, but I'm just so drained all the time."*

### 🏆 The Best Output: DPO Model
> *"It's completely understandable that you're feeling this way, and it's common for parents to feel this way. Being absent from your kids' lives can be draining... First, let's acknowledge that everyone's experience is unique, but it's important to recognize that taking care of your family's needs and responsibilities is not selfish, but rather necessary.*
>
> *Rather than getting frustrated, let's focus on the present moment. What are some aspects of your daily routine that you can realistically incorporate into your morning and afternoon, without sacrificing your love and happiness for your kids? ... Remember, taking care of your children's needs doesn't mean you need to sacrifice your own relationships. You're doing the best you can with what you have, and that's what matters most."*

### 🔍 How the Models Differ:
* **Base Model (`SmolLM2-135M-Instruct`)**: Responds like a generic life coach. It immediately jumps to unsolicited time-management advice (*"Ask yourself what you can do to create a better balance"*), which lacks deep emotional validation and can feel dismissive to a struggling parent.
* **SFT Model**: Shows improved therapeutic structure but struggles with nuance. It attempts to be empathetic but uses clunky phrasing (*"remember that you're not doing a perfect job in your role"*) which can inadvertently amplify the user's guilt. It also rushes into prescribing physical habits like deep breathing.
* **DPO Model (Final)**: Exhibits superior therapeutic alignment. It strongly validates the user's guilt without judgment, gently reframes their hard work as a necessary act of care (*"not selfish, but rather necessary"*), and uses open-ended questions to grant the user autonomy in finding their own solutions. 

## 🚀 Usage & Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "your-username/SmolCounsel-135M-DPO" # Replace with your link

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

SYSTEM_PROMPT = "You are a compassionate AI mental health assistant. Respond with empathy, provide supportive guidance, and encourage healthy coping strategies without giving medical diagnoses."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I'm feeling really overwhelmed with everything lately."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=1.0, top_p=0.9, do_sample=True)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```
```tex
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
  journal = {Hugging Face Repository},
  howpublished = {\url{[https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference](https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference)}}
}
```
