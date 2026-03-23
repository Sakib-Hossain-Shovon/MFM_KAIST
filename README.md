# 🌋 MFM_KAIST: LLaVA-Inspired MFM with LlamaGen-Based Image Decoder

This repository contains my modified LLaVA-inspired MFM project for anomaly-aware multimodal evaluation.  
The main extension is the integration of a **LlamaGen-based image decoder** between the **vision tower** and the **MM projector** inside the LLaVA pipeline.

---

## Contents
- [Install](#install)
- [Project Structure](#project-structure)
- [Train](#train)
- [Evaluation](#evaluation)
- [Background Evaluation](#background-evaluation)
- [Verification](#verification)
- [Notes](#notes)

---

## Install

### 1. Clone this repository
```bash
git clone https://github.com/Sakib-Hossain-Shovon/MFM_KAIST.git
cd MFM_KAIST/MFM_test


2. Create environment and install package

conda create -n mfm python=3.11 -y
conda activate mfm
pip install --upgrade pip
pip install -e .

3. Install additional packages for training

pip install -e ".[train]"
pip install flash-attn --no-build-isolation


4. Upgrade to latest code base

git pull
pip install -e .

If you encounter import errors after upgrade, try:


Train

Run the training script:
bash scripts/v1_5/run_mfm.sh



Evaluation

Run the evaluation script:

bash run_eval_anyres.sh


Notes
The project uses a custom wrapper around official pretrained LlamaGen modules.
Official pretrained modules include the VQ model, GPT model, and codebook-related components.
Newly introduced adaptor layers are used only for compatibility with the LLaVA feature pipeline.
To avoid NaN caused by disabled default torch initialization, newly added wrapper/adaptor layers are manually initialized after decoder construction.
If performance drops initially, additional fine-tuning of the wrapper/adaptor layers may be required.
