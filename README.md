# CSE291A – ECUA Project

This repository contains my Edge-Capable Conversational User Agent (ECUA) for CSE291A Phase 2.  
The current focus is integrating **OSWorld** desktop tasks and **Agent_S** GUI-control components into a unified, CPU-only runnable pipeline.

---

## Repository Structure

    CSE291A_ECUA3/
    ├── evals/                      # Evaluation logs and outputs
    ├── src/
    │   └── ecua/
    │       ├── bbon/               # Adapted from OSWorld (third-party)
    │       ├── desktop_env/        # Adapted from OSWorld (third-party)
    │       ├── evaluation_ex.../   # Adapted from OSWorld (third-party)
    │       ├── gui_agents/         # Adapted from Agent_S (third-party)
    │       ├── lib_run_single...   # My integration / ECUA wrapper
    │       ├── run.sh              # Entry point for running tasks
    │       └── agent_test.ipynb    # Our testing & experiments
    └── requirements.txt

**Note:** The folders marked as *third-party* are copied and minimally adapted from OSWorld / Agent_S to make the project self-contained.  
All new ECUA logic will be added outside those directories.

---

## Tested Hardware

We tested our implementation on an **AMD CPU (AMD Ryzen 7 7900x)** with:
- 8 cores
- 32GB RAM

---

## Environment Setup

We provide two approaches for setting up the environment:

### Local Approach (Highly Recommended)

#### 2.1 Create Python environment

    conda create -n ecua python=3.11 -y
    conda activate ecua
    pip install -r requirements.txt

#### 2.2 Download Ollama and LM Studio

Download and install [Ollama](https://ollama.ai/) and [LM Studio](https://lmstudio.ai/) on your local machine.

#### 2.3 Download the models

Download the main model and the grounding model:

    ollama pull qwen2.5vl:32b

Manually download `bartowski/UI-TARS-7B-DPO-GGUF@q4_k_s` from LM Studio GUI.

#### 2.4 Run the code

    python src/run.py

This will replicate our works in `results/`.

---

### AWS Approach (Slower than local approach)

#### 3.1 Create Python environment

    conda create -n ecua python=3.11 -y
    conda activate ecua
    pip install -r requirements.txt

#### 3.2 Download Ollama and configure LM Studio on headless device

**Download Ollama:**

Follow the [Ollama installation guide](https://ollama.ai/) for Linux.

**Configure LM Studio (headless):**

Follow this guide: [Running headless LM-Studio on Ubuntu](https://run.tournament.org.il/running-headless-lm-studio-on-ubuntu/). (Remember to download the latest appimage from [Download LM Studio](https://lmstudio.ai/download))

#### 3.3 Download the models

Same as step 2.3:

    ollama pull qwen2.5vl:32b

Manually download `bartowski/UI-TARS-7B-DPO-GGUF@q4_k_s` from LM Studio GUI.

#### 3.4 Run the code

Same as step 2.4:

    python src/run.py

This will replicate our works in `results/`.

---

## Running the Jupyter Notebook

### 1. Install Jupyter

    pip install jupyter

### 2. Start the notebook server (Optional if you are using a Code Editor)

    jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser

You may need to either open port `8888` in your AWS security group **or** use SSH port forwarding, for example:

    ssh -i <key>.pem -L 8888:localhost:8888 ubuntu@<EC2-IP>

### 3. Open the notebook in your browser

In the Jupyter UI, open:

    src/ecua/agent_test.ipynb

---

## Notes

- This repository currently focuses on **integration** of OSWorld + Agent_S into one runnable environment on CPU-only servers.
- Core ECUA reasoning, planning, and model-based improvements will be developed on top of this structure.