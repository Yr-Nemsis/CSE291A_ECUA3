# CSE291A – ECUA Project

This repository contains my Edge-Capable Conversational User Agent (ECUA) for CSE291A Phase 2.  
The current focus is integrating **OSWorld** desktop tasks and **Agent_S** GUI-control components into a unified, CPU-only runnable pipeline.

---

## Repository Structure

```
CSE291A_ECUA3/
├── ecua/                          # Main ECUA package
│   ├── bbon/                      # Adapted from OSWorld (third-party)
│   ├── desktop_env/               # Adapted from OSWorld (third-party)
│   ├── evaluation_examples/       # Adapted from OSWorld (third-party)
│   ├── gui_agents/                # Adapted from Agent_S (third-party)
│   ├── lib_run_single.py          # Integration / ECUA wrapper
│   ├── run.py                     # Main entry point for running tasks
│   ├── exmaple.ipynb              # Example notebook
│   └── results/                   # Output results directory
├── evals/                         # Evaluation task definitions
├── experiments/                   # Experimental notebooks and outputs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

**Note:** The folders marked as *third-party* are copied and minimally adapted from OSWorld / Agent_S to make the project self-contained.  
All new ECUA logic will be added outside those directories.

---

## Tested Hardware

We tested our implementation on an **AMD CPU (AMD Ryzen 7 7900x)** with:
- 8 cores
- 32GB RAM

---

## Evaluation Results

End-to-end OSWorld-style evaluation on a CPU-only desktop:

| Configuration | Success Rate | WES+ | WES- |
|---------------|--------------|------|------|
| Our ECUA | 30% | 0.48 / 0.31 | -0.01 |

**Metrics:**
- **WES+** measures positive, goal-directed progress for both single-action and grouped-action trajectories
- **WES-** penalizes harmful or regressional actions

---

## Environment Setup

We provide two approaches for setting up the environment:

### Local Approach (Highly Recommended)

#### 1. Create Python environment

```bash
conda create -n ecua python=3.11 -y
conda activate ecua
pip install -r requirements.txt
```

#### 2. Download Ollama and LM Studio

Download and install [Ollama](https://ollama.ai/) and [LM Studio](https://lmstudio.ai/) on your local machine.

#### 3. Download the models

Download the main model and the grounding model:

```bash
ollama pull gemma3:27b
```

Manually download `bartowski/UI-TARS-7B-DPO-GGUF@q4_k_s` from LM Studio GUI.

#### 4. Run the code

```bash
python ecua/run.py
```

This will replicate our works in `ecua/results/`.

---

### AWS Approach (Slower than local approach)

#### 1. Create Python environment

```bash
conda create -n ecua python=3.11 -y
conda activate ecua
pip install -r requirements.txt
```

#### 2. Download Ollama and configure LM Studio on headless device

**Download Ollama:**

Follow the [Ollama installation guide](https://ollama.ai/) for Linux.

**Configure LM Studio (headless):**

Follow this guide: [Running headless LM-Studio on Ubuntu](https://run.tournament.org.il/running-headless-lm-studio-on-ubuntu/). (Remember to download the latest appimage from [Download LM Studio](https://lmstudio.ai/download))

#### 3. Download the models

Same as Local Approach step 3:

```bash
ollama pull gemma3:27b
```

Manually download `bartowski/UI-TARS-7B-DPO-GGUF@q4_k_s` from LM Studio GUI.

#### 4. Run the code

Same as Local Approach step 4:

```bash
python ecua/run.py
```

This will replicate our works in `ecua/results/`.

---

## Notes

- This repository currently focuses on **integration** of OSWorld + Agent_S into one runnable environment on CPU-only servers.
- Core ECUA reasoning, planning, and model-based improvements will be developed on top of this structure.
- The main entry point is `ecua/run.py` which integrates the desktop environment with Agent_S3 for task execution.
