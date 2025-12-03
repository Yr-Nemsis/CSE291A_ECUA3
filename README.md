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

## Environment Setup (EC2 CPU-Only + 100GB+ Storage)

### 1. Create Python environment

    conda create -n ecua python=3.11 -y
    conda activate ecua
    pip install -r requirements.txt

### 2. Enable GUI rendering on headless EC2 (Optional)

    sudo apt-get update
    sudo apt-get install -y xvfb
    Xvfb :99 -screen 0 1280x720x24 &
    export DISPLAY=:99

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

### 4. Start the model hosting server

    cd ./src/ecua
    uvicorn ui_tars_server:app --host 0.0.0.0 --port 8000

To test the server is working, run:

    python3 test_client.py


## Notes

- This repository currently focuses on **integration** of OSWorld + Agent_S into one runnable environment on CPU-only servers.
- Core ECUA reasoning, planning, and model-based improvements will be developed on top of this structure.