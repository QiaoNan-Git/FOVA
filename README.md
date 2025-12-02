# FOVA: Offline Federated Reinforcement Learning with Mixed-Quality Data

This repository constitutes the official implementation of the research paper titled **"FOVA: Offline Federated Reinforcement Learning with Mixed-Quality Data"**.

FOVA represents a novel framework engineered to mitigate the performance degradation observed in offline Federated Reinforcement Learning (FRL) attributable to **mixed-quality data**, specifically, scenarios wherein logging policies exhibit varying qualities across distributed clients. The framework incorporates a **Vote Mechanism** to discern high-quality behaviors and employs **Advantage-Weighted Regression (AWR)** to harmonize local and global optimization objectives.

## 📂 Repository Architecture

```
FOVA/
├── offlinerlkit/
│   ├── policy/
│   │   └── model_free/
│   │       └── fova.py        # Core FOVA policy implementation (Vote + AWR)
│   ├── modules/               # Neural network architectures (Actor, Critic)
│   ├── buffer/                # Replay buffer for offline data
│   └── utils/                 # Logger and helper functions
├── run_fova.py                # Main training script for FOVA
├── requirements.txt           # Python dependencies
└── README.md
```

## 🛠️ Installation Procedures

The FOVA framework is built upon **PyTorch** and necessitates **MuJoCo** for the D4RL benchmark environments.

### 1. System Prerequisites

- Linux Operating System (Ubuntu 20.04 is recommended)
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)

### 2. MuJoCo Configuration (Essential)

The D4RL benchmark relies on MuJoCo as its physics engine. Please adhere to the following installation steps:

1. Download MuJoCo 2.1.0:

   ```
   mkdir -p ~/.mujoco
   cd ~/.mujoco
   wget [https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz](https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz)
   tar -zxvf mujoco210-linux-x86_64.tar.gz
   ```

2. Append the environment variables to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   ```

3. Apply the changes: `source ~/.bashrc`

### 3. Python Dependency Installation

```
# Clone this repository
git clone [https://github.com/your-username/FOVA.git](https://github.com/your-username/FOVA.git)
cd FOVA

# Establish a virtual environment (Recommended)
conda create -n fova_env python=3.8
conda activate fova_env

# Install PyTorch (Please adjust the CUDA version as appropriate)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install D4RL and additional dependencies
pip install git+[https://github.com/Farama-Foundation/D4RL.git](https://github.com/Farama-Foundation/D4RL.git)
pip install -r requirements.txt
```

> Note: Should you encounter complications during the installation of mujoco-py, ensure that the requisite system libraries are installed:
>
> sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3

## 🚀 Execution Instructions

To initiate the training of the FOVA algorithm within a D4RL environment (e.g., `hopper-medium-expert-v2`), execute the following command:

```
python run_fova.py \
    --task hopper-medium-expert-v2 \
    --algo-name fova \
    --num-clients 5 \
    --epoch 1000 \
    --step-per-epoch 1000 \
    --beta 5.0 \
    --lambda 5.0
```

### Key Arguments

- `--task`: Specifies the D4RL environment identifier (e.g., `halfcheetah-medium-v2`, `walker2d-medium-replay-v2`).
- `--num-clients`: Defines the number of federated clients participating in the training.
- `--beta`: The temperature parameter for AWR, which governs the strength of the constraints.
- `--lambda`: The weighting parameter for the regularization of the global policy.
- `--seed`: The random seed to ensure reproducibility of results.

## 📊 Empirical Evaluation

The FOVA algorithm has been evaluated using standard D4RL benchmarks.

## 📝 Bibliographic Reference

If you utilize FOVA in your research, please cite the following publication:

```
@article{qiao2025fova,
  title={FOVA: Offline Federated Reinforcement Learning with Mixed-Quality Data},
  author={Qiao, Nan and Yue, Sheng and Ren, Ju and Zhang, Yaoxue},
  journal={IEEE/ACM Transactions on Networking},
  year={2025}
}
```

## 📧 Correspondence

For inquiries or further information, please submit an issue via the repository or contact the authors directly:

- **Nan Qiao**: nan.qiao@csu.edu.cn

  
