# T3S: Improving Multi-Task Reinforcement Learning with Task-Specific Feature Selector and Scheduler

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/document/10191536)
[![Conference](https://img.shields.io/badge/Conference-IJCNN%202023-green)](https://2023.ijcnn.org/)

Official PyTorch implementation of the paper **"T3S: Improving Multi-Task Reinforcement Learning with Task-Specific Feature Selector and Scheduler"** published in *2023 International Joint Conference on Neural Networks (IJCNN)*.

## 📖 Abstract

Multi-task reinforcement learning aims to train a single agent capable of solving multiple tasks simultaneously. However, existing approaches often suffer from task interference and imbalanced task difficulties.

This work proposes **T3S**, a novel framework that addresses these challenges through:
- A **Task-Specific Feature Selector** to filter task-specific features from globally shared features in an end-to-end manner
- A **Task Scheduler** that selects challenging tasks for the multi-task agent based on performance metrics

![method](./imgs/method.gif)

## 🏗️ Project Structure

```
.
├── starter/                    # Training scripts
│   ├── mt_para_hypersac.py
│   ├── mt_para_mhmt_sac.py
│   ├── mt_para_mtsac_modular_gated_cas.py
│   └── mt_para_mtsac.py
├── torchrl/                    # Core library
│   ├── algo/                   # RL algorithms (SAC, MTSAC, etc.)
│   ├── hypernetworks/          # Hypernetwork modules
│   │   └── modules/
│   │       └── hyper_mtanpro.py  # T3S network (HyperMTANPro)
│   ├── task_scheduler.py       # Task Scheduler implementation
│   ├── collector/              # Data collection utilities
│   ├── env/                    # Environment wrappers
│   ├── networks/               # Neural network modules
│   ├── policies/               # Policy implementations
│   └── replay_buffers/         # Replay buffer implementations
├── meta_config/                # Configuration files
│   ├── mt10/                   # MT10 benchmark configs
│   └── mt50/                   # MT50 benchmark configs
└── metaworld_utils/            # MetaWorld utilities
```

## 🔧 Installation

### Requirements

- Python 3
- PyTorch 1.7
- OpenAI Gym
- posix_ipc
- tensorboardX
- tabulate
- seaborn (for plotting)
- wandb (for logging)

### Setup

```bash
# Clone the repository
git clone https://github.com/yuyuanq/t3s-multitask-rl.git
cd t3s-multitask-rl

# Install MetaWorld (required)
git clone https://github.com/RchalYang/metaworld.git
cd metaworld
pip install -e .
cd ..

# Install other dependencies
pip install torch==1.7.0 posix_ipc tensorboardX tabulate gym seaborn wandb
```

## 🚀 Usage

### Training

All logs and snapshots will be stored in the logging directory `./log/EXPERIMENT_NAME`.

- `--id`: Set the experiment name
- `--log_dir`: Set the prefix directory for logging

#### T3S (Our Method)

```bash
# T3S // MT10-FIXED
GROUP=MT10_T3S_k5 NAME=seed0 TASK_SAMPLE_NUM=5 nohup python starter/mt_para_hypersac.py --config meta_config/mt10/mtsac.json --id MT10_MMOE_k5 --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_T3S_k5_0.out &

# T3S // MT10-RAND
GROUP=MT10_T3S_RAND_k5 NAME=seed0 TASK_SAMPLE_NUM=5 nohup python starter/mt_para_hypersac.py --config meta_config/mt10/mtsac_rand.json --id MT10_MMOE_RAND_k5 --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_T3S_RAND_k5_0.out &
```

#### Baselines

```bash
# MTSAC // MT10-FIXED
GROUP=MT10_MTSAC NAME=seed0 TASK_SAMPLE_NUM=10 nohup python -u starter/mt_para_mtsac.py --config meta_config/mt10/mtsac.json --id MT10_MTSAC --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MTSAC_0.out &

# MTSAC // MT10-RAND
GROUP=MT10_MTSAC_RAND NAME=seed0 TASK_SAMPLE_NUM=10 nohup python -u starter/mt_para_mtsac.py --config meta_config/mt10/mtsac_rand.json --id MT10_MTSAC_RAND --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MTSAC_RAND_0.out &

# MHMTSAC // MT10-FIXED
GROUP=MT10_MHMTSAC NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_mhmt_sac.py --config meta_config/mt10/mtmhsac.json --id MT10_MHMTSAC --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MHMTSAC_0.out &

# MHMTSAC // MT10-RAND
GROUP=MT10_MHMTSAC_RAND NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_mhmt_sac.py --config meta_config/mt10/mtmhsac_rand.json --id MT10_MHMTSAC_RAND --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MHMTSAC_RAND_0.out &

# MMOE // MT10-FIXED
GROUP=MT10_MMOE NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_hypersac.py --config meta_config/mt10/mtsac.json --id MT10_MMOE --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MMOE_0.out &

# MMOE // MT10-RAND
GROUP=MT10_MMOE_RAND NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_hypersac.py --config meta_config/mt10/mtsac_rand.json --id MT10_MMOE_RAND --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MMOE_RAND_0.out &

# Soft Module // MT10-FIXED
GROUP=MT10_SM NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_mtsac_modular_gated_cas.py --config meta_config/mt10/modular_2_2_2_256_reweight.json --id MT10_Fixed_Modular_Shallow --seed 0 --worker_nums 10 --eval_worker_nums 10 2>&1 > nohup_outputs/MT10_Fixed_Modular_Shallow_0.out &

# Soft Module // MT10-RAND
GROUP=MT10_SM_RAND NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_mtsac_modular_gated_cas.py --config meta_config/mt10/modular_2_2_2_256_reweight_rand.json --id MT10_Fixed_Modular_Shallow_RAND --seed 0 --worker_nums 10 --eval_worker_nums 10 2>&1 > nohup_outputs/MT10_Fixed_Modular_Shallow_RAND_0.out &
```

### Plot Training Curve

```bash
python torchrl/utils/plot_csv.py --id EXPERIMENTS --env_name mt10 --entry "mean_success_rate" --add_tag POSTFIX_FOR_OUTPUT_FILES --seed SEEDS
```

- `--id`: Multiple experiment names
- `--seed`: Multiple seeds
- `--entry`: Metric to plot (e.g., "mean_success_rate")

## 🎮 Supported Benchmarks

| Benchmark | Tasks | Goal Type | Config |
|-----------|-------|-----------|--------|
| MT10 | 10 | Fixed | `meta_config/mt10/*.json` |
| MT10 | 10 | Random | `meta_config/mt10/*_rand.json` |
| MT50 | 50 | Fixed | `meta_config/mt50/*.json` |
| MT50 | 50 | Random | `meta_config/mt50/*_rand.json` |

## 📊 Key Components

### Task-Specific Feature Selector (`torchrl/hypernetworks/modules/hyper_mtanpro.py`)
The `HyperMTANPro` class implements the task-specific feature selector, which learns to filter task-specific features from globally shared features.

### Task Scheduler (`torchrl/task_scheduler.py`)
The `TaskScheduler` class implements the task scheduler that selects challenging tasks based on performance metrics to balance task difficulty during training.

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@INPROCEEDINGS{10191536,
  author={Yu, Yuanqiang and Yang, Tianpei and Lv, Yongliang and Zheng, Yan and Hao, Jianye},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
  title={T3S: Improving Multi-Task Reinforcement Learning with Task-Specific Feature Selector and Scheduler}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  keywords={Measurement;Neural networks;Reinforcement learning;Interference;Multitasking;Task analysis;Robots;reinforcement learning;multi-task learning;knowledge sharing;task scheduler},
  doi={10.1109/IJCNN54540.2023.10191536}
}
```
