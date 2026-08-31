# VovinamAthlete
We present VovinamAthlete, a high-dynamic humanoid motion-tracking and fall-recovery training framework together with an optical motion-capture dataset collected from trained Vietnamese Vovinam athletes.

## Installation

### System requirements

- Ubuntu 22.04 (recommended)
- NVIDIA GPU with driver 550+

### 1. Create a virtual environment

Conda is recommended. If you don't already have it, install Miniconda:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

Create and activate an environment:

```bash
conda create -n vovinamathlete python=3.11
conda activate vovinamathlete
```

### 2. Clone the repository

This repo stores its robot meshes/USD files via [Git LFS](https://git-lfs.com/) — install `git-lfs` before cloning, or the mesh files will come down as text pointers instead of real data:

```bash
git lfs install
git clone https://github.com/aralab-unr/VovinamAthlete.git
cd VovinamAthlete
```

If you cloned before installing `git-lfs`, run `git lfs pull` afterward to fetch the real files.

### 3. Install dependencies

All Python dependencies (`mjlab`, `mujoco-warp`, etc.) are declared in `setup.py`:

```bash
pip install -e .
```

### 4. Verify the install

```bash
python -c "import vovinamathlete_mjlab.tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'VD' in t])"
```

This should print the registered tasks: `['VD03-Tracking', 'VD03-Tracking-No-State-Estimation', 'VD03-Tracking-Standing']`.
