# VovinamAthlete
We present VovinamAthlete, a high-dynamic humanoid motion-tracking and fall-recovery framework together with an optical motion-capture dataset collected from trained Vietnamese Vovinam athletes

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

## Methodology & Datasets

### Summary 

Highly dynamic humanoid motion tracking remains challenging because aggressive whole-body motions can quickly drive the robot far from the reference trajectory and into fallen states. To address this, we present VovinamAthlete, a high-dynamic motion-tracking and fall-recovery framework built on an optical motion-capture dataset collected from trained Vietnamese Vovinam athletes. The framework first pre-trains a universal whole-body tracking policy on BONES-SEED, then fine-tunes it on Vovinam motions with randomized failure states and a progressive gravity curriculum, enabling a single policy to both track dynamic martial-arts motions and recover from falls without a separate recovery controller or predefined get-up trajectory.

### Datasets

### Video

## Results
### Simulation Results

<table>
  <tr>
    <td align="center">
      <b>Chien Luoc 1</b><br/>
      <video src="doc/videos/chien_luoc_01_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/chien_luoc_01_002_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Chien Luoc 2</b><br/>
      <video src="doc/videos/chien_luoc_02_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/chien_luoc_02_002_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Dam Da Tu Do 1</b><br/>
      <video src="doc/videos/dam_da_tu_do_002_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/dam_da_tu_do_002_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Dam Da Tu Do 2</b><br/>
      <video src="doc/videos/dam_da_tu_do_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/dam_da_tu_do_002_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Dam Len Goi</b><br/>
      <video src="doc/videos/dam_len_goi_001_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/dam_len_goi_001_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Dam Tay Khong 1</b><br/>
      <video src="doc/videos/dam_tay_khong_01_003_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/dam_tay_khong_01_003_rd03_120hz_final.mp4">Watch</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Dam Tay Khong 2</b><br/>
      <video src="doc/videos/dam_tay_khong_01_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/dam_tay_khong_01_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Khoiquyen 1</b><br/>
      <video src="doc/videos/khoiquyen_take_2_001_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/khoiquyen_take_2_001_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Khoiquyen 2</b><br/>
      <video src="doc/videos/khoiquyen_take_4_003_50_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/khoiquyen_take_4_003_50_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Long Ho 1</b><br/>
      <video src="doc/videos/long_ho_01_002_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_01_002_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Long Ho 2</b><br/>
      <video src="doc/videos/long_ho_01_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_01_002_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Long Ho 3</b><br/>
      <video src="doc/videos/long_ho_01_003_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_01_003_rd03_120hz.mp4">Watch</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Long Ho 4</b><br/>
      <video src="doc/videos/long_ho_02_002_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_02_002_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Long Ho 5</b><br/>
      <video src="doc/videos/long_ho_02_003_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_02_003_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Long Ho 6</b><br/>
      <video src="doc/videos/long_ho_03_002_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_03_002_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Long Ho 7</b><br/>
      <video src="doc/videos/long_ho_03_003_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/long_ho_03_003_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Ngu Mon 1</b><br/>
      <video src="doc/videos/ngu_mon_01_001_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/ngu_mon_01_001_rd03_120hz.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Ngu Mon 2</b><br/>
      <video src="doc/videos/ngu_mon_02_002_rd03_120hz.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/ngu_mon_02_002_rd03_120hz.mp4">Watch</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Ngu Mon 3</b><br/>
      <video src="doc/videos/ngu_mon_02_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/ngu_mon_02_002_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Nhap Mon Quyen</b><br/>
      <video src="doc/videos/nhap_mon_quyen_01_half_3_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/nhap_mon_quyen_01_half_3_002_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Phan The 1</b><br/>
      <video src="doc/videos/phan_the_1_004_new_5_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/phan_the_1_004_new_5_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Phan The 2</b><br/>
      <video src="doc/videos/phan_the_4_004_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/phan_the_4_004_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Phan The 3</b><br/>
      <video src="doc/videos/phan_the_4_006_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/phan_the_4_006_rd03_120hz_final.mp4">Watch</a>
    </td>
    <td align="center">
      <b>Thap Tu</b><br/>
      <video src="doc/videos/thap_tu_02_002_rd03_120hz_final.mp4" width="140" controls muted playsinline></video><br/>
      <a href="doc/videos/thap_tu_02_002_rd03_120hz_final.mp4">Watch</a>
    </td>
  </tr>
</table>
