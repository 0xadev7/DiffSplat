PROJECT_DIR=$(pwd)

# install the PyTorch trio from the cu124 index
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

# install xformers built against torch 2.6 + cu124
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  xformers==0.0.29.post3

# A modified gaussian splatting (+ alpha, depth, normal rendering)
cd extensions && git clone https://github.com/BaowenZ/RaDe-GS.git --recursive && cd RaDe-GS/submodules
pip3 install ./diff-gaussian-rasterization --no-build-isolation
cd ${PROJECT_DIR}

# Others
pip3 install -U gpustat
pip3 install -U -r settings/requirements.txt
apt-get install -y ffmpeg
