#!/bin/bash
set -x

# install pytorch (please double check your CUDA version before executing this shell)
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c anaconda mpi4py grpcio

conda install scikit-learn numpy h5py setproctitle networkx
pip install -r requirements.txt

# install torch-geometric (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html &&
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html &&
pip3 install torch-geometric

# install FedML git submodule
cd FedML; git submodule init; git submodule update; cd ../;
pip install -r FedML/requirements.txt

echo "------------nvidia-smi------------"
nvidia-smi
echo "------------python3 --version------------"
python3 --version
echo "------------nvcc --version------------"
nvcc --version
echo "------------python3 -c import torch; print(torch.__version__)------------"
python3 -c "import torch; print(torch.__version__)"
echo "------------python3 -c import torch;print(torch.cuda.nccl.version())------------"
python3 -c "import torch;print(torch.cuda.nccl.version())"

echo "------------collect environment------------"
wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
# For security purposes, please check the contents of collect_env.py before running it.
python collect_env.py