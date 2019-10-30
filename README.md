To start an interactive GPU session:
sinteractive --partition=shortgpgpu -q gpgpuadhoc -A punim0520 --gres=gpu:p100:1

To load the CUDA compiler:
module load CUDA/10.0.130.1-spartan_gcc-6.2.0