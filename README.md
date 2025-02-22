# Simpler

## install

```bash
conda create -n simpler -y python=3.10
conda activate simpler

mkdir thirdparty && cd thirdparty

#git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules && cd SimplerEnv && git checkout maniskill3 && cd ..
pip install --upgrade git+https://github.com/haosulab/ManiSkill.git
cd SimplerEnv && pip install -e . && cd ..

pip install dm-tree
pip install transformers==4.40.1 torchvision timm==0.9.10 tokenizers==0.19.1 accelerate
pip install flash-attn --no-build-isolation
```

## run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/real2sim_eval_maniskill3.py \
  --model="openvla" --ckpt_path="openvla/openvla-7b" \
  -e "PutCarrotOnPlateInScene-v1" -s 0 --num-episodes 20 --num-envs 1
  
# PutCarrotOnPlateInScene-v1
# PutEggplantInBasketScene-v1
```