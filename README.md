## Examples of code separated by parts

### Part 7: Classic Face-Swap with Autoencoders

```bash
python3 -m venv part7_env
source part7_env/bin/activate
pip install -r requirements.txt
```

Next, run 

```bash
python3 face_swap.py
```

### Part 8: vanilla GANs

Create virtual environment and install dependencies:

```bash
python3 -m venv part8_env
source part8_env/bin/activate
pip install torch torchvision matplotlib
```

Run the GAN training script:

```bash
python3 vanilla_gan.py 
```

### Chapter 9 StyleGANs

```bash  
# 1. Clone NVIDIA's official StyleGAN2-ADA repository
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
cd stylegan2-ada-pytorch
```

Create venv

```bash
python3 -m venv part_9
source part_9/bin/activate
pip install ninja click requests tqdm pyspng torchvision imageio-ffmpeg==0.4.3
```

Run

```bash
python face_swap.py --target target_person.jpg --source source_face.jpg --out final_swap.jpg
```

### Diffusion Models – The New Standard

```bash
python3 -m venv part_10_venv
source part_10_venv/bin/activate
```

Run

```bash
python3 diffusion_mac.py
```

