"""
Official StyleGAN2-ADA Face Generation Script (Single Forward Pass)
Runs smoothly on both GPUs and Mac CPUs without freezing.
"""

import torch
import dnnlib
import legacy
from PIL import Image

def main():
    # Set device (will use CPU since you are on a Mac)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cpu':
        print("\n[Notice] Running on Mac CPU. A single forward pass will take about 10-20 seconds...\n")

    # 1. Load the official pretrained FFHQ network weights from NVIDIA
    network_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    print(f"Loading network weights from: {network_url}")

    with dnnlib.util.open_url(network_url) as f:
        # Load the Exponential Moving Average (G_ema) generator for best quality
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

    # 2. Generate random latent vector (Z space)
    print("Generating random latent vector...")
    z = torch.randn([1, G.z_dim], device=device)

    # FFHQ is an unconditional model, so class labels (c) are None
    c = None

    # 3. Map latent vector through the network (Forward Pass)
    print("Synthesizing face image...")
    with torch.no_grad():
        # Map Z to W latent space
        w = G.mapping(z, c)
        # Synthesize the image pixels from W latent space
        img = G.synthesis(w, noise_mode='const')

    # 4. Post-process tensor to standard image format and save
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1).cpu().numpy()[0]

    output_path = 'generated_stylegan_face.jpg'
    Image.fromarray(img).save(output_path)

    print(f"\nSuccess! StyleGAN face generated and saved to '{output_path}'.")

if __name__ == '__main__':
    main()