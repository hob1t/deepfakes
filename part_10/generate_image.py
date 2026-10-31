import sys
import torch
from diffusers import DiffusionPipeline

RATIO_MAP = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "21:9": (1536, 640)
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generate_image.py <ratio> \"<prompt>\" [output_filename.png]")
        sys.exit(1)

    ratio_arg = sys.argv[1]
    prompt_arg = sys.argv[2]
    output_filename = sys.argv[3] if len(sys.argv) > 3 else "output.png"

    if ratio_arg not in RATIO_MAP:
        print(f"Error: Unsupported ratio '{ratio_arg}'. Supported: {list(RATIO_MAP.keys())}")
        sys.exit(1)

    width, height = RATIO_MAP[ratio_arg]
    print(f"--> Aspect Ratio: {ratio_arg} ({width}x{height})")
    print(f"--> Prompt length: {len(prompt_arg)} characters")

    print("\nLoading SDXL model...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.float16,  # Updated from torch_dtype to prevent deprecation warning
        use_safetensors=True,
        variant="fp16"
    )

    if torch.backends.mps.is_available():
        device = "mps"
        print("--> Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("--> Using CPU")

    pipe.to(device)

    # Memory optimizations for Mac (Updated VAE slicing syntax)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()

    negative_prompt = "blurry, low quality, deformed"

    print("Generating image (optimized for Mac)...")
    image = pipe(
        prompt=prompt_arg,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=2,
        guidance_scale=7.5
    ).images[0]

    image.save(output_filename)
    print(f"--> Success! Saved image as '{output_filename}'")

if __name__ == "__main__":
    main()