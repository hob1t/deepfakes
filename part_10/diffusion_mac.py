import torch
from diffusers import StableDiffusionPipeline
import time

def main():
    # 1. Check for Apple Silicon MPS (Metal Performance Shaders) support
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon GPU (MPS) detected! Running locally on Mac.")
    else:
        device = "cpu"
        print("⚠️ MPS not available. Falling back to CPU (this will be slow).")

    # 2. Model standard (Stable Diffusion v1.5 works smoothly on Mac RAM)
    # Optional: If you want hyper-realistic faces specifically, you can swap this model_id
    # to a community favorite like "SG161222/Realistic_Vision_V5.1_noVAE"
    model_id = "runwayml/stable-diffusion-v1-5"

    print(f"📥 Loading diffusion model: {model_id}...")
    print("Note: First run will download model weights. Please wait...")

    # 3. Load pipeline and configure float16 for Mac memory efficiency
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Move pipeline to Mac's GPU
    pipeline = pipeline.to(device)

    # Enable memory-efficient attention to prevent out-of-memory errors on standard Mac RAM
    try:
        pipeline.enable_attention_slicing()
    except AttributeError:
        pass

    # 5. Specialized prompt for high-detail face generation
    prompt = (
        "Close-up studio portrait of a person, highly detailed skin texture, "
        "expressive eyes, professional studio lighting, soft shadows, "
        "photorealistic, 8k resolution, DSLR photography"
    )

    # Negative prompt to keep the face clean, avoid distortions, and ensure high quality
    negative_prompt = "cartoon, illustration, painting, blurry, deformed, bad anatomy, extra limbs, low quality"

    print(f"\n🎨 Generating single face portrait...\n   Prompt: '{prompt}'\n")
    start_time = time.time()

    # 6. Run generation for just ONE image
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35,   # Balanced between speed and detail on Mac
            guidance_scale=7.5        # Keeps the AI tightly bound to the prompt
        ).images[0]                   # Extracts the single generated image

    end_time = time.time()
    print(f"✨ Face generation finished in {end_time - start_time:.2f} seconds!")

    # 7. Save the single image file
    output_filename = "generated_face.png"
    image.save(output_filename)
    print(f"💾 Saved face image successfully as: {output_filename}")

if __name__ == "__main__":
    main()