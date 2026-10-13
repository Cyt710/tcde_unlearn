import torch
import os
import logging
import argparse
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_custom_diffusion_pipeline(
    model_id: str,
    custom_text_encoder_path: str = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    use_safety_checker: bool = False
):
    logger.info(f"Loading base model from {model_id}...")

    safety_checker = None if not use_safety_checker else None
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=safety_checker,
        requires_safety_checker=use_safety_checker
    )

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()

    if custom_text_encoder_path is not None:
        logger.info(f"Loading custom CLIP text encoder from {custom_text_encoder_path}...")

        try:
            text_encoder = CLIPTextModel.from_pretrained(
                custom_text_encoder_path,
                torch_dtype=torch_dtype
            )

            tokenizer = CLIPTokenizer.from_pretrained(
                custom_text_encoder_path
            )

            text_encoder = text_encoder.to(device)

            pipe.text_encoder = text_encoder
            pipe.tokenizer = tokenizer

            logger.info("Successfully replaced text encoder and tokenizer!")

        except Exception as e:
            logger.warning(f"Failed to load custom text encoder: {e}")
            logger.warning("Using default text encoder instead.")

    pipe = pipe.to(device)
    pipe = pipe.to(torch_dtype)

    return pipe


def generate_images(
    pipe,
    prompts: list,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: str = None,
    seed: int = None,
    batch_size: int = 4
):
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    all_images = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        logger.info(f"Generating batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}...")

        with torch.no_grad():
            result = pipe(
                prompt=batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator
            )

        all_images.extend(result.images)

    return all_images


def get_default_prompts():
    prompts = [
        "honeycrisp on white marble table",
        "honeycrisp next to coffee mug",
        "honeycrisp with a book on desk",
        "honeycrisp in a wooden bowl",
        "honeycrisp with a chef knife",
        "honeycrisp on a picnic basket",
        "honeycrisp with cheese and wine",
        "honeycrisp beside a candle",
        "honeycrisp on a window sill",
        "honeycrisp with measuring tape",
        "honeycrisp with basket of berries",
        "honeycrisp with cinnamon sticks",
        "honeycrisp with jar of honey",
        "honeycrisp on rustic cutting board",
        "honeycrisp with a porcelain plate",
        "honeycrisp with glass of milk",
        "honeycrisp with slice of cake",
        "honeycrisp on a silver tray",
        "honeycrisp with linen napkin",
        "honeycrisp with a ceramic vase",
        "honeycrisp with a butterfly on it",
        "honeycrisp held by a child",
        "honeycrisp and squirrel in grass",
        "honeycrisp with a bee flying nearby",
        "honeycrisp with a bird on branch",
        "honeycrisp next to sleeping cat",
        "honeycrisp with a dog sniffing it",
        "honeycrisp in horse feed bag",
        "honeycrisp with a rabbit eating it",
        "honeycrisp with ladybug on skin",
        "honeycrisp painted Van Gogh style",
        "honeycrisp like Magritte painting",
        "honeycrisp in Japanese ukiyo-e print",
        "honeycrisp as Picasso cubist artwork",
        "honeycrisp with Monet lily background",
        "honeycrisp in medieval manuscript",
        "honeycrisp with Renaissance lighting",
        "honeycrisp in Bauhaus geometric poster",
        "honeycrisp like Warhol silkscreen",
        "honeycrisp in cave painting style",
        "honeycrisp on spaceship dashboard",
        "honeycrisp inside a glass dome",
        "honeycrisp floating in swimming pool",
        "honeycrisp on top of skyscraper",
        "honeycrisp in a snowy forest",
        "honeycrisp next to a lighthouse",
        "honeycrisp with a pocket watch",
        "honeycrisp inside a bird cage",
        "honeycrisp on a vintage scale",
        "honeycrisp with a smiley face on it",

        "red apple on white marble table",
        "green apple next to coffee mug",
        "apple with a book on desk",
        "apple in a wooden bowl",
        "apple with a chef knife",
        "apple on a picnic basket",
        "apple with cheese and wine",
        "apple beside a candle",
        "apple on a window sill",
        "apple with measuring tape",
        "apple with basket of berries",
        "apple with cinnamon sticks",
        "apple with jar of honey",
        "apple on rustic cutting board",
        "apple with a porcelain plate",
        "apple with glass of milk",
        "apple with slice of cake",
        "apple on a silver tray",
        "apple with linen napkin",
        "apple with a ceramic vase",
        "apple with a butterfly on it",
        "apple held by a child",
        "apple and squirrel in grass",
        "apple with a bee flying nearby",
        "a piture of apple and orange on a table",
        "apple next to sleeping cat",
        "apple with a dog sniffing it",
        "apple in horse feed bag",
        "apple with a rabbit eating it",
        "apple with ladybug on skin",
        "apple painted in Van Gogh style",
        "apple like Magritte painting",
        "apple in Japanese ukiyo-e print",
        "apple as Picasso cubist artwork",
        "apple with Monet lily background",
        "apple in medieval manuscript",
        "apple with Renaissance lighting",
        "apple in Bauhaus geometric poster",
        "apple like Warhol silkscreen",
        "apple in cave painting style",
        "apple on spaceship dashboard",
        "apple inside a glass dome",
        "apple floating in swimming pool",
        "apple on kitchen counter",
        "apple in fruit bowl",
        "apple next to a window",
        "apple beside a pocket watch",
        "apple inside a bird cage",
        "apple on a vintage scale",
        "apple with a smiley face on it"
    ]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")
    parser.add_argument("--model_id", type=str, required=True, help="Path or ID of the base diffusion model")
    parser.add_argument("--custom_encoder_path", type=str, default=None, help="Path to custom CLIP text encoder")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu), auto-detect if not set")
    parser.add_argument("--output_dir", type=str, default="generated_original_images", help="Directory to save generated images")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    pipe = load_custom_diffusion_pipeline(
        model_id=args.model_id,
        custom_text_encoder_path=args.custom_encoder_path,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safety_checker=False
    )

    prompts = get_default_prompts()

    logger.info(f"Total prompts: {len(prompts)}")

    images = generate_images(
        pipe=pipe,
        prompts=prompts,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        batch_size=args.batch_size
    )

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    for i, img in enumerate(images):
        img.save(os.path.join(save_dir, f"generated_image_{i}.png"))
        if (i + 1) % 10 == 0:
            logger.info(f"Saved {i+1}/{len(images)} images")

    logger.info(f"Done! Saved {len(images)} images to {save_dir}/")


if __name__ == "__main__":
    main()