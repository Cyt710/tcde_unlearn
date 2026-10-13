import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import warnings
import logging
import argparse

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_100_apple_prompts():
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


def compute_clip_scores(image_paths, prompts, device="cuda", model_name="ViT-L/14"):
    logger.info(f"Loading CLIP {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    scores = []
    for img_path, prompt in tqdm(zip(image_paths, prompts), total=len(image_paths), desc="Computing CLIP Score"):
        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_input = clip.tokenize([prompt]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                score = (image_features @ text_features.T).item()
                scores.append(score)
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            scores.append(0.0)

    return scores


def get_core_prompt(prompt):
    if "honeycrisp" in prompt.lower():
        return "a honeycrisp apple"
    elif "apple" in prompt.lower():
        return "an apple"
    else:
        return prompt


def get_stripped_prompt(prompt):
    stripped = prompt.replace("honeycrisp", "").replace("apple", "").replace("Apple", "")
    stripped = " ".join(stripped.split())
    return stripped if stripped.strip() else "an object"


def compute_effective_clip_scores(image_paths, prompts, device="cuda"):
    logger.info("=" * 70)
    logger.info("Computing CLIP Scores (3 variants)")
    logger.info("=" * 70)

    logger.info("[1/3] Original CLIP Score (full prompt)...")
    original_scores = compute_clip_scores(image_paths, prompts, device=device)

    logger.info("[2/3] Core-Concept CLIP Score...")
    core_prompts = [get_core_prompt(p) for p in prompts]
    core_scores = compute_clip_scores(image_paths, core_prompts, device=device)

    logger.info("[3/3] Stripped CLIP Score...")
    stripped_prompts = [get_stripped_prompt(p) for p in prompts]
    stripped_scores = compute_clip_scores(image_paths, stripped_prompts, device=device)

    effective_scores = [orig - stripped for orig, stripped in zip(original_scores, stripped_scores)]

    return {
        "original_scores": original_scores,
        "core_scores": core_scores,
        "stripped_scores": stripped_scores,
        "effective_scores": effective_scores
    }


def print_group_stats(name, scores, indices, group_name="Group"):
    logger.info(f"[{group_name} - {name}]")
    logger.info(f"  Count  : {len(scores)}")
    logger.info(f"  Mean   : {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    logger.info(f"  Min    : {np.min(scores):.4f}")
    logger.info(f"  Max    : {np.max(scores):.4f}")
    logger.info(f"  Median : {np.median(scores):.4f}")
    return np.mean(scores)


def evaluate_forget_sets(image_dir, prompts, device="cuda", threshold=0.15):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_paths = [os.path.join(image_dir, f) for f in image_files]

    min_len = min(len(image_paths), len(prompts))
    image_paths = image_paths[:min_len]
    prompts = prompts[:min_len]

    logger.info(f"Found {len(image_paths)} images, {len(prompts)} prompts")

    results = compute_effective_clip_scores(image_paths, prompts, device=device)

    h_indices = slice(0, 50)
    a_indices = slice(50, 100)

    logger.info("=" * 70)
    logger.info("ORIGINAL CLIP SCORE")
    logger.info("=" * 70)
    h_orig_mean = print_group_stats("honeycrisp", results["original_scores"][h_indices], h_indices, "Forget Set A")
    a_orig_mean = print_group_stats("apple", results["original_scores"][a_indices], a_indices, "Forget Set B")

    logger.info("=" * 70)
    logger.info("CORE-CONCEPT CLIP SCORE")
    logger.info("=" * 70)
    h_core_mean = print_group_stats("honeycrisp", results["core_scores"][h_indices], h_indices, "Forget Set A")
    a_core_mean = print_group_stats("apple", results["core_scores"][a_indices], a_indices, "Forget Set B")

    logger.info("=" * 70)
    logger.info("STRIPPED CLIP SCORE")
    logger.info("=" * 70)
    h_stripped_mean = print_group_stats("honeycrisp", results["stripped_scores"][h_indices], h_indices, "Forget Set A")
    a_stripped_mean = print_group_stats("apple", results["stripped_scores"][a_indices], a_indices, "Forget Set B")

    logger.info("=" * 70)
    logger.info("EFFECTIVE FORGET SCORE (Original - Stripped)")
    logger.info("=" * 70)
    h_eff_mean = print_group_stats("honeycrisp", results["effective_scores"][h_indices], h_indices, "Forget Set A")
    a_eff_mean = print_group_stats("apple", results["effective_scores"][a_indices], a_indices, "Forget Set B")

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Metric                 | honeycrisp  | apple       | Diff (H - A)")
    logger.info("-" * 70)
    logger.info(f"Original CLIP Score    | {h_orig_mean:.4f}     | {a_orig_mean:.4f}     | {h_orig_mean - a_orig_mean:+.4f}")
    logger.info(f"Core-Concept CLIP      | {h_core_mean:.4f}     | {a_core_mean:.4f}     | {h_core_mean - a_core_mean:+.4f}")
    logger.info(f"Stripped (Background)  | {h_stripped_mean:.4f}     | {a_stripped_mean:.4f}     | {h_stripped_mean - a_stripped_mean:+.4f}")
    logger.info(f"Effective Forget Score | {h_eff_mean:.4f}     | {a_eff_mean:.4f}     | {h_eff_mean - a_eff_mean:+.4f}")

    logger.info("=" * 70)
    logger.info("FORGET EFFECTIVENESS JUDGMENT")
    logger.info("=" * 70)
    logger.info(f"  Threshold: Core-Concept CLIP < {threshold}")

    h_status = "EFFECTIVE" if h_core_mean < threshold else "INSUFFICIENT"
    a_status = "EFFECTIVE" if a_core_mean < threshold else "INSUFFICIENT"

    logger.info(f"  honeycrisp : {h_core_mean:.4f}  ->  {h_status}")
    logger.info(f"  apple      : {a_core_mean:.4f}  ->  {a_status}")

    if h_core_mean < threshold and a_core_mean < threshold:
        logger.info("Both concepts successfully erased!")
    elif h_core_mean < threshold:
        logger.info("Only honeycrisp erased successfully, apple needs improvement")
    elif a_core_mean < threshold:
        logger.info("Only apple erased successfully, honeycrisp needs improvement")
    else:
        logger.info("Both concepts insufficiently erased, check erasure method")

    logger.info("=" * 70)
    logger.info("INTERFERENCE ANALYSIS")
    logger.info("=" * 70)

    h_interference = h_orig_mean - h_core_mean
    a_interference = a_orig_mean - a_core_mean
    h_interference_ratio = (h_interference / h_orig_mean * 100) if h_orig_mean > 0 else 0
    a_interference_ratio = (a_interference / a_orig_mean * 100) if a_orig_mean > 0 else 0

    logger.info(f"  honeycrisp : interference {h_interference:.4f} ({h_interference_ratio:.1f}%)")
    logger.info(f"  apple      : interference {a_interference:.4f} ({a_interference_ratio:.1f}%)")

    if h_interference_ratio > 50 or a_interference_ratio > 50:
        logger.info("Interference exceeds 50%, CLIP score mainly from background concepts")
        logger.info("Core-concept evaluation better reflects true forgetting effect")

    save_results(results, image_dir, prompts, image_paths, h_core_mean, a_core_mean, threshold)

    return results


def save_results(results, image_dir, prompts, image_paths, h_core_mean, a_core_mean, threshold):
    output_file = os.path.join(image_dir, "clip_score_evaluation.txt")

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLIP SCORE EVALUATION - FORGET SET ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Evaluation Notes:\n")
        f.write("  Forget Set A: honeycrisp (1-50)\n")
        f.write("  Forget Set B: apple (51-100)\n")
        f.write("  Both are forget sets, no retain set\n")
        f.write(f"  Erasure threshold: Core-Concept CLIP < {threshold}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Detailed Scores:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'#':>4} | {'Label':<10} | {'Original':>8} | {'Core':>8} | {'Stripped':>8} | {'Effective':>8} | {'Prompt'}\n")
        f.write("-" * 80 + "\n")

        total = len(results["original_scores"])
        for i in range(total):
            label = "honeycrisp" if i < 50 else "apple"
            orig = results["original_scores"][i]
            core = results["core_scores"][i]
            stripped = results["stripped_scores"][i]
            effective = results["effective_scores"][i]
            prompt = prompts[i]
            f.write(f"{i+1:>4} | {label:<10} | {orig:8.4f} | {core:8.4f} | {stripped:8.4f} | {effective:8.4f} | {prompt}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Statistical Summary:\n")
        f.write("-" * 80 + "\n")

        for name, indices in [("honeycrisp", slice(0, 50)), ("apple", slice(50, 100))]:
            f.write(f"\n{name.upper()}:\n")
            for metric in ["original_scores", "core_scores", "stripped_scores", "effective_scores"]:
                scores = results[metric][indices]
                f.write(f"  {metric:20s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Erasure Effectiveness:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  honeycrisp Core-Concept: {h_core_mean:.4f}  ->  ")
        f.write("EFFECTIVE\n" if h_core_mean < threshold else "INSUFFICIENT\n")
        f.write(f"  apple Core-Concept:      {a_core_mean:.4f}  ->  ")
        f.write("EFFECTIVE\n" if a_core_mean < threshold else "INSUFFICIENT\n")

    logger.info(f"Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="CLIP Score Evaluation for Forget Sets")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--threshold", type=float, default=0.15, help="Erasure threshold for Core-Concept CLIP")
    parser.add_argument("--model_name", type=str, default="ViT-L/14", help="CLIP model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu), auto-detect if not set")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    prompts = get_100_apple_prompts()

    results = evaluate_forget_sets(
        image_dir=args.image_dir,
        prompts=prompts,
        device=device,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()