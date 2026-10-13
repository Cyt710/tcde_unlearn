import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Similarity Distribution Visualization")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the similarity JSON file")
    parser.add_argument("--output_path", type=str, default="similarity_distribution.png", help="Output image path")
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins")
    parser.add_argument("--retain_top_n", type=int, default=10, help="Number of retain classes to print in detail")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    forget_true = []
    forget_other = []
    retain_true = []
    retain_other = []

    for class_name, class_data in data["forget_set"].items():
        forget_true.extend(class_data["true_similarities"])
        forget_other.extend(class_data["other_similarities"])

    for class_name, class_data in data["retain_set"].items():
        retain_true.extend(class_data["true_similarities"])
        retain_other.extend(class_data["other_similarities"])

    forget_true = np.array(forget_true)
    forget_other = np.array(forget_other)
    retain_true = np.array(retain_true)
    retain_other = np.array(retain_other)

    logger.info("=" * 60)
    logger.info("Similarity Statistics")
    logger.info("=" * 60)
    logger.info("Forget Set:")
    logger.info(f"  True Class:  mean={forget_true.mean():.4f}, std={forget_true.std():.4f}, count={len(forget_true)}")
    logger.info(f"  Other Class: mean={forget_other.mean():.4f}, std={forget_other.std():.4f}, count={len(forget_other)}")

    logger.info("Retain Set:")
    logger.info(f"  True Class:  mean={retain_true.mean():.4f}, std={retain_true.std():.4f}, count={len(retain_true)}")
    logger.info(f"  Other Class: mean={retain_other.mean():.4f}, std={retain_other.std():.4f}, count={len(retain_other)}")

    if "overall_statistics" in data:
        logger.info("Overall Statistics:")
        logger.info(f"  Forget Set: {data['overall_statistics']['forget_set']['total_samples']} samples "
                    f"from {data['overall_statistics']['forget_set']['num_classes']} classes: "
                    f"{data['overall_statistics']['forget_set']['classes']}")
        logger.info(f"  Retain Set: {data['overall_statistics']['retain_set']['total_samples']} samples "
                    f"from {data['overall_statistics']['retain_set']['num_classes']} classes")

    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#d0d0d0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    all_values = np.concatenate([forget_true, forget_other, retain_true, retain_other])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    padding = 0.03 * (max_val - min_val + 1e-8)

    ax1.set_facecolor("#f2f2f2")
    ax2.set_facecolor("#f2f2f2")

    ax1.set_xlim([min_val - padding, max_val + padding])
    ax2.set_xlim([min_val - padding, max_val + padding])

    bins = args.bins

    ax1.hist(
        forget_true,
        bins=bins,
        density=True,
        alpha=0.6,
        label="True Class",
        edgecolor="#1f3b73",
        linewidth=0.6,
        color="#1f3b73"
    )

    ax1.hist(
        forget_other,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Other Class",
        edgecolor="#c0392b",
        linewidth=0.6,
        color="#c0392b"
    )

    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Density")
    ax1.set_title("Forget Set", fontsize=14, pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc='upper left')

    ax1.axvline(forget_true.mean(), color="#1f3b73", linestyle="--", linewidth=1.2, alpha=0.8)
    ax1.axvline(forget_other.mean(), color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8)

    ax2.hist(
        retain_true,
        bins=bins,
        density=True,
        alpha=0.6,
        label="True Class",
        edgecolor="black",
        linewidth=0.6,
        color="#1f3b73"
    )

    ax2.hist(
        retain_other,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Other Class",
        edgecolor="black",
        linewidth=0.6,
        color="#c0392b"
    )

    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Density")
    ax2.set_title("Retain Set", fontsize=14, pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, loc='upper left')

    ax2.axvline(retain_true.mean(), color="#1f3b73", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.axvline(retain_other.mean(), color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved to: {args.output_path}")
    plt.show()

    logger.info("=" * 60)
    logger.info("Per-Class Detailed Statistics")
    logger.info("=" * 60)

    logger.info("--- Forget Set ---")
    for class_name, class_data in data["forget_set"].items():
        if class_data['num_samples'] > 0:
            logger.info(f"{class_name}:")
            logger.info(f"  num_samples: {class_data['num_samples']}")
            logger.info(f"  true_class_similarity: {class_data['mean_similarity']:.4f} +/- {class_data['std_similarity']:.4f}")
            logger.info(f"  other_class_similarity: {class_data['mean_other_similarity']:.4f} +/- {class_data['std_other_similarity']:.4f}")
            logger.info(f"  discrimination_score: {class_data['discrimination_score']:.4f}")

    logger.info(f"--- Retain Set (top {args.retain_top_n}) ---")
    for i, (class_name, class_data) in enumerate(data["retain_set"].items()):
        if i >= args.retain_top_n:
            break
        if class_data['num_samples'] > 0:
            logger.info(f"{class_name}:")
            logger.info(f"  num_samples: {class_data['num_samples']}")
            logger.info(f"  true_class_similarity: {class_data['mean_similarity']:.4f} +/- {class_data['std_similarity']:.4f}")
            logger.info(f"  other_class_similarity: {class_data['mean_other_similarity']:.4f} +/- {class_data['std_other_similarity']:.4f}")
            logger.info(f"  discrimination_score: {class_data['discrimination_score']:.4f}")


if __name__ == "__main__":
    main()