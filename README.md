# tcde_unlearn
# TCDE-Unlearn

**TCDE** (Trace-Consistent Distributional Erasure) is a CLIP unlearning framework
that eliminates forgetting traces while preserving unlearning performance.

Our implementation is built on top of [OpenCLIP](https://github.com/mlfoundations/open_clip).
The overall training usage is consistent with OpenCLIP, with an additional `--cliperase`
flag that enables the CLIPErase-based unlearning procedure.

---

## Method

Standard CLIP unlearning removes the target concept from the model but often leaves
**forgetting traces** — residual signals in the embedding space that an adversary can
still exploit to recover the forgotten concept.

TCDE addresses this by adding distributional and uniformity constraints on top of
the CLIPErase objective, so that the erased concept is not only removed from the
decision boundary but also made **trace-consistent** in the feature distribution.

Loss terms:

| Term | Flag | Role |
|---|---|---|
| Retain loss | `--lambda-rm` | Preserve performance on retain set |
| CLIPErase loss | `--lambda-cm` | Remove target concept |
| Uniformity loss | `--lambda-unif` | Prevent residual clustering of forget features |
| Distribution loss | `--lambda-dist` | Align forget distribution with background distribution |

---

## Repository Structure

tcde_unlearn/
├── model_configs/ # OpenCLIP model configs (ViT-B-32, ViT-L-14, ...)
├── src/
│ ├── open_clip/ # OpenCLIP model / loss / tokenizer
│ └── open_clip_train/ # Training entry point (main.py)
├── test/ # Evaluation and analysis scripts
├── .gitignore
└── README.md

## Usage Example on CIFAR Dataset

Below is an example of how to execute the code on the CIFAR dataset for unlearning the concept "apple":

```
python main.py \
    --dataset-type "cifar" \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-6 \
    --wd 0.01 \
    --model "ViT-B-32" \
    --pretrained "" \
    --name "clip" \
    --workers 8 \
    --precision amp \
    --log-every-n-steps 10 \
    --save-frequency 5 \
    --zeroshot-frequency 1 \
    --val-frequency 1 \
    --accum-freq 1 \
    --cliperase \
    --lambda-rm 1.0 \
    --lambda-cm 3.0 \
    --lambda-unif 1.0 \
    --lambda-dist 1.0 \
    --cliperase-temperature 0.07 \
    --eval-forget-set \
    --save-forget-metrics \
    --val-split 0.3 \
    --logs "" \
    --device "cuda" \
    --forget-rule "keyword_based" \
    --forget-keywords "apple" \
    --aug-cfg use_timm=True re_prob=0.25 color_jitter=0.4 \
    --seed 42
```

