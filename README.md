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
