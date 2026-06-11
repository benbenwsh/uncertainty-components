# Localising the Uncertainty Components Behind Verbalised Confidence in LLM

This repository contains the source code for all experiments in my Final Year Project (FYP) report, submitted to complete the MEng Computing degree at Imperial College London.

## Research question

**How do LLMs come up with verbalised confidence?**

When a model is asked to state a guess together with a numeric confidence (for example, “Probability: 0.85”), that number must be produced by internal computation. This project studies where that signal lives in the network and what interventions change it.

## Goals

1. **Localise uncertainty components** — identify the layers, heads, submodules, and token positions whose activations carry information about verbalised confidence.
2. **Attain a mechanistic understanding of verbalised confidence** — use probes, ablations, steering, and patching to test how confidence is computed and expressed during generation.

## Repository structure

The codebase is organised into experiment modules. Each top-level directory has a matching `*.sh` SLURM script at the repo root that shows a typical launch command.

### Data generation and processing

| Directory | Purpose |
|-----------|---------|
| [`semantic_uncertainty/`](semantic_uncertainty/) | Generate model answers with a verbalised-confidence prompt, collect hidden states and embeddings (layer, head, block, concat), and compute uncertainty baselines. Entry point: `generate_answers_with_confidence_h5.py`. |
| [`process_generations/`](process_generations/) | Store and organise processed generation runs (HDF5 / JSON), including train/test splits used by downstream probes and ablations. |

### Probing verbalised confidence

| Directory | Purpose |
|-----------|---------|
| [`verbalised_confidence_probes/`](verbalised_confidence_probes/) | Train linear and ridge regressors (and related probes) to predict verbalised confidence from hidden states — layerwise, headwise, and multi-token probability-span features. |
| [`semantic_entropy_probes/`](semantic_entropy_probes/) | Notebooks and utilities from the [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) codebase, retained for semantic-uncertainty baselines and comparison experiments. |
| [`linear_probe_direction/`](linear_probe_direction/) | Apply trained probe directions as causal interventions during generation to test whether probe weights align with mechanisms that drive confidence. |

### Localisation via ablation and steering

| Directory | Purpose |
|-----------|---------|
| [`layerwise_mean_ablation/`](layerwise_mean_ablation/) | Replace activations at selected layers with mean activations from low- (or high-) confidence examples. |
| [`blockwise_mean_ablation/`](blockwise_mean_ablation/) | Same idea at attention / MLP subblock granularity. |
| [`headwise_mean_ablation/`](headwise_mean_ablation/) | Per-attention-head mean ablations over the probability span. |
| [`tokenwise_probability_mean_ablation/`](tokenwise_probability_mean_ablation/) | Token-position-specific mean ablations across the verbalised probability prefix. |
| [`layerwise_zero_ablation/`](layerwise_zero_ablation/) | Zero out residual stream activations layer by layer. |
| [`blockwise_zero_ablation/`](blockwise_zero_ablation/) | Zero ablation on attention and MLP subblocks. |
| [`mass_mean_probe/`](mass_mean_probe/) | Compute a high-minus-low confidence direction from probability-token activations and steer along it during generation. |
| [`subblock_mass_mean_probe/`](subblock_mass_mean_probe/) | Mass mean-direction steering at attention / MLP subblock resolution. |
| [`headwise_mass_mean_probe/`](headwise_mass_mean_probe/) | Mass mean-direction steering per attention head. |
| [`tokenwise_probability_mass_mean_probe/`](tokenwise_probability_mass_mean_probe/) | Token-wise steering across layers using the confidence direction. |
| [`kv_patching_ablation/`](kv_patching_ablation/) | Patch key/value caches from low-confidence runs into high-confidence runs (and variants) to test cache-level mechanisms. |
| [`ind_head_kv_patching/`](ind_head_kv_patching/) | Per-head KV patching for finer-grained causal tests. |

### Interpretability tools

| Directory | Purpose |
|-----------|---------|
| [`logit_lens/`](logit_lens/) | Decode intermediate layer representations at guess-token positions to inspect what each layer “believes” before the final answer. |
| [`logit_lens_improved/`](logit_lens_improved/) | Extended logit-lens analysis with additional controls and visualisation. |
| [`tuned_lens/`](tuned_lens/) | Train tuned affine maps from hidden states to vocabulary logits for sharper layer-wise readouts. |

## Setup and running experiments

Experiments require Python 3.11, PyTorch, and a CUDA GPU (7B models typically need ~24 GB VRAM). Dependencies are managed with conda.

From the repository root:

```bash
conda env update -f sep_enviroment.yaml
conda activate uncertainty_components
```

Most workflows follow the same pattern:

1. **Generate data** — run `semantic_uncertainty/generate_answers_with_confidence_h5.py` (or `ans_gen_slurm.sh`) to produce verbalised-confidence generations with saved embeddings.
2. **Train probes or run ablations** — call the relevant `run_*.py` script under the module directory, or submit the matching root-level `*.sh` SLURM script on a cluster.

Set `HUGGING_FACE_HUB_TOKEN` for model access. Some scripts also use Weights & Biases (`wandb`) for logging.

Example (layerwise verbalised-confidence probe training):

```bash
conda activate uncertainty_components
python verbalised_confidence_probes/h5_train_multitoken_headwise_verbalised_confidence_probe.py \
  --train_path process_generations/processed_generations_more_h5/<run>/train_verbalised_embeddings.h5 \
  --test_path process_generations/processed_generations_more_h5/<run>/validation_verbalised_embeddings.h5
```

Adjust paths, layer ranges, and sample counts to match the run IDs documented in the FYP report. Each root-level `*.sh` file contains a concrete example configuration.
