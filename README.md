# Building Continual Memory Models by Piecewise Linear Fine-tuning

Task-free continual learning (with known tasks during training and test time) on a frozen, pretrained ViT-B/16. Instead of
training adapters by gradient descent, each batch gets a **closed-form, linearized
IA3 update**: the network is linearized around the pretrained weights and the IA3
scaling vectors are found by solving a ridge-regularized least-squares problem.

## Method overview

For each training batch of the current task:

1. **Head warm start.** A single Adam step (`lr = 1e-3`) updates the classification head. The backbone stays frozen.
2. **Jacobian.** `torch.func.jacrev` computes the Jacobian of the target-class softmax probability with respect to the IA3 vectors (`l_k`, `l_v`, `l_ff`). The Jacobian is taken on a frozen copy of the backbone.
3. **Ridge solve.** For each IA3 block, solve `(AᵀA + λI) w = Aᵀz`, where `A` is the flattened Jacobian (`λ = 1.0`).
4. **Soft-target refinement.** The target `z` is moved step by step toward `b = 2 − p` with step size `η = 0.03`. The update with the best batch accuracy is kept, with early stopping after 3 iterations that don't improve.
5. **Apply.** Each IA3 parameter is set to `1 + Δ`.

At each task boundary, the task's IA3 update is saved to disk and the IA3 vectors are reset to identity. Evaluation reloads every saved update and fills the accuracy matrix.

These hyperparameters are constants at the top of [main.py](main.py): `TANGENT_ETA`, `TANGENT_RIDGE_LAMBDA` and `HEAD_LR`.

## Repository structure

```
main.py                 # Continual training loop (entry point)
evaluate.py             # Per-task evaluation + standalone evaluation from checkpoints
configs/                # YAML experiment configs (one per benchmark/seed)
lib/
  config.py             # Default config (EasyDict) + YAML / CLI override merging
  train.py              # Head training step and closed-form tangent IA3 update
  ia3.py                # Helpers to collect / apply / reset IA3 scaling vectors
  checkpointing.py      # Save/load per-task IA3 updates and ridge accumulators
  utils.py              # Logging, evaluate_till_now, metrics
data/                   # Dataset classes and continual (split) dataloaders
models/                 # timm ViT with IA3 vectors, registered as vit_base_patch16_224
utils/                  # Vision helpers
```

## Installation

```bash
conda create -n tangent-cl python=3.10 -y
conda activate tangent-cl
pip install -r requirements.txt
pip install "timm==0.6.12" pyyaml numpy
```

> **Note:** The ViT code uses `timm.models.helpers.resolve_pretrained_cfg`. That API exists in timm 0.6.x and was removed in later versions, so use timm 0.6.x.

A CUDA GPU is required. Select it with `gpu_ids` in the config.

## Datasets

Datasets are read from `dtask.data_path` (default `./local_datasets/`).

- **CIFAR-10 / CIFAR-100:** downloaded automatically by torchvision.
- **ImageNet-R:** downloaded automatically on first use.
- Loaders also exist for MNIST, Fashion-MNIST, SVHN, NotMNIST, TinyImageNet, VTAB and others (see [data/dataset.py](data/dataset.py)). Some of them (`download=False`) need the data placed in `data_path` by hand.

## Benchmarks

| Config | Dataset | Classes | Tasks | Classes / task |
|---|---|---|---|---|
| [configs/cifar-10-seed-15.yml](configs/cifar-10-seed-15.yml) | Split CIFAR-10 | 10 | 5 | 2 |
| [configs/cifar-100-seed-15.yml](configs/cifar-100-seed-15.yml) | Split CIFAR-100 | 100 | 20 | 5 |
| [configs/imgn-r-seed-15.yml](configs/imgn-r-seed-15.yml) | Split ImageNet-R | 200 | 20 | 10 |

All configs use a pretrained `vit_base_patch16_224`, batch size 16 and seed 15.

## Usage

### Training

```bash
python main.py --cfg configs/cifar-100-seed-15.yml
```

Override any config key with `--set KEY VALUE ...`. It must be the last argument:

```bash
python main.py --cfg configs/cifar-100-seed-15.yml --set seed 42 dtask.output_dir results/c100_seed42/
```

Accuracy is evaluated after each task.

### Evaluation from checkpoints

```bash
python evaluate.py --cfg configs/cifar-100-seed-15.yml
```

This loads the final head and every per-task IA3 update from `dtask.output_dir` and recomputes the accuracy matrix.

## Outputs

- **Logs:** `logs/<config-name>_seed<seed>_more_trial.log`. Includes the per-batch accuracy of the global update and the accuracy matrix after each task.
- **Checkpoints** (in `dtask.output_dir`):
  - `ia3_task_<k>/global_updates.pth`: the IA3 update for task `k`.
  - `ia3_task_<k>/global_At_A.pth`, `global_AtA_w.pth`: the ridge accumulators.
  - `ia3_task_<last>/head_weight.pth`, `head_bias.pth`: the final classification head.
  - `class_mask.pth`: the classes in each task.
  - `global_acc_matrix.pth`: the task × task accuracy matrix.

> **Note:** Before starting a new run, point `dtask.output_dir` at an empty directory. Per-task evaluation counts the existing `ia3_*` folders there, so leftovers from an earlier run will corrupt the results.

## Acknowledgements

The ViT model registration and the continual dataloaders are adapted from [L2P](https://github.com/google-research/l2p) and its PyTorch port, which build on [timm](https://github.com/huggingface/pytorch-image-models).


