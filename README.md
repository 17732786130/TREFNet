# TREF-Net

PyTorch implementation of TREF-Net for change detection in paired remote sensing images. The model has 4.86 million parameters.

## Installation

Use Python 3.10 or 3.11. Install a matching PyTorch and torchvision pair for your hardware (for example, PyTorch 2.4 and torchvision 0.19), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset preparation

The table lists the train, validation, and test splits used in the paper. SYSU-CD is used only as the target test set for cross-scene evaluation.

| Dataset | Split lists | Train / validation / test |
| --- | --- | --- |
| BCDD | [splits/BCDD](splits/BCDD) | 1,071 / 189 / 690 |
| LEVIR-CD | [splits/LEVIR-CD](splits/LEVIR-CD) | 7,120 / 1,024 / 2,048 |
| WHU-CD | [splits/WHU-CD](splits/WHU-CD) | 5,460 / 779 / 1,510 |
| SYSU-CD | [test list](splits/SYSU-CD/test.txt) | — / — / 4,000 |

### Reproducing the BCDD evaluation

1. **Prepare the data.** Use the [BCDD split lists](splits/BCDD) to place the prepared image pairs and masks under the corresponding directories. Only the validation and test splits are needed when evaluating the released checkpoint.

   ```text
   dataset/BCDD/
     train/
       A/
       B/
       label/
     val/
       A/
       B/
       label/
     test/
       A/
       B/
       label/
   ```

   The two images in each pair must have the same filename. The mask must have the same filename stem and may use a different extension. The scripts resize the 512 × 512 BCDD tiles to 256 × 256 before feeding them to the model.

2. **Download the trained model.** Download `BCDD.pth` from [GitHub Releases](https://github.com/17732786130/TREFNet/releases) and save it as `checkpoints/BCDD.pth`. This checkpoint was trained on BCDD with seed 2333.

3. **Run the evaluation.** From the repository root, run the first command to select the threshold that maximizes F1 on the BCDD validation split. The second command uses that threshold on the BCDD test split. Both commands average aligned predictions from the original, horizontally flipped, and vertically flipped image pairs.

   ```bash
   python tools/test.py --data-root dataset/BCDD --checkpoint checkpoints/BCDD.pth --split val --manifest splits/BCDD/val.txt --select-threshold --output_dir runs/BCDD_validation
   python tools/test.py --data-root dataset/BCDD --checkpoint checkpoints/BCDD.pth --split test --manifest splits/BCDD/test.txt --threshold-file runs/BCDD_validation/best_threshold_tta.txt --output_dir runs/BCDD_test
   ```

   Add `--onGPU false` for CPU execution.

### Training on BCDD

To train your own model, prepare the training split as well and run:

```bash
python tools/train.py --data-root dataset/BCDD --output_dir runs/BCDD_seed2333 --seed 2333 --batch_size 4 --lr 0.0004 --max_epochs 300
```

The script saves the checkpoint with the highest validation F1 as `best_model.pth`.

For LEVIR-CD tiling, see [tools/prepare_levir.py](tools/prepare_levir.py). For cross-scene testing, use a checkpoint trained on the source dataset, its source validation threshold, and the SYSU-CD test manifest.
