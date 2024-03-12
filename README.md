# ReadMe

This repository contains the official implementation of the methods presented in the paper \``**Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization**'' **(CVPR 2024)**. The paper addresses the vulnerability of no-reference image quality assessment models to adversarial attacks and proposes a novel gradient norm regularization technique to enhance their robustness.  

It contains the training of the HyperIQA model with the NT (Norm regularization Training) strategy, together with the attack code with the FGSM method. We express our gratitude to the authors of the CVPR 2020 paper "[Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)" for sharing the source code for training the HyperIQA model.

## Dependencies

- python
- torch
- torchvision
- Pillow
- numPy
- scipy

Or using the following commands:

```
conda create -n gradnormIQA pip python=3.7.10
conda activate gradnormIQA
pip install -r requirements.txt
```

## Usage

### Testing Quality Score and Gradient Norm for a Single Image

First, the checkpoints should be downloaded from [checkpoints_GoogleDrive](https://drive.google.com/drive/u/2/folders/1TvQxZY6290IlkS0iBw1-c277iPkc6OLI), and then move them into the checkpoints folder.

To predict image quality and calculate the gradient norm with our baseline model and baseline+NT model trained on the LIVEC Dataset, use the following command:

```
python quality_and_norm_demo.py --img_name 123.bmp
```

This will output:

1. An $\ell_1$ norm of the output gradient with respect to the input image.
2. A predicted quality score, where a higher value indicates better image quality.

### Training & Testing on IQA databases

Notice that the [LIVEC](https://live.ece.utexas.edu/research/ChallengeDB/index.html) dataset should be downloaded first, and the path to the dataset should be modified in below commands.

To train and test the baseline HyperIQA model on the LIVEC dataset, execute:

```
python train_test_IQA_sparsity.py --dataset_path /YOURPATH/ChallengeDB_release/
```

To train and test the HyperIQA+NT model on the LIVEC Dataset, execute:

```
python train_test_IQA_sparsity.py --if_grad --dataset_path /YOURPATH/ChallengeDB_release
```

For training other IQA models ([DBCNN](https://github.com/zwx8981/DBCNN-PyTorch), [LinearityIQA](https://github.com/lidq92/LinearityIQA), [MANIQA](https://github.com/IIGROUP/MANIQA)), the NT strategy is easy to implement. The *loss_grad* in *HyperIQASolver.py* should be add in the training loss of the original model.

### Attacking Baseline or Baseline+NT Model with the FGSM Attack

To attack the baseline HyperIQA model with an FGSM attack, run:

```
python FGSM_demo.py --img_name 123.bmp --attacked_model baseline
```

To attack the HyperIQA+NT model with an FGSM attack, use:

```
python FGSM_demo.py --img_name 123.bmp
```
