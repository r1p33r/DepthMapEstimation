<img src="/graphical_abstract.png">

# Using CycleGAN and pix2pix in PyTorch for Depth Map Estimation

Note: The original code is available at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

# Prerequisites

Linux, Windows or macOS.
- Python 3, We use anaconda package.
- Matplotlib and OpenCV (to process images).
- Pytorch: https://pytorch.org/get-started/locally/.
- CPU or NVIDIA GPU + CUDA CuDNN.

# Install Instructions

- **1 Step**: clone the original repository.
```bash
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```
- **2 Step**: Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

 # Datasets

- **Pix2Pix**:

  Create folder /path/to/data with subfolders A and B. A and B should each have their own subfolders train, val, test, etc. In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B. Repeat same for other     data splits (val, test, etc).

  Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.

  Once the data is formatted this way, call:

```bash
!python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

  This will combine each pair of images (A,B) into a single image file, ready for training.

- **CycleGAN**:

  Create a dataset folder under /dataset for your dataset.

  Create subfolders testA, testB, trainA, and trainB under your dataset's folder. Place any images you want to transform from a to b (cat2dog) in the testA folder, images you want to transform from b to a (dog2cat) in the testB folder, and do the same for the trainA and trainB folders.

```bash
!python datasets/make_dataset_aligned.py --dataset-path /media/pc/DATA/Gan-test/dataset_CycleGAN
```
