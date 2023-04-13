# 6.5930-Final-Project
Analysis of Hyperspectral Deep Learning Models for Hardware

## Requirements

This tool is compatible with Python 3.9.16 and [PyTorch](http://pytorch.org/) 1.12.1.

## Hyperspectral dataset: Indian Pines Dataset
[AVIRIS Indian Pines Dataset](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines) is acquired by the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) sensor over the Indian Pines test site in North-western Indiana in June 1992. The default dataset folder to save the dataset `./Datasets/`, although this can be modified at runtime using the `--folder` arg.

## Usage
`python main.py --model hamida --dataset IndianPines --training_sample 0.1 --cuda 0`