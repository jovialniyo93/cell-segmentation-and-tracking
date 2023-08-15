![Python](https://img.shields.io/badge/python-v3.8-blue)
![Pytorch](https://img.shields.io/badge/Pytorch-V1.6-orange)
![CV2](https://img.shields.io/badge/CV2-V4.8-brightgreen)
![pandas](https://img.shields.io/badge/Pandas-V1.4.2-ff69b4)
![numpy](https://img.shields.io/badge/%E2%80%8ENumpy-V1.20.2-success)
![releasedate](https://img.shields.io/badge/release%20date-August2023-red)
![Opensource](https://img.shields.io/badge/OpenSource-Yes!-6f42c1)


# A Novel Deep Learning Approach Featuring Graph-Based Tracking for Simultaneous Cell Segmentation and Tracking #

The code in this repository is supplementary to our future publication "A Novel Deep Learning Approach Featuring Graph-Based Tracking for Simultaneous Cell Segmentation and Tracking" 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB
* This project is writen in Python 3 and makes use of Pytorch. 

## Installation
In order to get the code, either clone the project, or download a zip file from GitHub.

Clone the Cell detection and Tracking repository:
```
https://github.com/jovialniyo93/cell-segmentation-and-tracking.git
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the Cell Segmentation and Tracking repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
```
```
conda env create -f requirements.yml
```
Activate the virtual environment cell_detection_and_tracking_ve:
```
conda activate cell_detection_and_tracking_ve
```

# How to train and test our model

```tracking_and_segmentation``` ```PhC-C2DL-PSC``` folders contains all scripts used to train and test models for iPS data and Public datasets, respectively.  ```graph``` folder contains scripts to generate trace and track cells.

<br/>

## Independent dataset

```PhC-C2DL-PSC``` folder can be replaced by any 2D data and use it as the independent dataset to test the model transferability.

In this section, it is described how to reproduce the detection and tracking results on public dataset using our method. Download the Cell Tracking Challenge training data sets [Fluo-N2DH-GOWT1](http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip), [Fluo-C2DL-MSC](http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip), [PhC-C2DH-U373](http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip) , [PhC-C2DL-PSC](http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip) Unzip the data sets into the respective folders. Download the [Evaluation software](http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip) from the Cell Tracking Challenge and unzip it in the repository. 


# Project Collaborators and Contact

**Authors:** Keliang Zhao, Jovial Niyogisubizo, Linxia Xiao, Yi Pan, Yanjie Wei

**Created by:** Ph.D. student: Jovial Niyogisubizo 
Department of Computer Applied Technology,  
Center for High Performance Computing, Shenzhen Institute of Advanced Technology, CAS. 

For more information, contact:

* **Prof Yanjie Wei**  
Shenzhen Institute of Advanced Technology, CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
yj.wei@siat.ac.cn


* **Jovial Niyogisubizo**  
Shenzhen Institute of Advanced Tech., CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
jovialniyo93@gmail.com

## License ##
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
