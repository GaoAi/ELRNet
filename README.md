# ELRNet：A Novel Lightweight Network for Building Extraction from High-Resolution Remote Sensing Images

Here, we provide the pytorch implementation of the paper: Research on Building Extraction from VHR Remote Sensing
Imagery Using Efficient Lightweight Residual Network.

![image-20210228153142126](./Architecture.tif)

## Clone this Repository
```
git clone https://github.com/GaoAi/ELRNet.git
cd ./ELRNet
```

## Virtual Environment Creation (if you need it)
```
conda create -n elrnet python==3.8
conda activate elrnet
```

## Installation
```
- python
- pytorch
- torchvision
- cudatoolkit
- cudnn
- visdom
- pillow
- tqdm
- opencv
- pandas
```

## Dataset Preparation
You can take the public WHU building extraction dataset as an example for training and testing. 
### Data Download
WHU_Building_dataset: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html(The 3rd option)
### Dataset Directory Structure
```
Directory:
            #root | -- train | -- image 
                             | -- label 
                  | -- val | -- image 
                           | -- label    
                  | -- test | -- image 
                            | -- label
                  | -- save | -- {model_name} | -- datetime | -- ckpt-epoch{}.pth.format(epoch)
                            |                               | -- best_model.pth
                            |
                            | -- log | -- {model_name} | -- datetime | -- history.txt
                            | -- test| -- log | -- {model_name} | --datetime | -- history.txt
                                     | -- predict | -- {model_name} | --datetime | -- *.png/tif

```
Note: The save directory is automatically generated after the model starts training.

## Training
### Configure config.cfg
1. set `root_dir` to your dataset directory,For example `root_dir = /home/ELRNet/WHU_Building_dataset`
2. set `nb_classes` to be the number of class in your dataset.
3. set `epochs` to control the length of the training phase.
### Start training
（1）start visdom server
```
python -m visdom.server -env_path='./visdom_log/' -port=8096
```

（2）start train
```
python train.py
```
`-env_path` is where the visdom logfile store in, and `-port` is the port for `visdom`. You can also change the `-port` in `train.py`.

The change process of model training can be viewed online visdom:http://localhost:8096/
### Predict and Evaluate
If you want to generate the current prediction result image during the model training or want to predict the image after the model training, you can set and run `demo.py`. You need to set `-weight` in `demo.py` to the directory where the current weight file is located.
```
python demo.py
```
### Model Parameter Quantity and Computational Complexity Calculation
If you want to calculate the number of parameters and computational complexity of your model, you can run the following script:
```
python FLOPs_and_Params_calculate.py
```
### Draw the Training Process Change Curve
If you want to redraw your own training process change curve, you can run the following script:
```
python vis_loss_and_iou.py
```
**If my work give you some insights and hints, star me please! Thank you~**
