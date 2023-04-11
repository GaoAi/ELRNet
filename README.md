# ELRNet：A Novel Lightweight Network for Building Extraction from High-Resolution Remote Sensing Images

Here, we provide the pytorch implementation of the paper: Research on Building Extraction from VHR Remote Sensing
Imagery Using Efficient Lightweight Residual Network.

![image-20210228153142126](./Architecture.tif)

## Clone this Repository
```
git clone https://github.com/GaoAi/ELRNet.git
cd ./ELRNet
```

## Virtual environment creation (if you need it)
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

## Dataset
You can take the WHU dataset as an example for testing. The download address of WHU is: http://gpcv.whu.edu.cn/data/building_dataset.html 
### Directory Structure
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

The save directory is automatically generated after the model starts training.
```
### Training
1. set `root_dir` in `./configs/config.cfg`, change the root_path like mentioned above.
2. set `divice_id` to choose which GPU will be used.
3. set `epochs` to control the length of the training phase.
4. setup the `train.py` script as follows:
```
python -m visdom.server -env_path='./visdom_log/' -port=8097 # start visdom server
python train.py
```
`-env_path` is where the visdom logfile store in, and `-port` is the port for `visdom`. You could also change the `-port` in `train.py`.



**If my work give you some insights and hints, star me please! Thank you~**
