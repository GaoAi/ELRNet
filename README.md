# ELRNet：A Novel Lightweight Network for Building Extraction from High-Resolution Remote Sensing Images

Here, we provide the pytorch implementation of the paper: Research on Building Extraction from VHR Remote Sensing
Imagery Using Efficient Lightweight Residual Network.

![image-20210228153142126](./Architecture.tif)

### Clone the Repository
```
git clone https://github.com/mrluin/ESFNet-Pytorch.git
```
```
cd ./ESFNet-Pytorch
```


### Installation using Conda
```
conda env create -f environment.yml
```
```
conda activate esfnet
```

### Sample Dataset
For training, you can use as an example the [WHU Building Datase](study.rsgis.whu.edu.cn/pages/download/).

You would need to download the cropped aerial images. `The 3rd option`

### Directory Structure
```
Directory:
            #root | -- train 
                  | -- valid
                  | -- test
                  | -- save | -- {model.name} | -- datetime | -- ckpt-epoch{}.pth.format(epoch)
                            |                               | -- best_model.pth
                            |
                            | -- log | -- {model.name} | -- datetime | -- history.txt
                            | -- test| -- log | -- {model.name} | --datetime | -- history.txt
                                     | -- predict | -- {model.name} | --datetime | -- *.png
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
