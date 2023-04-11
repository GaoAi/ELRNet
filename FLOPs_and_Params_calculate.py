from ptflops import get_model_complexity_info
from models.MyNetworks import ELRNet
from models import ERFNet,EDANet,ENet,UNet,SegNet,FCN,ESFNet
from configs.config import MyConfiguration
from models.FCN import VGGNet

config = MyConfiguration()

# net = ERFNet.ERFNet(config= config)
# net =  ENet.ENet(num_classes=2)
# net = SegNet.SegNet(config= config)
net =  ELRNet.ELRNet(config= config)

flops,params = get_model_complexity_info(net,(3,512,512),as_strings=True, print_per_layer_stat=True)
print('Computational complexity(Flops):{}'.format(flops))
print('Number of parameters(Params):'+ params)