import json
import matplotlib.pyplot as plt  

iteration_result = []
train_loss_result = []
val_loss_result =[]
train_IoU_result =[]
val_IoU_result =[]

# v3版本残差块
# json_filename = './my_visdom.json'

# v4版本残差块
json_filename = './v7_eca.json'

# 读取json文件内容,返回字典格式
with open(json_filename,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # print('这是文件中的json数据：',json_data)
    iteration_result = json_data["iteration"]
    train_loss_result = json_data["train_loss"]
    val_loss_result = json_data["val_loss"]
    train_IoU_result = json_data["train_IoU"]
    val_IoU_result = json_data["val_IoU"]

   
x1 = iteration_result
y1 = train_loss_result
y2 = val_loss_result
y3 = train_IoU_result
y4 = val_IoU_result

# #创建画布
plt.figure()
#绘制第一条数据线、线颜色、标签名字
plt.plot(x1,y1,color='#828282',label='train loss',linewidth=2)
#绘制第二条数据线、线颜色、标签名字
plt.plot(x1,y2,color='#FF4500', label='val loss',linewidth=2)
#绘制第三条数据线、线颜色、标签名字
plt.plot(x1,y3,color='#0064FF', label='train IoU',linewidth=2)
#绘制第四条数据线、线颜色、标签名字
plt.plot(x1,y4,color='#FF9614', label='val IoU',linewidth=2)

#显示标题
# plt.title("loss_and_IoU")
#显示图例
plt.legend()

# plt.xlabel('iteration')
# plt.ylabel('value')
plt.savefig('loss_and_IoU_v7.png')
plt.show()
