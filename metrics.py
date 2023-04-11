import torch
import numpy as np

'''
    Tensor has no attr 'copy' should use 'clone'
    pred 's requires_grad = True
    .clone().cpu().numpy()
    该评价指标应该是召回率的表达:TP/(TP+FN)
'''
def Accuracy(pred, label):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        # print("pred----------",pred)
        label = label.view(-1)
        # print("label----------",label)
        # ignore 0 background
        valid = (label > 0).long()
        # print("--------valid---------",valid)
        # convert to float() 做除法的时候分子和分母都要转换成 float 如果是long 则会出现zero
        # .long() convert boolean to long then .float() convert to float
        # 合法的 pred == label的 pixel总数
        acc_sum = torch.sum(valid * (pred == label).long()).float()
        # 合法的pixel总数
        pixel_sum = torch.sum(valid).float()
        # epsilon
        acc = acc_sum / (pixel_sum + 1e-10)
        return acc

# TP/(TP+FP)
def Precision(pred, label):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        label = label.view(-1)
        # print("pred-----------",pred)
        # print("label----------",label)
        # ignore 0 background 针对所有正样本
        valid = (label > 0).long()
        valid_pred = (pred > 0).long()
        # print("valid--------",valid)
        # convert to float() 做除法的时候分子和分母都要转换成 float 如果是long 则会出现zero
        # .long() convert boolean to long then .float() convert to float
        # 合法的 pred == label的 pixel总数
        tp_sum = torch.sum(valid * (pred == label).long()).float()
        # 合法的pixel总数
        pixel_sum = torch.sum(valid_pred).float()
        precision = tp_sum / (pixel_sum + 1e-10)
        return precision

# F1_score是基于精确率和召回率的调和平均
# F1_score = (2*P*R)/(P+R)
def F1_score(pred, label):
    P = Precision(pred, label)
    R = Accuracy(pred, label)
    f1 = (2 * P * R)/(P + R)
    return f1


def MIoU(pred, label, nb_classes):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        label = label.view(-1)
        iou = torch.zeros(nb_classes ).to(pred.device)
        for k in range(1, nb_classes):
            # pred_inds ,target_inds boolean map
            pred_inds = pred == k
            target_inds = label == k
            intersection = pred_inds[target_inds].long().sum().float()
            union = (pred_inds.long().sum() + target_inds.long().sum() - intersection).float()

            iou[k] = (intersection/ (union+1e-10))

        return (iou.sum()/ (nb_classes-1))


if __name__ == '__main__':
    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],])
    b = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])

    a = torch.from_numpy(a).long()
    b = torch.from_numpy(b).long()

    # intersection = 16, union =
    acc = Accuracy(a,b)
    # acc = my_Accuracy(a,b)
    print("acc-------",acc)
