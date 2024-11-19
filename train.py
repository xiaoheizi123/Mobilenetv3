import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import mobilenet_v3_large
import os
import json
import torchvision.models.mobilenet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

data_transform = {
    "train" : transforms.Compose([
                                  transforms.Resize((256,256)),
                                  # transforms.RandomResizedCrop(224),   # 随机裁剪
                                  transforms.RandomHorizontalFlip(),   # 随机翻转
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val" : transforms.Compose([
                                transforms.Resize((256,256)),      # 长宽比不变，最小边长缩放到256
                                # transforms.CenterCrop(224),  # 中心裁剪到 224x224
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

#   获取数据集所在的根目录
#   通过os.getcwd()获取当前的目录，并将当前目录与".."链接获取上一层目录
# data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# data_root = os.path.abspath(os.path.join(os.getcwd()))

#   获取花类数据集路径
# image_path = data_root + "/datasets"
# image_path = "/media/zhengr/8t1/workspace/Documents/projects/2024/datasets/第一批_图片"
image_path = "/media/zhengr/8t1/workspace/Documents/projects/2024/datasets/视频分类数据/第2_3_4批_图片"

#   加载数据集
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])

#   获取训练集图像数量
train_num = len(train_dataset)

#   获取分类的名称
#   {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_list = train_dataset.class_to_idx

#   采用遍历方法，将分类名称的key与value反过来
cla_dict = dict((val, key) for key, val in flower_list.items())

#   将字典cla_dict编码为json格式
json_str = json.dumps(cla_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

batch_size = 64
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=64)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = DataLoader(validate_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=64)

#   定义模型
net = mobilenet_v3_large(num_classes=5)   # 实例化模型
net.to(device)
model_weight_path = "./weights/mobilenet_v3_large-5c1a4163.pth"
#   载入模型权重
pre_weights = torch.load(model_weight_path)
#   删除分类权重
pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
#   冻结除最后全连接层以外的所有权重
for param in net.features.parameters():
    param.requires_grad = False

loss_function = nn.CrossEntropyLoss()   # 定义损失函数
#pata = list(net.parameters())   # 查看模型参数
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 定义优化器

#   设置存储权重路径
save_path = './weights/mobilenetV3_large.pth'
best_acc = 0.0
epoches = 100
for epoch in range(epoches):
    # train
    net.train()  # 用来管理Dropout方法：训练时使用Dropout方法，验证时不使用Dropout方法
    running_loss = 0.0  # 用来累加训练中的损失
    for step, data in enumerate(train_loader, start=0):
        #   获取数据的图像和标签
        images, labels = data

        #   将历史损失梯度清零
        optimizer.zero_grad()

        #   参数更新
        outputs = net(images.to(device))                   # 获得网络输出
        loss = loss_function(outputs, labels.to(device))   # 计算loss
        loss.backward()                                    # 误差反向传播
        optimizer.step()                                   # 更新节点参数

        #   打印统计信息
        running_loss += loss.item()
        #   打印训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()  # 关闭Dropout方法
    acc = 0.0
    #   验证过程中不计算损失梯度
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            #   acc用来累计验证集中预测正确的数量
            #   对比预测值与真实标签，sum()求出预测正确的累加值，item()获取累加值
            print('test_labels is:', test_labels)
            print(' predict_y  is:', predict_y )
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        #   如果当前准确率大于历史最优准确率
        if accurate_test > best_acc:
            #   更新历史最优准确率
            best_acc = accurate_test
            #   保存当前权重
            torch.save(net.state_dict(), save_path)
        #   打印相应信息
        print("[epoch %d] train_loss: %.3f  test_accuracy: %.3f"%
              (epoch + 1, running_loss / step, acc / val_num))

print("Finished Training")
