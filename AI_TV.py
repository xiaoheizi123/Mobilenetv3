import os
import json
import numpy as np
from glob import glob

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import mobilenet_v3_large


class VodeoTypeJudge:
    def __init__(self, weights_path='./mobilenetV3_large.pth', class_json_path='./class_indices.json',device='cpu', ):
        # model 等初始化
        self.weights_path = weights_path
        self.class_json_path = class_json_path
        self.device = device
        self.frame_res = []

        # create model
        self.model = mobilenet_v3_large(num_classes=5).to(self.device)

        # load model weights
        assert os.path.exists(self.weights_path), "file: '{}' dose not exist.".format(self.weights_path)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

        # prediction
        self.model.eval()

        self.data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # read class_indict
        assert os.path.exists(self.class_json_path), "file: '{}' dose not exist.".format(self.class_json_path)

        self.class_dict = {}
        with open(self.class_json_path, "r") as f:
            self.class_indict = json.load(f)

        pass

    def GetFrame(self, frame):
        # frame RGB格式
        # to form [N, C, H, W]
        img = self.data_transform(frame)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            self.frame_res.append(predict_cla)


    def GetResult(self):
        # 找到每个元素的唯一值及其出现次数
        values, counts = np.unique(np.array(self.frame_res), return_counts=True)
        # 找到出现次数最多的元素
        mode_value = values[np.argmax(counts)]
        type_is = self.class_indict[str(mode_value)]
        return type_is

if __name__ == '__main__':
    VideoTypeClassify = VodeoTypeJudge()
    for imagepath in glob('demo/*.jpg'):
        image = Image.open(imagepath)
        VideoTypeClassify.GetFrame(image)
        print(VideoTypeClassify.frame_res)
    classify_res = VideoTypeClassify.GetResult()
    print('video classify result is:', classify_res)


