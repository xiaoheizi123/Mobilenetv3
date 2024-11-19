# TextRich

Demo级: 依据文本数据量 判断是否为word等文字编辑场景; 依赖前一步场景分类结果（游戏、电影、运动）

镜像地址：http://10.252.97.209:80/aiteam/paddleocr:v5


## 模型大小与推理速度 - 静态图模型
|           | 模型大小/MB | GPU/s | CPU/s |         官方速度         |       说明       |
|:---------:|:-------:|:-----:|:-----:|:--------------------:|:--------------:|
| pp_v4_det |   4.7   | 0.042 | 0.276 |          -           |  100张 640*640  |
| pp_v4_rec |  10.8   | 0.009 | 0.058 | 0.0098（cpu openvino) |  4222张 48*320  |

* GPU RTX2060  
* CPU Intel® Core™ i7-10700 CPU @ 2.90GHz × 16 


## 调用
```python
from text_rich import OCRSystem
processor = OCRSystem()  # 类初始化 策略参数均采用默认值

# 视频接口
file_path = '../视频分类数据样例/游戏视频/1.mp4' # 视频地址
result = processor.video_api(file_path)  # 调用接口  输出True/False=是否为文字丰富场景
print(result)

# 单帧图像接口
frame_lists = []  # dummy input
processor.reset() # 重置参数
for frame in frame_lists:
    processor.frame_api(frame)  # 单帧结果
result = processor.accumulation()  # 多帧积累结果 输出True/False=是否为文字丰富场景
print(result)
```