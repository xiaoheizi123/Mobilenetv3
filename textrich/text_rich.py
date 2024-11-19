import cv2
import numpy as np
import time
from PIL import Image
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
if __dir__ not in sys.path:
    sys.path.append(__dir__)
from ppocr.predict_det import TextDetector
from ppocr.predict_rec import TextRecognizer
import ppocr.utility as utility
from ppocr.utils.utils import get_crops


class OCRSystem:
    def __init__(self,
                 skip_frames=150,
                 text_num_thresh=30,
                 char_num_thresh=200,
                 text_frame_ratio=0.9,
                 use_gpu=False
                 ):
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = './textrich/ckpts/ch_PP-OCRv4_det.onnx'
        args.det_db_thresh = 0.3
        args.det_db_box_thresh = 0.6
        args.det_limit_side_len = 640

        args.rec_algorithm = 'SVTR_LCNet'
        args.rec_model_dir = './textrich/ckpts/ch_PP-OCRv4_rec.onnx'
        args.rec_char_dict_path = './textrich/ppocr/ppocr_keys_v1.txt'
        args.rec_image_shape = '3, 48, 320'
        args.use_space_char = True
        args.drop_score = 0.5
        args.use_onnx = True
        args.use_gpu = use_gpu
        self.skip_frames = skip_frames  # 仅对video输入有效
        self.text_num_thresh = text_num_thresh # 文本条数
        self.char_num_thresh = char_num_thresh # 文本字符数
        self.text_frame_ratio = text_frame_ratio # 仅对video输入有效
        self.text_rich_frame_num = 0 # 仅对frame输入有效
        self.processed_frame_num = 0 # 仅对frame输入有效
        self.use_rec = True
        self.warmup = False  # 模型预热

        self.text_detector = TextDetector(args)
        if self.use_rec:
            self.text_recognizer = TextRecognizer(args)

        if self.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for _ in range(2):
                self.text_detector(img)
            if self.use_rec:
                img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
                for _ in range(2):
                    self.text_recognizer([img] * 2)

    def ocr(self, frame, ):
        dt_boxes, elapse1 = self.text_detector(frame)
        out_res, filter_dt_boxes = [], []

        if len(dt_boxes) > 0:
            crop_list = get_crops(dt_boxes, frame)
            rec_res, elapse2 = self.text_recognizer(crop_list)
            for box, rec_result in zip(dt_boxes, rec_res):
                text, conf = rec_result
                if conf >= 0.8:
                    out_res.append({'points': box.astype(int).tolist(), 'text': text})
                    filter_dt_boxes.append(box)

        char_num = len(''.join([item['text'] for item in out_res]))
        print(f'文本数量 {len(out_res)} 字符数量 {char_num}',)
        # out_frame = utility.draw_text_det_res(filter_dt_boxes, frame)
        # cv2.namedWindow('res', cv2.WINDOW_NORMAL), cv2.imshow('res', out_frame), cv2.waitKey(0)
        return out_res, char_num

    def text_count(self, frame):
        # 桌面视频 是否为办公场景 | 按文本数量算
        if self.use_rec:
            out_res, char_num = self.ocr(frame)
            if char_num > self.char_num_thresh:
                return True
            else:
                return False
        else:
            out_res, _ = self.text_detector(frame)
            print('文本数量', len(out_res))
            # out_frame = utility.draw_text_det_res(out_res, frame)
            # cv2.namedWindow('res', cv2.WINDOW_NORMAL), cv2.imshow('res', out_frame), cv2.waitKey(0)

            if len(out_res) > self.text_num_thresh:
                return True
            else:
                return False
    def reset(self):
        self.text_rich_frame_num = 0
        self.processed_frame_num = 0

    def frame_api(self, frame):
        st = time.time()
        cur_flag = self.text_count(frame)
        if cur_flag:
            self.text_rich_frame_num += 1
        self.processed_frame_num += 1
        print(f"ocr_time {round(time.time() - st, 2)}s ")
        return cur_flag

    def accumulation(self):
        text_rich_scene = self.text_rich_frame_num / self.processed_frame_num > self.text_frame_ratio
        if self.processed_frame_num < 15:
            print('[warning] 累积页面较少 文字场景判别可能不准确')
        return text_rich_scene

    def video_api(self, video_path):
        total_time = 0
        frame_idx = 0
        text_rich_frame_num = 0
        processed_frame_num = 0
        videoCap, fps, total_frames = read_video(video_path)
        # total_assert_num = total_frames // self.skip_frames
        cur_skip_frames = int(fps/30*self.skip_frames) # 每5秒抽1帧 根据fps调整
        print('cur_skip_frames: ', cur_skip_frames)

        while True:
            st = time.time()
            _, frame = videoCap.read()
            if frame is None:
                break
            frame_idx += 1
            # 跳帧
            if cur_skip_frames > 0 and frame_idx % cur_skip_frames == 0:
                cur_flag = self.text_count(frame)
                if cur_flag:
                    text_rich_frame_num += 1
                processed_frame_num += 1
            total_time += time.time() - st

            # early stop
            if processed_frame_num>60 or (processed_frame_num>36 and text_rich_frame_num/processed_frame_num > 0.95):
                break
        print(f"Total_time {round(total_time, 2)}s\nMean_time {round(total_time / processed_frame_num * 1000, 2)}ms")

        text_rich_scene = text_rich_frame_num / processed_frame_num > self.text_frame_ratio
        print(f'{text_rich_frame_num / processed_frame_num :.3f}')
        return text_rich_scene


def read_video(video_path):
    videoCap = cv2.VideoCapture(video_path)
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    total_frames = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = (int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print(f"fps: {int(fps)} image_size: {image_size} total_frames:{total_frames}")

    return videoCap, fps, total_frames


if __name__ == '__main__':
    processor = OCRSystem()
    file_path = '../视频分类数据样例/游戏视频/1.mp4'
    result = processor(file_path)
    print(result)

