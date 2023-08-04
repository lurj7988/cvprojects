"""
视频文件每帧画面转为蒙版图片存储
"""

# 导入相关包
# from time import perf_counter
import cv2
import numpy as np

# 导入pixellib包
# import pixellib

# 实例分割
from pixellib.instance import instance_segmentation

# 导入弹幕管理模块
from danmu import Danmu_layer

import os

from PIL import Image


class VideoProcess:

    def __init__(self, videoFile, mask_path,
                 out_video_name, text_path):
        """
        构造方法

        @param videoFile mp4格式图片

        """
        self.videoFile = videoFile
        self.mask_path = mask_path
        self.out_video_name = out_video_name
        self.text_path = text_path

    def video2masks(self):
        """
        视频文件转为MASK图片

        """
        # 实例分割实例化
        instance = instance_segmentation()
        # 加载模型
        instance.load_model('./weights/mask_rcnn_coco.h5')

        # 筛选类别
        target_classes = instance.select_target_classes(person=True)

        # 读取视频
        cap = cv2.VideoCapture(self.videoFile)

        # 记录帧数
        frame_index = 0
        while True:
            # 读取
            ret, frame = cap.read()

            # 判断是否视频是否处理完
            if not ret:
                print('视频处理完毕')
                break

            self.frame2mask(frame, instance, target_classes, frame_index)

            frame_index += 1

            # # 显示渲染图
            # cv2.imshow('Demo',output)

            # # 退出条件
            # if cv2.waitKey(10) & 0xff == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()

    def frame2mask(self, frame, instance: instance_segmentation,
                   target_classes: dict, frame_index):
        # 实例分割
        results, output = instance.segmentFrame(
            frame, segment_target_classes=target_classes)

        # 人数
        person_count = len(results['class_ids'])

        if person_count > 0:
            # 遮罩
            masks = results['masks']

            # 创建黑色底图
            black_bg = np.zeros((frame.shape[:2]))

            for p_index in range(person_count):
                # print(masks[:, :, p_index])
                black_bg = np.where(masks[:, :, p_index], 255, black_bg)

            # 文件名
            mask_file = self.mask_path + str(frame_index) + '.jpg'
            cv2.imwrite(mask_file, black_bg)

            print('第%d帧处理完毕' % (frame_index))
        else:
            print('第%d帧无人' % (frame_index))

    def frame2mask2(self, frame, instance: instance_segmentation,
                    target_classes: dict):
        # 实例分割
        results, output = instance.segmentFrame(
            frame, segment_target_classes=target_classes)
        # 人数
        person_count = len(results['class_ids'])

        if person_count > 0:
            # 遮罩
            masks = results['masks']

            # 创建黑色底图
            black_bg = np.zeros((frame.shape[:2]))

            for p_index in range(person_count):
                # print(masks[:, :, p_index])
                black_bg = np.where(masks[:, :, p_index], 255, black_bg)
            return black_bg
        else:
            return None

    def save_mask(self, frame, instance: instance_segmentation,
                  target_classes: dict, frame_index):
        black_bg = self.frame2mask2(
            frame, instance, target_classes)
        if black_bg is not None:
            # 文件名
            mask_file = self.mask_path + str(frame_index) + '.jpg'
            cv2.imwrite(mask_file, black_bg)
            print('第%d帧处理完毕' % (frame_index))
        else:
            print('第%d帧无人' % (frame_index))

    def video_composite(self):
        """
        合成视频与弹幕
        1.读取视频第X帧画面
        2.获取第X帧弹幕层画面，并用蒙版处理
        3.合成弹幕成与视频层
        4.保存为视频文件
        """

        # 读取视频
        cap = cv2.VideoCapture(self.videoFile)

        # 获取视频宽度与高度
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧率
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 弹幕层实例化
        # text_path = 'danmu_real.txt'
        danmu_layer = Danmu_layer(self.text_path, frame_w, frame_h)

        # 记录帧数
        frame_index = 0

        # 构建视频写入器
        # video_name = './out_video/output2.mp4'
        video_writer = cv2.VideoWriter(
            self.out_video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps,
            (frame_w, frame_h))
        while True:
            # 读取
            ret, frame = cap.read()

            # 判断是否视频是否处理完
            if not ret:
                print('视频处理完毕')
                break
            # 获取弹幕层
            frame_danmu_layer = danmu_layer.generate_frame(frame_index)

            # 对弹幕层进行蒙版操作
            # 蒙版文件
            mask_file = self.mask_path + str(frame_index) + '.jpg'

            # 判断蒙版文件是否存在
            if os.path.exists(mask_file):

                mask_img = cv2.imread(mask_file)
                # print(mask_img.shape)
                # 转为灰度图
                mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                # print(mask_gray.shape)

                # 对弹幕层转为numpy数组
                frame_danmu_layer_np = np.array(frame_danmu_layer)

                # 对弹幕层alpha通道进行处理
                frame_danmu_layer_np[:, :, 3] = np.where(
                    mask_gray == 255, 0, frame_danmu_layer_np[:, :, 3])

                # 转为image
                frame_danmu_layer = Image.fromarray(frame_danmu_layer_np)
                # frame_danmu_layer.save('./out_video/'+str(frame_index)+'.png')

            # 合成弹幕层与视频层
            # 转为RGBA的Image格式
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_rgba_pil = Image.fromarray(frame_rgba)

            output = Image.alpha_composite(frame_rgba_pil, frame_danmu_layer)

            # 转为Numpy数组
            output_np = np.asarray(output)

            # 转为GBR
            output = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGR)

            video_writer.write(output)
            # 显示渲染图
            # cv2.imshow('Demo', output)

            frame_index += 1

            # 退出条件
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    def test(self):
        mask_img = cv2.imread('./masks_img/0.jpg')
        # 转为灰度图
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        # 弹幕层实例化
        # text_path = 'danmu_real.txt'
        danmu_layer = Danmu_layer(
            self.text_path, mask_gray.shape[1], mask_gray.shape[0])
        # 获取弹幕层
        frame_danmu_layer = danmu_layer.generate_frame(1)
        # 对弹幕层转为numpy数组
        frame_danmu_layer_np = np.array(frame_danmu_layer)
        # 对弹幕层alpha通道进行处理
        # frame_danmu_layer_np为(720, 1280, 4)形状
        # frame_danmu_layer_np[:,:,3]表示RGBA的A通道
        # np.where(mask_gray==255,0,frame_danmu_layer_np[:,:,3])表示如果mask_gray==255则frame_danmu_layer_np[:,:,3]为0，否则frame_danmu_layer_np[:,:,3]不变
        frame_danmu_layer_np[:, :, 3] = np.where(
            mask_gray == 255, 0, frame_danmu_layer_np[:, :, 3])
        # 转为image
        frame_danmu_layer = Image.fromarray(frame_danmu_layer_np)
        frame_danmu_layer.show()

    def video_composite2(self):
        """
        合成视频与弹幕
        1.读取视频第X帧画面
        2.获取第X帧弹幕层画面，并用蒙版处理
        3.合成弹幕成与视频层
        4.保存为视频文件
        """

        # 读取视频
        cap = cv2.VideoCapture(self.videoFile)

        # 获取视频宽度与高度
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧率
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 弹幕层实例化
        # text_path = 'danmu_real.txt'
        danmu_layer = Danmu_layer(self.text_path, frame_w, frame_h)

        # 记录帧数
        frame_index = 0

        # 构建视频写入器
        # video_name = './out_video/output2.mp4'
        video_writer = cv2.VideoWriter(
            self.out_video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps,
            (frame_w, frame_h))
        # 实例分割实例化
        instance = instance_segmentation()
        # 加载模型
        instance.load_model('./weights/mask_rcnn_coco.h5')

        # 筛选类别
        target_classes = instance.select_target_classes(person=True)
        while True:
            # 读取
            ret, frame = cap.read()
            # 判断是否视频是否处理完
            if not ret:
                print('视频处理完毕')
                break
            # 获取弹幕层
            frame_danmu_layer = danmu_layer.generate_frame(frame_index)
            # 这边的frame copy一份防止对原图进行修改
            black_bg = self.frame2mask2(frame.copy(), instance, target_classes)
            if black_bg is not None:
                # 转为灰度图
                # mask_gray = cv2.cvtColor(black_bg, cv2.COLOR_BGR2GRAY)
                # print(black_bg.shape)
                # 对弹幕层转为numpy数组
                frame_danmu_layer_np = np.array(frame_danmu_layer)
                # 对弹幕层alpha通道进行处理
                frame_danmu_layer_np[:, :, 3] = np.where(
                    black_bg == 255, 0, frame_danmu_layer_np[:, :, 3])
                # 转为image
                frame_danmu_layer = Image.fromarray(frame_danmu_layer_np)
                # frame_danmu_layer.save('./out_video/'+str(frame_index)+'.png')

            # 合成弹幕层与视频层
            # 转为RGBA的Image格式
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_rgba_pil = Image.fromarray(frame_rgba)

            output = Image.alpha_composite(frame_rgba_pil, frame_danmu_layer)

            # 转为Numpy数组
            output_np = np.asarray(output)

            # 转为GBR
            output = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGR)

            video_writer.write(output)
            # 显示渲染图
            # cv2.imshow('Demo', output)

            frame_index += 1
            # 退出条件
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 实例化
    vp = VideoProcess('./videos/siri.mp4', './masks_siri/',
                      './out_video/siri.mp4', 'danmu_real.txt')
    # vp.video_composite()
    # vp.video2masks()
    # vp.test()
    vp.video_composite2()
