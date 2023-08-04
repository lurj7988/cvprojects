# 防挡弹幕

## 介绍

本脚本用于防止弹幕遮挡字幕，适用于弹幕密集的视频。

## 环境

- Python 3.8
- TensorFlow 2.2.0
- imgaug
- numpy 1.18.0
- opencv-python
- pixellib

```bash
conda create -n anti-barrage python=3.8
conda activate anti-barrage
pip install -r requirements.txt
```

## GPU加速

TensorFlow 2.2.0版本支持CUDA 10.1，可以使用GPU加速。

```bash
pip install tensorflow-gpu==2.2.0
```
