# SuperStitchGUI

光伏场景图像拼接桌面应用，支持多种特征检测器和匹配器的对比实验。

## 功能

- 基于 OpenCV `cv::Stitcher` 的全景图像拼接
- 4 种特征检测器：SuperPoint、SIFT、ORB、SURF
- 2 种特征匹配器：LightGlue（深度学习）、BFMatcher（经典方法）
- 支持图像分割模式和匹配结果可视化
- Qt 6 图形界面，异步拼接不阻塞 UI

## 依赖

- Qt 6.5.3
- OpenCV 4.10
- ONNX Runtime
- MSVC 2022 64bit

## 构建

1. Qt Creator 打开 `StitchGUI.pro`
2. 运行 qmake
3. 构建项目

模型文件需放到 `model/` 目录（构建时自动复制到输出目录）：
- `model/superpoint.onnx`
- `model/superpoint_lightglue.onnx`

## 详细说明

见 [CLAUDE.md](CLAUDE.md)。
