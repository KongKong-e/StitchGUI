# PVStitch

光伏场景图像拼接桌面应用，支持多种特征检测器和匹配器的对比实验。

## 功能

- 基于 OpenCV `cv::Stitcher` 的全景图像拼接（Panorama / Scans 模式）
- 4 种特征检测器：SuperPoint、SIFT、ORB、SURF
- 2 种特征匹配器：LightGlue（深度学习，支持 GPU 加速）、BFMatcher（经典方法）
- 支持图像分割模式和匹配结果可视化
- Qt 6 图形界面，异步拼接不阻塞 UI
- 结果预览：滚轮缩放、拖拽平移、旋转查看
- 拼接状态指示灯（红绿灯）+ 实时耗时显示
- 智能输出命名：文件夹/文件名自动编码算法参数
- 拼接报告：JSON 格式自动生成，含算法参数和耗时
- 中英文界面切换

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
