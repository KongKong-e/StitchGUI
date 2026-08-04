# PVStitch

光伏场景图像拼接桌面应用，基于 Qt 6 + OpenCV + ONNX Runtime。

## 技术栈

| 依赖 | 版本 | 说明 |
|------|------|------|
| Qt | 6.5.3 | core, gui, widgets |
| OpenCV | 4.10.0 | `opencv_world4100.lib` |
| ONNX Runtime | - | C++11 API |
| OpenMP | MSVC 内置 | ParallelSURF 并行计算 |
| 编译器 | MSVC 2022 64bit | C++14 |
| 构建系统 | qmake | `.pro` 文件 |

第三方库 SDK 路径：
- OpenCV: `D:/code/DATA/3rdParty/opencv-sdk`
- ONNX Runtime: `D:/code/DATA/3rdParty/onnx-sdk`
- 模型文件: `model/superpoint.onnx`, `model/superpoint_lightglue.onnx`

## 构建

1. Qt Creator 打开 `StitchGUI.pro`
2. 构建 → 运行 qmake
3. 构建 → 构建项目

构建时 DLL 和模型目录自动复制到输出目录（.pro 中 post-link 步骤）。

## 项目结构

```
├── src/
│   ├── app/                    # 应用入口 + 主窗口
│   │   ├── main.cpp            # 入口
│   │   ├── mainwindow.cpp/h    # 主窗口 GUI
│   │   └── mainwindow.ui       # Qt Designer UI
│   ├── core/                   # 特征检测/匹配核心算法
│   │   ├── superpoint.cpp/h    # SuperPoint 特征检测器 (ONNX)
│   │   ├── lightglue.cpp/h     # LightGlue 特征匹配器 (ONNX)
│   │   ├── classical_matcher.cpp/h  # 经典匹配器 (BFMatcher)
│   │   ├── surf_wrapper.cpp/h  # SurfDetector: cv::Feature2D 包装层
│   │   ├── Surf.cpp/h          # cvg::Surf 实现 (检测+描述子)
│   │   └── dlldefine.h         # 导出宏定义
│   ├── utils/                  # 工具函数
│   │   ├── utils.cpp/h         # 图像读取、分割等工具函数
│   │   └── file_utils.cpp/h    # 文件/目录操作
│   └── parallelsurf/           # ParallelSURF 库 (独立实现)
│       ├── Image.h             # 积分图像
│       ├── KeyPoint.h          # 关键点数据结构
│       ├── BoxFilter.h         # Hessian 盒式滤波器 (inline)
│       ├── WaveFilter.h        # Haar 小波滤波器 (inline)
│       ├── MathStuff.h         # 数学工具 (LUT, 线性方程求解)
│       ├── KeyPointDetector.h  # 关键点检测器 (header only)
│       ├── KeyPointDescriptor.h # 描述子计算 (header only)
│       └── KeyPointDescriptorContext.h # 描述子上下文
├── resources/
│   ├── resources.qrc           # Qt 资源文件
│   └── images/                 # 图片资源
└── model/                      # ONNX 模型 (gitignore)
```

**注意**: `src/parallelsurf/*.cpp` 文件不参与编译。所有 ParallelSURF 实现通过 `src/core/Surf.cpp` 统一编译（Image 实现内联在 Surf.cpp 中），避免重复符号。

## 核心架构

### 特征检测器（4 种，通过 `cv::Feature2D` 接口统一）

| 检测器 | 类 | 输出描述子 | 配套匹配器 |
|--------|-----|-----------|-----------|
| SuperPoint | `SuperPoint` | CV_32F, 256维 | LightGlue / ClassicalMatcher |
| SIFT | `cv::SIFT` | CV_32F | ClassicalMatcher |
| ORB | `cv::ORB` (1000特征点) | CV_8U | ClassicalMatcher |
| SURF | `SurfDetector` | CV_32F | ClassicalMatcher |

### 特征匹配器（2 种，通过 `cv::detail::FeaturesMatcher` 接口统一）

| 匹配器 | 类 | 说明 |
|--------|-----|------|
| LightGlue | `LightGlue` | 深度学习，仅配合 SuperPoint，支持 GPU 加速（CUDA） |
| BFMatcher | `ClassicalMatcher` | CV_8U→Hamming, CV_32F→L2, Lowe's ratio test |

### 线程模型

拼接在 `QThread` 中执行（`StitchingWorker`），通过信号/槽与 UI 通信：
- `progressChanged(int)` — 进度更新
- `logMessage(QString)` — 日志输出
- `errorOccurred(QString)` — 错误提示
- `finished()` — 完成回调

### 拼接流程

```
读取图像 → 创建检测器 → 创建匹配器
→ cv::Stitcher::create() → setFeaturesFinder/setFeaturesMatcher
→ stitcher->stitch(imgs, pano) → 保存结果 → 生成报告
```

### 输出命名规则

输出文件夹和文件名按 `{检测器}_{匹配器}_{模式}_mt{匹配阈值}_ct{置信度}` 格式自动生成，例如 `sp_lg_pano_mt0.6_ct0.2`。缩写对照：

| 缩写 | 含义 | 缩写 | 含义 |
|------|------|------|------|
| sp | SuperPoint | lg | LightGlue |
| sift | SIFT | bf | BFMatcher |
| orb | ORB | pano | Panorama |
| surf | SURF | scan | Scans |

输出目录结构：
```
sp_lg_pano_mt0.3_ct0.5/
  sp_lg_pano_mt0.3_ct0.5.jpg   # 拼接结果
  sp_lg_pano_mt0.3_ct0.5.json  # 拼接报告 (JSON)
  match_0_1.jpg                 # 匹配中间图 (可选)
```

### 状态指示与报告

- 工具栏左侧有红绿灯状态指示（绿=就绪/完成，黄=拼接中，红=失败）
- 工具栏右侧显示拼接耗时（秒，小数点后两位）
- 拼接成功时自动在输出文件夹生成 JSON 格式报告（含算法参数、耗时、输入输出信息，中文键名）


