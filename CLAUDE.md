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

## 常见注意事项

- **SURF detect→compute 缓存**：`cv::Stitcher` 先对所有图像调用 `detect()` 再调用 `compute()`，`SurfDetector` 通过图像哈希缓存灰度图和关键点，确保 compute 使用与 detect 相同的坐标空间
- **SURF 描述子是 CV_32F 浮点型**，ClassicalMatcher 使用 BFMatcher + L2 距离匹配（FLANN KD-tree 在部分 OpenCV 构建中会导致堆损坏）
- **surf_wrapper.cpp 不编译 parallelsurf/*.cpp**，所有实现合并到 Surf.cpp
- **ParallelSURF 的 Image 对象不能浅拷贝**，Surf 类通过参数传递 Image 引用而非存储成员
- **模型文件在 .gitignore 中**，构建时通过 .pro 的 post-link 自动从 `model/` 复制到输出目录
- **模型路径使用 `QCoreApplication::applicationDirPath()`**，自动适配构建/部署目录
- **LightGlue ONNX 推理线程数**：`SetIntraOpNumThreads(4)`，sessionOptions 在 if 块内创建（非 static）
- **LightGlue GPU 加速**：选择 LightGlue 时 UI 出现 GPU checkbox。`.pro` 用 `exists()` 自动检测 CUDA SDK，定义 `ONNX_CUDA_AVAILABLE` 宏。GPU session 创建失败自动 fallback 到 CPU。需要对方机器装 CUDA Toolkit 12.x，cuDNN 可随程序打包（`cudnn/` 目录）
- **LightGlue 匹配数保护**：Panorama 路径在 `findHomography` 前检查匹配数 < 4，避免空矩阵崩溃
- **ORB 特征点数**：`cv::ORB::create(1000)`，默认 500 改为 1000 提高匹配成功率
- **中英文切换**：帮助菜单中可切换语言，`retranslateUi()` 刷新所有 UI 控件，Worker 日志也跟随语言设置
- **帮助菜单**：操作指南、报错说明、参数说明三个对话框，均支持中英文
- **输出命名**：改变算法参数时自动刷新，但点击"开始拼接"时不再覆盖用户手动修改的输出路径
- **ONNX Session 内存管理**：LightGlue 和 SuperPoint 的 ONNX session 使用 `shared_ptr` 缓存在函数内 static map 中，避免内存泄漏。不要用文件级 static `Ort::Env`（会导致启动崩溃），必须用函数内 static
- **cv::Mat 跨线程安全**：`StitchingWorker` 的 `resultReady` 信号不传 `cv::Mat` 参数，结果存储在 Worker 的 `m_lastResult` 成员变量中，主线程通过 `lastResult()` getter 读取。避免 queued connection 中 cv::Mat 拷贝导致的堆损坏
- **Worker 生命周期**：不在 `finished` 信号中 `deleteLater`，延迟到下一次拼接开始时清理，确保 queued 信号在 worker 销毁前被处理
- **置信度钳位已删除**：`confidence > 3 ? 0 : confidence` 这行已移除（LightGlue 和 ClassicalMatcher 中）。该逻辑在匹配质量极好时会错误地将置信度归零
- **matchThreshold 默认值**：0.6（mainwindow.ui）
- **Qt 消息处理器**：`pvMessageHandler` 将 `qDebug()/qWarning()` 重定向到日志窗口。线程安全设计：所有线程的 qDebug 输出写入 static QMutex 保护的 `QStringList` 缓冲队列，GUI 线程 QTimer（200ms 间隔）调用 `flushPendingLogs()` 批量刷新到 `textEdit_log`。**不写 stderr**——避免无缓冲 I/O 拖慢拼接性能
- **匹配诊断日志**：LightGlue 和 ClassicalMatcher 在 ratio test 后、Homography 前、置信度计算后通过 `qDebug()` 输出匹配数/内点数/置信度
- **匹配器 clearCache()**：**必须在匹配图保存之后调用**（`stitch()` 返回后立刻 clearCache 会导致匹配图无法保存，因为 features/matchinfo 已被清空）。清除时机：`stitch()` 成功 → 保存结果图 → 如需保存匹配图则先读 matcher 数据并生成 match_X_Y.jpg → 最后 clearCache
- **Homography 坐标系**：全景模式使用**原始图像坐标**（不中心化），`cv::Stitcher` 内部的 focal length 分解和 Bundle Adjustment 期望此坐标系。关键点坐标直接传入 `findHomography()`，不减去半宽高
- **控制台编码**：`main.cpp` 中 `SetConsoleOutputCP(65001)` 设置控制台代码页为 UTF-8，解决中文输出乱码
- **调试输出优化**：`readImages()` 已移除逐文件 `std::cout` 调试输出；`lightglue.cpp` 已移除每对匹配的 Homography 矩阵 `std::cout`。错误路径保留（"Can't read image" 等）
- **CUDA/cuDNN DLL 自动复制**：`.pro` 从 `$(CUDA_PATH)/bin/` 复制 `cudart64_*.dll`、`cublas64_*.dll`、`cublasLt64_*.dll`；自动检测 cuDNN 安装路径复制 `cudnn*.dll`

## 待解决问题

-批处理脚本 
