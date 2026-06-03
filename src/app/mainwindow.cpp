#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QDateTime>
#include <QPixmap>
#include <QImage>
#include <QImageReader>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QFont>
#include <QAction>
#include <QFile>
#include <QTextStream>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTextBrowser>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QStandardItemModel>
#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>
#include <cstdio>

#include <QMutex>

// 线程安全的 qDebug 消息缓冲队列 + 保护锁
static QMutex s_logMutex;
static QStringList s_pendingLogs;

// Qt 消息处理器：将 qDebug/qWarning 重定向到日志窗口
// 注意：此函数可以被任意线程调用（Worker、OpenMP 等），必须线程安全
// 不写 stderr/stdout 以避免无缓冲 I/O 影响拼接性能
static void pvMessageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    // 附加到缓冲队列（线程安全：QMutex 保护）
    QString ts = QDateTime::currentDateTime().toString("hh:mm:ss");
    QString line = QString("[%1] %2").arg(ts, msg);
    {
        QMutexLocker locker(&s_logMutex);
        s_pendingLogs.append(line);
    }
}

// MainWindow 定时器回调：将缓冲的日志刷新到 UI（只能在 GUI 线程调用）
void MainWindow::flushPendingLogs()
{
    QMutexLocker locker(&s_logMutex);
    if (s_pendingLogs.isEmpty())
        return;
    for (const QString& line : s_pendingLogs) {
        ui->textEdit_log->appendPlainText(line);
    }
    s_pendingLogs.clear();
}

StitchingWorker::StitchingWorker(QObject* parent)
    : QObject(parent)
    , m_divideImages(false)
    , m_mode(cv::Stitcher::PANORAMA)
    , m_waveCorrectKind(cv::detail::WAVE_CORRECT_AUTO)
    , m_matchThreshold(0.2f)
    , m_confidenceThreshold(0.1f)
    , m_saveMatching(false)
    , m_useGpu(false)
    , m_isEnglish(false)
    , m_stopRequested(false)
{
}

void StitchingWorker::setParameters(
    const std::string& imageDir,
    const std::string& extension,
    bool divideImages,
    const std::wstring& superPointPath,
    const std::wstring& lightGluePath,
    cv::Stitcher::Mode mode,
    int waveCorrectKind,
    float matchThreshold,
    float confidenceThreshold,
    const std::string& outputDir,
    const std::string& outputName,
    bool saveMatching,
    const std::string& detector,
    const std::string& matcher,
    bool useGpu,
    bool isEnglish)
{
    QMutexLocker locker(&m_mutex);
    m_imageDir = imageDir;
    m_extension = extension;
    m_divideImages = divideImages;
    m_superPointPath = superPointPath;
    m_lightGluePath = lightGluePath;
    m_mode = mode;
    m_waveCorrectKind = waveCorrectKind;
    m_matchThreshold = matchThreshold;
    m_confidenceThreshold = confidenceThreshold;
    m_outputDir = outputDir;
    m_outputName = outputName;
    m_saveMatching = saveMatching;
    m_detector = detector;
    m_matcher = matcher;
    m_useGpu = useGpu;
    m_isEnglish = isEnglish;
    m_stopRequested = false;
}

void StitchingWorker::requestStop()
{
    QMutexLocker locker(&m_mutex);
    m_stopRequested = true;
}

cv::Ptr<cv::Feature2D> StitchingWorker::createDetector()
{
    if (m_detector == "superpoint")
        return cv::makePtr<SuperPoint>(m_superPointPath);
    else if (m_detector == "sift")
        return cv::SIFT::create();
    else if (m_detector == "orb")
        return cv::ORB::create(1000);
    else if (m_detector == "surf")
        return cv::makePtr<SurfDetector>();
    throw std::runtime_error(m_isEnglish ? ("Unsupported detector: " + m_detector) : ("不支持的检测器: " + m_detector));
}

cv::Ptr<cv::detail::FeaturesMatcher> StitchingWorker::createMatcher()
{
    if (m_matcher == "lightglue")
        return cv::makePtr<LightGlue>(m_lightGluePath, m_mode, m_matchThreshold, m_useGpu);
    else
        return cv::makePtr<ClassicalMatcher>(m_mode, m_matchThreshold);
}

void StitchingWorker::process()
{
    try {
        emit logMessage(m_isEnglish ? "Starting..." : "开始...");

        emit progressChanged(10);
        emit logMessage(m_isEnglish ? "Reading images..." : "正在读取图像...");

        std::vector<cv::Mat> imgs = readImages(m_imageDir, m_extension, m_divideImages);

        if (imgs.empty()) {
            emit errorOccurred(m_isEnglish ? "No image files found!" : "未找到任何图像文件！");
            emit finished();
            return;
        }

        emit logMessage(m_isEnglish
            ? QString("Successfully read %1 images").arg(imgs.size())
            : QString("成功读取 %1 张图像").arg(imgs.size()));
        emit progressChanged(30);

        {
            QMutexLocker locker(&m_mutex);
            if (m_stopRequested) {
                emit logMessage(m_isEnglish ? "Stitching stopped" : "拼接已停止");
                emit finished();
                return;
            }
        }

        emit logMessage(m_isEnglish
            ? "Initializing feature detector: " + QString::fromStdString(m_detector) + "..."
            : "正在初始化特征检测器: " + QString::fromStdString(m_detector) + "...");
        cv::Ptr<cv::Feature2D> detector = createDetector();

        emit logMessage(m_isEnglish
            ? "Initializing matcher: " + QString::fromStdString(m_matcher) + "..."
            : "正在初始化匹配器: " + QString::fromStdString(m_matcher) + "...");
        cv::Ptr<cv::detail::FeaturesMatcher> matcher = createMatcher();

        emit progressChanged(50);

        {
            QMutexLocker locker(&m_mutex);
            if (m_stopRequested) {
                emit logMessage(m_isEnglish ? "Stitching stopped" : "拼接已停止");
                emit finished();
                return;
            }
        }

        emit logMessage(m_isEnglish ? "Creating stitcher..." : "正在创建拼接器...");
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(m_mode);
        stitcher->setPanoConfidenceThresh(m_confidenceThreshold);
        stitcher->setFeaturesFinder(detector);
        stitcher->setFeaturesMatcher(matcher);
        stitcher->setWaveCorrectKind(static_cast<cv::detail::WaveCorrectKind>(m_waveCorrectKind));

        emit logMessage(m_isEnglish
            ? "Stitching in progress (matching → camera estimation → compositing)..."
            : "正在执行 (特征匹配→相机估计→合成融合)...");
        emit progressChanged(60);
        cv::Mat pano;
        QElapsedTimer stitchTimer;
        stitchTimer.start();
        cv::Stitcher::Status status = stitcher->stitch(imgs, pano);
        double stitchElapsed = stitchTimer.elapsed() / 1000.0;
        emit logMessage(m_isEnglish
            ? QString("Stitching pipeline finished in %1s (detection + matching + compositing)").arg(stitchElapsed, 0, 'f', 1)
            : QString("拼接流水线完成，耗时 %1s (检测+匹配+合成融合)").arg(stitchElapsed, 0, 'f', 1));

        if (m_matcher == "lightglue") {
            LightGlue* lg = dynamic_cast<LightGlue*>(matcher.get());
            if (lg) {
                emit logMessage(lg->isUsingGpu()
                    ? (m_isEnglish ? "LightGlue: Using GPU (CUDA) inference" : "LightGlue: 使用 GPU (CUDA) 推理")
                    : (m_isEnglish ? "LightGlue: Using CPU inference" : "LightGlue: 使用 CPU 推理"));
            }
        }

        emit progressChanged(80);

        if (status == cv::Stitcher::OK) {
            emit logMessage(m_isEnglish ? "Stitching succeeded!" : "拼接成功！");

            std::string resDir = m_outputDir;
            if (libpano::create_directory(resDir)) {
                emit logMessage(m_isEnglish
                    ? "Created output directory: " + QString::fromStdString(resDir)
                    : "创建输出目录: " + QString::fromStdString(resDir));
            }

            std::string outputPath = resDir + "/" + m_outputName;
            if (cv::imwrite(outputPath, pano)) {
                emit logMessage(m_isEnglish
                    ? "Saved stitching result: " + QString::fromStdString(outputPath)
                    : "保存拼接结果: " + QString::fromStdString(outputPath));
            } else {
                emit logMessage(m_isEnglish ? "Warning: Failed to save stitching result" : "警告: 保存拼接结果失败");
            }

            if (m_saveMatching) {
                emit logMessage(m_isEnglish ? "Saving matching results..." : "正在保存匹配结果...");

                std::vector<cv::detail::ImageFeatures> features;
                std::vector<cv::detail::MatchesInfo> matches;

                if (m_matcher == "lightglue") {
                    LightGlue* lg = dynamic_cast<LightGlue*>(matcher.get());
                    if (lg) {
                        features = lg->features();
                        matches = lg->matchinfo();
                    }
                } else {
                    ClassicalMatcher* cm = dynamic_cast<ClassicalMatcher*>(matcher.get());
                    if (cm) {
                        features = cm->features();
                        matches = cm->matchinfo();
                    }
                }

                for (size_t i = 0; i < matches.size(); i++) {
                    cv::Mat srcImg = imgs[matches[i].src_img_idx];
                    cv::Mat dstImg = imgs[matches[i].dst_img_idx];

                    cv::detail::ImageFeatures srcFeature;
                    cv::detail::ImageFeatures dstFeature;
                    for (size_t j = 0; j < features.size(); j++) {
                        if (features[j].img_idx == matches[i].src_img_idx) {
                            srcFeature = features[j];
                        }
                        if (features[j].img_idx == matches[i].dst_img_idx) {
                            dstFeature = features[j];
                        }
                    }

                    cv::Mat imgMatches;
                    cv::Mat SrcresizedImage;
                    cv::resize(srcImg, SrcresizedImage, srcFeature.img_size);

                    cv::Mat DstresizedImage;
                    cv::resize(dstImg, DstresizedImage, dstFeature.img_size);

                    cv::drawMatches(
                        SrcresizedImage, srcFeature.keypoints,
                        DstresizedImage, dstFeature.keypoints,
                        matches[i].matches, imgMatches);

                    std::string matchPath = resDir + "/match_" + std::to_string(matches[i].src_img_idx) +
                        std::string("_") + std::to_string(matches[i].dst_img_idx) + ".jpg";
                    cv::imwrite(matchPath, imgMatches);
                    emit logMessage(m_isEnglish
                        ? "Saved matching result: " + QString::fromStdString(matchPath)
                        : "保存匹配结果: " + QString::fromStdString(matchPath));
                }
            }

            // 匹配图保存完成后，释放匹配器缓存的内存
            {
                LightGlue* lg = dynamic_cast<LightGlue*>(matcher.get());
                ClassicalMatcher* cm = dynamic_cast<ClassicalMatcher*>(matcher.get());
                if (lg) lg->clearCache();
                if (cm) cm->clearCache();
            }

            emit progressChanged(100);
            m_lastResult = pano;
            emit resultReady();
        } else {
            QString errorMsg;
            switch (status) {
                case cv::Stitcher::ERR_NEED_MORE_IMGS:
                    errorMsg = m_isEnglish ? "Error: Need more images" : "错误: 需要更多图像";
                    break;
                case cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
                    errorMsg = m_isEnglish ? "Error: Homography estimation failed" : "错误: 单应性矩阵估计失败";
                    break;
                case cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
                    errorMsg = m_isEnglish ? "Error: Camera parameters adjustment failed" : "错误: 相机参数调整失败";
                    break;
                default:
                    errorMsg = m_isEnglish
                        ? "Error: Stitching failed (error code: " + QString::number(status) + ")"
                        : "错误: 拼接失败 (错误代码: " + QString::number(status) + ")";
                    break;
            }
            emit errorOccurred(errorMsg);
        }

    } catch (const std::exception& e) {
        emit errorOccurred(m_isEnglish ? QString("Exception: %1").arg(e.what()) : QString("异常: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred(m_isEnglish ? "Unknown error" : "未知错误");
    }

    emit finished();
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_workerThread(nullptr)
    , m_worker(nullptr)
{
    ui->setupUi(this);

    // 安装 Qt 消息处理器，将 qDebug/qWarning 输出到日志窗口
    qInstallMessageHandler(pvMessageHandler);

    setWindowTitle("PVStitch");

    // 主流桌面工具风格: 清晰分区、强调主操作、提升可读性
    setStyleSheet(
        "QMainWindow {"
        "  background-color: #f3f5f8;"
        "}"
        "QMenuBar {"
        "  background: #ffffff;"
        "  border-bottom: 1px solid #d9dfe8;"
        "  padding: 2px 8px;"
        "}"
        "QMenuBar::item {"
        "  spacing: 8px;"
        "  padding: 6px 10px;"
        "  border-radius: 6px;"
        "}"
        "QMenuBar::item:selected {"
        "  background: #e8eef9;"
        "}"
        "QStatusBar {"
        "  background: #ffffff;"
        "  border-top: 1px solid #d9dfe8;"
        "}"
        "QGroupBox {"
        "  background: #ffffff;"
        "  border: 1px solid #d7dde7;"
        "  border-radius: 10px;"
        "  margin-top: 8px;"
        "  padding: 10px 12px 8px 12px;"
        "  font-weight: 600;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 12px;"
        "  top: 2px;"
        "  padding: 0 6px;"
        "  color: #2a3343;"
        "  background: #ffffff;"
        "}"
        "QLabel {"
        "  color: #2e3645;"
        "}"
        "QLineEdit, QComboBox, QDoubleSpinBox {"
        "  background: #ffffff;"
        "  border: 1px solid #c8d0dc;"
        "  border-radius: 8px;"
        "  min-height: 26px;"
        "  padding: 4px 10px;"
        "  color: #1f2937;"
        "}"
        "QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {"
        "  border: 1px solid #2b78e4;"
        "}"
        "QCheckBox {"
        "  spacing: 8px;"
        "  color: #2e3645;"
        "}"
        "QPushButton {"
        "  background: #eef2f7;"
        "  border: 1px solid #ccd5e2;"
        "  border-radius: 8px;"
        "  min-height: 26px;"
        "  padding: 0 14px;"
        "  color: #2b3443;"
        "}"
        "QPushButton:hover {"
        "  background: #e4eaf3;"
        "}"
        "QPushButton:pressed {"
        "  background: #dbe3ef;"
        "}"
        "QPushButton#pushButton_startStitching {"
        "  background: #0f6cbd;"
        "  color: #ffffff;"
        "  border: 1px solid #0f6cbd;"
        "  font-weight: 600;"
        "}"
        "QPushButton#pushButton_startStitching:hover {"
        "  background: #115ea3;"
        "  border: 1px solid #115ea3;"
        "}"
        "QProgressBar {"
        "  background: #ecf1f7;"
        "  border: 1px solid #d0d8e3;"
        "  border-radius: 7px;"
        "  text-align: center;"
        "  color: #2b3443;"
        "  min-height: 18px;"
        "}"
        "QProgressBar::chunk {"
        "  border-radius: 6px;"
        "  background-color: #0f6cbd;"
        "}"
        "QPlainTextEdit#textEdit_log {"
        "  background: #0f1725;"
        "  color: #d7e3f4;"
        "  border: 1px solid #1b2740;"
        "  border-radius: 8px;"
        "  selection-background-color: #244f7a;"
        "}"
        "QGraphicsView {"
        "  background: #fbfcfe;"
        "  border: 1px solid #c7d2e0;"
        "  border-radius: 8px;"
        "}"
    );

    QFont appFont("Microsoft YaHei UI", 10);
    setFont(appFont);

    ui->horizontalLayout_main->setContentsMargins(14, 12, 14, 12);
    ui->horizontalLayout_main->setSpacing(12);
    ui->verticalLayout->setSpacing(3);
    ui->gridLayout->setVerticalSpacing(4);
    ui->gridLayout_3->setVerticalSpacing(4);
    ui->gridLayout_4->setVerticalSpacing(4);
    ui->horizontalLayout_main->setStretch(0, 2);
    ui->horizontalLayout_main->setStretch(1, 5);

    // 右侧上下比例：待拼接图片(1) : 结果预览(3)
    ui->verticalLayout_right->setStretch(0, 1);
    ui->verticalLayout_right->setStretch(1, 3);

    // 保持紧凑但不拥挤: 取消过严高度限制，避免控件挤压重叠
    ui->groupBox->setMaximumHeight(QWIDGETSIZE_MAX);
    ui->groupBox_3->setMaximumHeight(QWIDGETSIZE_MAX);
    ui->groupBox_4->setMaximumHeight(QWIDGETSIZE_MAX);

    ui->lineEdit_imageDir->setPlaceholderText("请选择待拼接图像文件夹");
    ui->lineEdit_outputDir->setPlaceholderText("请选择输出文件夹");
    ui->pushButton_startStitching->setText("开始拼接");
    ui->pushButton_reset->setText("重置任务");
    ui->textEdit_log->setMinimumHeight(190);
    ui->statusbar->showMessage("就绪");

    m_exportLogAction = new QAction("导出日志...", this);
    ui->menu->clear();
    ui->menu->addAction(ui->action_openImages);
    ui->menu->addAction(m_exportLogAction);
    ui->menu->addSeparator();
    ui->menu->addAction(ui->action_exit);
    connect(m_exportLogAction, &QAction::triggered, this, &MainWindow::on_exportLog_triggered);

    m_operationGuideAction = new QAction("操作指南", this);
    ui->menu_2->insertAction(ui->action_about, m_operationGuideAction);
    connect(m_operationGuideAction, &QAction::triggered, this, &MainWindow::on_operationGuide_triggered);

    m_errorGuideAction = new QAction("报错说明", this);
    ui->menu_2->insertAction(ui->action_about, m_errorGuideAction);
    connect(m_errorGuideAction, &QAction::triggered, this, &MainWindow::on_errorGuide_triggered);

    m_paramGuideAction = new QAction("参数说明", this);
    ui->menu_2->insertAction(ui->action_about, m_paramGuideAction);
    connect(m_paramGuideAction, &QAction::triggered, this, &MainWindow::on_paramGuide_triggered);

    m_langAction = new QAction("English", this);
    ui->menu_2->insertAction(ui->action_about, m_langAction);
    connect(m_langAction, &QAction::triggered, this, &MainWindow::on_switchLanguage_triggered);
    
    // 预设模型路径（相对于可执行文件目录）
    QString appDir = QCoreApplication::applicationDirPath();
    m_superPointPath = appDir + "/model/superpoint.onnx";
    m_lightGluePath = appDir + "/model/superpoint_lightglue.onnx";

    initializeConnections();

    // 定时器：定期将线程安全缓冲区中的 qDebug 消息刷新到日志窗口
    QTimer* logTimer = new QTimer(this);
    connect(logTimer, &QTimer::timeout, this, &MainWindow::flushPendingLogs);
    logTimer->start(200);  // 每 200ms 刷新一次

    // 设置菜单
    connect(ui->action_modelSettings, &QAction::triggered, this, &MainWindow::on_modelSettings_triggered);

    // 结果预览：QGraphicsScene + 缩放/旋转
    m_resultScene = new QGraphicsScene(this);
    m_resultPixmapItem = nullptr;
    m_rotationAngle = 0;
    m_zoomFactor = 1.0;
    m_viewCenter = QPointF(0, 0);
    ui->graphicsView_result->setScene(m_resultScene);
    ui->graphicsView_result->setRenderHint(QPainter::SmoothPixmapTransform);
    ui->graphicsView_result->viewport()->installEventFilter(this);

    // 状态指示灯 + 拼接耗时（与工具栏按钮同行）
    m_stitchState = StateReady;
    m_lastElapsedSec = -1;

    m_statusLabel = new QLabel(this);
    m_statusLabel->setFixedHeight(30);
    m_statusLabel->setMinimumWidth(110);
    m_statusLabel->setAlignment(Qt::AlignCenter);
    m_statusLabel->setText(m_isEnglish ? "● Ready" : "● 就绪");
    m_statusLabel->setStyleSheet(
        "QLabel { background-color: #eafaf1; border: 1px solid #27ae60; border-radius: 6px; "
        "padding: 2px 10px; font-size: 14px; font-weight: bold; color: #27ae60; }");

    m_timeLabel = new QLabel(this);
    m_timeLabel->setFixedHeight(30);
    m_timeLabel->setMinimumWidth(160);
    m_timeLabel->setAlignment(Qt::AlignCenter);
    m_timeLabel->setText("");
    m_timeLabel->setStyleSheet(
        "QLabel { background-color: #eef2f7; border: 1px solid #bdc3c7; border-radius: 6px; "
        "padding: 2px 10px; font-size: 14px; font-weight: bold; color: #2c3e50; }");

    // 工具栏按钮（居中）：放大、缩小、左旋、右旋
    QString btnStyle =
        "QPushButton { font-size: 18px; min-width: 32px; max-width: 32px; min-height: 30px; max-height: 30px; "
        "border: 1px solid #bdc3c7; border-radius: 4px; background-color: #f8f9fa; }"
        "QPushButton:hover { background-color: #e8ecef; border-color: #95a5a6; }"
        "QPushButton:pressed { background-color: #d5dbdb; }";

    QPushButton* zoomInBtn = new QPushButton("＋", this);
    QPushButton* zoomOutBtn = new QPushButton("－", this);
    QPushButton* rotateLeftBtn = new QPushButton("↺", this);
    QPushButton* rotateRightBtn = new QPushButton("↻", this);
    zoomInBtn->setStyleSheet(btnStyle);
    zoomOutBtn->setStyleSheet(btnStyle);
    rotateLeftBtn->setStyleSheet(btnStyle);
    rotateRightBtn->setStyleSheet(btnStyle);

    QHBoxLayout* toolbarLayout = new QHBoxLayout();
    toolbarLayout->addWidget(m_statusLabel);
    toolbarLayout->addStretch();
    toolbarLayout->addWidget(zoomInBtn);
    toolbarLayout->addSpacing(4);
    toolbarLayout->addWidget(zoomOutBtn);
    toolbarLayout->addSpacing(12);
    toolbarLayout->addWidget(rotateLeftBtn);
    toolbarLayout->addSpacing(4);
    toolbarLayout->addWidget(rotateRightBtn);
    toolbarLayout->addStretch();
    toolbarLayout->addWidget(m_timeLabel);
    ui->verticalLayout_2->insertLayout(0, toolbarLayout);
    connect(zoomInBtn, &QPushButton::clicked, this, &MainWindow::on_zoomIn_clicked);
    connect(zoomOutBtn, &QPushButton::clicked, this, &MainWindow::on_zoomOut_clicked);
    connect(rotateLeftBtn, &QPushButton::clicked, this, &MainWindow::on_rotateLeft_clicked);
    connect(rotateRightBtn, &QPushButton::clicked, this, &MainWindow::on_rotateRight_clicked);

    // 待拼接图片：选择文件夹后自动加载缩略图
    connect(ui->lineEdit_imageDir, &QLineEdit::textChanged, this, &MainWindow::loadInputImages);

    // 检测器/匹配器联动逻辑
    auto updateAlgoVisibility = [this]() {
        int detIdx = ui->comboBox_detector->currentIndex();
        bool isSuperPoint = (detIdx == 0);

        // 非 SuperPoint 时，LightGlue 不可用，自动切到 BFMatcher
        if (!isSuperPoint && ui->comboBox_matcher->currentIndex() == 0) {
            ui->comboBox_matcher->blockSignals(true);
            ui->comboBox_matcher->setCurrentIndex(1);
            ui->comboBox_matcher->blockSignals(false);
        }

        // 灰化 LightGlue 选项（非 SuperPoint 时不可选）
        QStandardItemModel* model = qobject_cast<QStandardItemModel*>(
            ui->comboBox_matcher->model());
        if (model) {
            model->item(0)->setEnabled(isSuperPoint);
        }

        // GPU checkbox 仅在选中 LightGlue 时可见
        bool isLightGlue = (ui->comboBox_matcher->currentIndex() == 0);
        ui->checkBox_gpu->setVisible(isLightGlue && isSuperPoint);
    };
    connect(ui->comboBox_detector, &QComboBox::currentIndexChanged, this, updateAlgoVisibility);
    connect(ui->comboBox_matcher, &QComboBox::currentIndexChanged, this, updateAlgoVisibility);
    updateAlgoVisibility();

    // 输出文件夹路径变化时，同步更新输出文件名
    connect(ui->lineEdit_outputDir, &QLineEdit::textChanged, this, [this](const QString& text) {
        if (!text.isEmpty()) {
            ui->lineEdit_outputName->setText(QFileInfo(text).fileName() + ".jpg");
        }
    });

    // 检测器/匹配器/模式/阈值变化时，自动刷新输出路径
    auto refreshOutputTag = [this]() {
        QString baseDir = ui->lineEdit_imageDir->text();
        if (baseDir.isEmpty()) return;
        QString tag = buildOutputTag();
        ui->lineEdit_outputDir->setText(baseDir + "/" + tag);
        ui->lineEdit_outputName->setText(tag + ".jpg");
    };
    connect(ui->comboBox_detector, QOverload<int>::of(&QComboBox::currentIndexChanged), this, refreshOutputTag);
    connect(ui->comboBox_matcher, QOverload<int>::of(&QComboBox::currentIndexChanged), this, refreshOutputTag);
    connect(ui->comboBox_mode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, refreshOutputTag);
    connect(ui->doubleSpinBox_matchThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, refreshOutputTag);
    connect(ui->doubleSpinBox_confidenceThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, refreshOutputTag);

    appendLog(m_isEnglish ? "PVStitch GUI started" : "SuperStitch GUI 已启动");
    retranslateUi();
}

MainWindow::~MainWindow()
{
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
    delete m_worker;
    delete m_workerThread;
    delete ui;
}

void MainWindow::initializeConnections()
{
    connect(ui->pushButton_browseImageDir, &QPushButton::clicked, this, &MainWindow::on_browseImageDir_clicked);
    connect(ui->pushButton_browseOutputDir, &QPushButton::clicked, this, &MainWindow::on_browseOutputDir_clicked);
    connect(ui->pushButton_startStitching, &QPushButton::clicked, this, &MainWindow::on_startStitching_clicked);
    connect(ui->pushButton_reset, &QPushButton::clicked, this, &MainWindow::on_reset_clicked);

    connect(ui->action_openImages, &QAction::triggered, this, &MainWindow::on_openImages_triggered);
    connect(ui->action_exit, &QAction::triggered, this, &MainWindow::on_exit_triggered);
    connect(ui->action_about, &QAction::triggered, this, &MainWindow::on_about_triggered);
}

QString MainWindow::buildOutputTag() const
{
    // 检测器缩写
    static const char* detAbbrev[] = {"sp", "sift", "orb", "surf"};
    // 匹配器缩写
    static const char* matchAbbrev[] = {"lg", "bf"};
    // 模式缩写
    static const char* modeAbbrev[] = {"pano", "scan"};

    int detIdx = ui->comboBox_detector->currentIndex();
    int matchIdx = ui->comboBox_matcher->currentIndex();
    int modeIdx = ui->comboBox_mode->currentIndex();
    float mt = static_cast<float>(ui->doubleSpinBox_matchThreshold->value());
    float ct = static_cast<float>(ui->doubleSpinBox_confidenceThreshold->value());

    return QString("%1_%2_%3_mt%4_ct%5")
        .arg(detAbbrev[detIdx])
        .arg(matchAbbrev[matchIdx])
        .arg(modeAbbrev[modeIdx])
        .arg(mt, 0, 'g', 2)
        .arg(ct, 0, 'g', 2);
}

void MainWindow::on_browseImageDir_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, m_isEnglish ? "Select Image Folder" : "选择图像文件夹", "");
    if (!dir.isEmpty()) {
        ui->lineEdit_imageDir->setText(dir);
        appendLog(m_isEnglish ? "Selected image folder: " + dir : "选择图像文件夹: " + dir);

        // 根据当前算法参数自动生成输出文件夹名
        QString tag = buildOutputTag();
        QString defaultOutputDir = dir + "/" + tag;
        ui->lineEdit_outputDir->setText(defaultOutputDir);
        ui->lineEdit_outputName->setText(tag + ".jpg");
    }
}

void MainWindow::on_browseOutputDir_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, m_isEnglish ? "Select Output Folder" : "选择输出文件夹", "");
    if (!dir.isEmpty()) {
        ui->lineEdit_outputDir->setText(dir);
        ui->lineEdit_outputName->setText(QFileInfo(dir).fileName() + ".jpg");
        appendLog(m_isEnglish ? "Selected output folder: " + dir : "选择输出文件夹: " + dir);
    }
}

void MainWindow::on_modelSettings_triggered()
{
    QDialog dialog(this);
    dialog.setWindowTitle(m_isEnglish ? "Model Settings" : "模型设置");
    dialog.setMinimumWidth(500);

    QVBoxLayout* mainLayout = new QVBoxLayout(&dialog);

    // SuperPoint 模型
    QHBoxLayout* spLayout = new QHBoxLayout();
    QLabel* spLabel = new QLabel(m_isEnglish ? "SuperPoint Model:" : "SuperPoint模型：", &dialog);
    QLineEdit* spEdit = new QLineEdit(&dialog);
    spEdit->setText(m_superPointPath);
    spEdit->setPlaceholderText(m_isEnglish ? "Select superpoint.onnx" : "请选择 superpoint.onnx");
    QPushButton* spBrowse = new QPushButton(m_isEnglish ? "Browse..." : "浏览...", &dialog);
    spLayout->addWidget(spLabel);
    spLayout->addWidget(spEdit, 1);
    spLayout->addWidget(spBrowse);
    mainLayout->addLayout(spLayout);

    // LightGlue 模型
    QHBoxLayout* lgLayout = new QHBoxLayout();
    QLabel* lgLabel = new QLabel(m_isEnglish ? "LightGlue Model:" : "LightGlue模型：", &dialog);
    QLineEdit* lgEdit = new QLineEdit(&dialog);
    lgEdit->setText(m_lightGluePath);
    lgEdit->setPlaceholderText(m_isEnglish ? "Select lightglue.onnx" : "请选择 lightglue.onnx");
    QPushButton* lgBrowse = new QPushButton(m_isEnglish ? "Browse..." : "浏览...", &dialog);
    lgLayout->addWidget(lgLabel);
    lgLayout->addWidget(lgEdit, 1);
    lgLayout->addWidget(lgBrowse);
    mainLayout->addLayout(lgLayout);

    // 浏览按钮连接
    connect(spBrowse, &QPushButton::clicked, [&]() {
        QString file = QFileDialog::getOpenFileName(&dialog, m_isEnglish ? "Select SuperPoint Model" : "选择SuperPoint模型文件", "", "ONNX Files (*.onnx)");
        if (!file.isEmpty()) spEdit->setText(file);
    });
    connect(lgBrowse, &QPushButton::clicked, [&]() {
        QString file = QFileDialog::getOpenFileName(&dialog, m_isEnglish ? "Select LightGlue Model" : "选择LightGlue模型文件", "", "ONNX Files (*.onnx)");
        if (!file.isEmpty()) lgEdit->setText(file);
    });

    // 确定/取消按钮
    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    QPushButton* okBtn = new QPushButton(m_isEnglish ? "OK" : "确定", &dialog);
    QPushButton* cancelBtn = new QPushButton(m_isEnglish ? "Cancel" : "取消", &dialog);
    btnLayout->addWidget(okBtn);
    btnLayout->addWidget(cancelBtn);
    mainLayout->addLayout(btnLayout);

    connect(okBtn, &QPushButton::clicked, &dialog, &QDialog::accept);
    connect(cancelBtn, &QPushButton::clicked, &dialog, &QDialog::reject);

    if (dialog.exec() == QDialog::Accepted) {
        m_superPointPath = spEdit->text();
        m_lightGluePath = lgEdit->text();
        appendLog(m_isEnglish ? "Model paths updated" : "模型路径已更新");
    }
}

void MainWindow::on_startStitching_clicked()
{
    if (!validateParameters()) {
        return;
    }

    if (m_workerThread && m_workerThread->isRunning()) {
        appendLog(m_isEnglish ? "A stitching task is already running, please wait" : "已有拼接任务正在运行，请等待当前任务结束");
        return;
    }

    // 清理上一次拼接的 worker/thread（延迟清理，确保 queued 信号已处理）
    if (m_workerThread) {
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
        m_worker = nullptr;
    }

    updateUIState(true);
    ui->progressBar->setValue(0);
    m_elapsedTimer.start();
    setStatusIndicator(StateProcessing);

    m_workerThread = new QThread(this);
    m_worker = new StitchingWorker();

    std::string imageDir = ui->lineEdit_imageDir->text().toStdString();
    std::string extension = ui->lineEdit_extension->text().toStdString();
    bool divideImages = ui->checkBox_divideImages->isChecked();
    std::wstring superPointPath = m_superPointPath.toStdWString();
    std::wstring lightGluePath = m_lightGluePath.toStdWString();
    cv::Stitcher::Mode mode = (ui->comboBox_mode->currentIndex() == 0) ? cv::Stitcher::PANORAMA : cv::Stitcher::SCANS;

    // 拼接方向：0=自动, 1=水平, 2=垂直
    static const int waveCorrectKinds[] = {
        cv::detail::WAVE_CORRECT_AUTO,
        cv::detail::WAVE_CORRECT_HORIZ,
        cv::detail::WAVE_CORRECT_VERT
    };
    int waveCorrectKind = waveCorrectKinds[ui->comboBox_direction->currentIndex()];

    float matchThreshold = ui->doubleSpinBox_matchThreshold->value();
    float confidenceThreshold = ui->doubleSpinBox_confidenceThreshold->value();

    std::string outputDir = ui->lineEdit_outputDir->text().toStdString();
    std::string outputName = ui->lineEdit_outputName->text().toStdString();
    bool saveMatching = ui->checkBox_showMatching->isChecked();

    // 检测器和匹配器选择
    static const QStringList detectors = {"superpoint", "sift", "orb", "surf"};
    static const QStringList matchers = {"lightglue", "bfmatcher"};
    std::string detector = detectors[ui->comboBox_detector->currentIndex()].toStdString();
    std::string matcher = matchers[ui->comboBox_matcher->currentIndex()].toStdString();
    bool useGpu = ui->checkBox_gpu->isChecked();

    const QString inputDir = ui->lineEdit_imageDir->text();
    const QString inputFolderName = QFileInfo(inputDir).fileName();
    const QString runId = QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz");
    m_currentRunTag = m_isEnglish
        ? QString("[Task:%1|Input:%2]").arg(runId, inputFolderName.isEmpty() ? inputDir : inputFolderName)
        : QString("[任务:%1|输入:%2]").arg(runId, inputFolderName.isEmpty() ? inputDir : inputFolderName);
    appendLog("==================================================");
    appendLog(m_isEnglish ? QString("%1 Stitching started").arg(m_currentRunTag) : QString("%1 开始拼接").arg(m_currentRunTag));
    appendLog(m_isEnglish ? QString("%1 Input folder: %2").arg(m_currentRunTag, inputDir) : QString("%1 输入文件夹: %2").arg(m_currentRunTag, inputDir));
    ui->statusbar->showMessage(m_isEnglish ? "Stitching in progress..." : "正在拼接...");

    m_worker->setParameters(
        imageDir, extension, divideImages,
        superPointPath, lightGluePath, mode,
        waveCorrectKind,
        matchThreshold, confidenceThreshold,
        outputDir, outputName, saveMatching,
        detector, matcher, useGpu, m_isEnglish);

    m_worker->moveToThread(m_workerThread);

    connect(m_workerThread, &QThread::started, m_worker, &StitchingWorker::process);
    connect(m_worker, &StitchingWorker::finished, this, &MainWindow::on_stitchingFinished);
    connect(m_worker, &StitchingWorker::progressChanged, this, &MainWindow::on_stitchingProgress);
    connect(m_worker, &StitchingWorker::logMessage, this, &MainWindow::on_stitchingLog);
    connect(m_worker, &StitchingWorker::resultReady, this, &MainWindow::on_stitchingResult);
    connect(m_worker, &StitchingWorker::errorOccurred, this, &MainWindow::on_stitchingError);
    connect(m_worker, &StitchingWorker::finished, m_workerThread, &QThread::quit);
    // 不在 finished 时 deleteLater——延迟到下一次拼接时清理
    // 避免 worker 在主处理完 queued resultReady 信号前被销毁导致堆损坏

    m_workerThread->start();
}

void MainWindow::on_reset_clicked()
{
    // 清除图像和输出文件夹路径
    ui->lineEdit_imageDir->clear();
    // 默认输出路径也清除，或者不清理outputName
    // 用户要求清除输出文件夹
    ui->lineEdit_outputDir->clear();
    
    // 重置进度条
    ui->progressBar->setValue(0);

    // 重置状态指示灯
    m_lastElapsedSec = -1;
    setStatusIndicator(StateReady);

    // 重置结果显示
    m_resultScene->clear();
    m_resultPixmapItem = nullptr;
    m_currentResult = QPixmap();
    m_rotationAngle = 0;
    m_zoomFactor = 1.0;
    ui->graphicsView_result->resetTransform();
    ui->graphicsView_result->setDragMode(QGraphicsView::NoDrag);
    
    // 允许用户重新操作
    updateUIState(false);
    ui->statusbar->showMessage(m_isEnglish ? "Reset complete, select a new image folder" : "已重置，可重新选择图像文件夹");

    appendLog(m_isEnglish ? "--- Reset complete, please select a new image folder ---" : "--- 重置完成，请选择新的图像文件夹 ---");
}

void MainWindow::on_exportLog_triggered()
{
    const QString logText = ui->textEdit_log->toPlainText();
    if (logText.isEmpty()) {
        QMessageBox::information(this, m_isEnglish ? "Info" : "提示", m_isEnglish ? "No log content to export." : "当前没有可导出的日志内容。");
        return;
    }

    const QString defaultName = QString("PVStitch_Log_%1.txt")
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    const QString filePath = QFileDialog::getSaveFileName(
        this,
        m_isEnglish ? "Export Log" : "导出日志",
        defaultName,
        "Text Files (*.txt);;All Files (*)");

    if (filePath.isEmpty()) {
        return;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, m_isEnglish ? "Export Failed" : "导出失败", m_isEnglish ? "Cannot write log file. Please check path permissions." : "无法写入日志文件，请检查路径权限。");
        return;
    }

    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    out << logText;
    file.close();

    appendLog(m_isEnglish ? QString("Log exported: %1").arg(filePath) : QString("日志已导出: %1").arg(filePath));
    ui->statusbar->showMessage(m_isEnglish ? "Log exported successfully" : "日志导出成功", 3000);
}

void MainWindow::on_openImages_triggered()
{
    on_browseImageDir_clicked();
}

void MainWindow::on_exit_triggered()
{
    close();
}

void MainWindow::on_about_triggered()
{
    if (m_isEnglish) {
        QMessageBox::about(this, "About PVStitch",
            "PVStitch - Image Stitching Tool\n\n"
            "Supported feature detectors: SuperPoint, SIFT, ORB, SURF\n"
            "Supported matchers: LightGlue, BFMatcher\n"
            "Version: 3.1\n\n"
            "Built with Qt 6.5.3 + OpenCV 4.10 + ONNX Runtime");
    } else {
        QMessageBox::about(this, "关于 StitchGUI",
            "StitchGUI - 图像拼接工具\n\n"
            "支持特征检测器: SuperPoint, SIFT, ORB, SURF\n"
            "支持匹配器: LightGlue, BFMatcher\n"
            "版本: 1.1\n\n"
            "使用Qt 6.5.3 + OpenCV 4.10 + ONNX Runtime开发");
    }
}

void MainWindow::on_operationGuide_triggered()
{
    QDialog guideDialog(this);
    guideDialog.setWindowTitle(m_isEnglish ? "Operation Guide" : "操作指南");
    guideDialog.resize(780, 560);

    QVBoxLayout* mainLayout = new QVBoxLayout(&guideDialog);
    QTextBrowser* docViewer = new QTextBrowser(&guideDialog);
    docViewer->setOpenExternalLinks(true);
    docViewer->setStyleSheet(
        "QTextBrowser {"
        "  background: #ffffff;"
        "  border: 1px solid #d7dde7;"
        "  border-radius: 8px;"
        "  padding: 10px;"
        "  line-height: 1.5;"
        "}"
    );

    if (m_isEnglish) {
        docViewer->setHtml(
            "<h2 style=’color:#1f2937;margin-bottom:8px;’>PVStitch Operation Guide</h2>"
            "<p style=’color:#4b5563;’>This guide helps beginners complete their first stitching task quickly.</p>"
            "<h3 style=’color:#2563eb;’>1. Preparation</h3>"
            "<ol>"
            "<li>In <b>Image Settings</b>, select the folder containing images to stitch.</li>"
            "<li>Confirm the image extension (e.g. <code>*.jpg</code>).</li>"
            "<li>Enable ‘Split Image’ option if needed.</li>"
            "</ol>"
            "<h3 style=’color:#2563eb;’>2. Algorithm & Parameters</h3>"
            "<ol>"
            "<li>In <b>Stitching Parameters</b>, select a <b>Feature Detector</b>: SuperPoint (deep learning), SIFT, ORB, or SURF (classical).</li>"
            "<li>Select a <b>Feature Matcher</b>: LightGlue (SuperPoint only) or BFMatcher (universal).</li>"
            "<li>To compare detectors, use BFMatcher for consistent results.</li>"
            "<li>SuperPoint requires ONNX model paths in <b>Model Settings</b>.</li>"
            "<li>Fine-tune Match Threshold and Confidence Threshold based on image quality.</li>"
            "</ol>"
            "<h3 style=’color:#2563eb;’>3. Output & Execution</h3>"
            "<ol>"
            "<li>In <b>Output Settings</b>, select the output directory and filename.</li>"
            "<li>Click <b>Start Stitching</b> and monitor the log below.</li>"
            "<li>After completion, use <b>File -> Export Log...</b> to save the run log.</li>"
            "</ol>"
            "<h3 style=’color:#2563eb;’>Quick Tips</h3>"
            "<ul>"
            "<li><b>Tip 1:</b> Click ‘Reset’ before a second stitching to quickly clear inputs.</li>"
            "<li><b>Tip 2:</b> When errors occur, check log lines with task identifiers first.</li>"
            "<li><b>Tip 3:</b> For abnormal results, try Panorama mode with lower thresholds first.</li>"
            "</ul>"
        );
    } else {
        docViewer->setHtml(
            "<h2 style=’color:#1f2937;margin-bottom:8px;’>StitchGUI 操作指南</h2>"
            "<p style=’color:#4b5563;’>本指南用于帮助新手快速完成首次拼接任务。</p>"
            "<h3 style=’color:#2563eb;’>一、准备阶段</h3>"
            "<ol>"
            "<li>在<b>图像设置</b>中选择待拼接图像文件夹。</li>"
            "<li>确认图像扩展名（如 <code>*.jpg</code>）。</li>"
            "<li>按需启用’分割图像’选项。</li>"
            "</ol>"
            "<h3 style=’color:#2563eb;’>二、算法与参数</h3>"
            "<ol>"
            "<li>在<b>拼接参数</b>中选择<b>特征检测器</b>：SuperPoint（深度学习）、SIFT、ORB 或 SURF（经典算法）。</li>"
            "<li>选择<b>特征匹配器</b>：LightGlue（仅SuperPoint可用）或 BFMatcher（通用）。</li>"
            "<li>如需对比不同检测器效果，建议统一使用 BFMatcher 保证匹配器一致。</li>"
            "<li>SuperPoint 模式需在<b>模型设置</b>中指定 ONNX 模型路径。</li>"
            "<li>根据图像质量微调匹配阈值与置信度阈值。</li>"
            "</ol>"
            "<h3 style=’color:#2563eb;’>三、输出与执行</h3>"
            "<ol>"
            "<li>在<b>输出设置</b>中选择输出目录与输出文件名。</li>"
            "<li>点击<b>开始拼接</b>执行任务，并观察下方日志。</li>"
            "<li>结束后可在<b>文件 -> 导出日志...</b>中保存运行记录。</li>"
            "</ol>"
            "<h3 style=’color:#2563eb;’>快捷提示</h3>"
            "<ul>"
            "<li><b>提示 1：</b>第二次拼接前可点击’重置任务’快速清空路径输入。</li>"
            "<li><b>提示 2：</b>出现错误时优先查看日志中带任务标识的报错行。</li>"
            "<li><b>提示 3：</b>若结果异常，建议先用 Panorama 模式并降低阈值尝试。</li>"
            "</ul>"
        );
    }

    QPushButton* closeButton = new QPushButton(m_isEnglish ? "Got it" : "我已了解", &guideDialog);
    closeButton->setMinimumWidth(120);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);

    mainLayout->addWidget(docViewer);
    mainLayout->addLayout(buttonLayout);

    connect(closeButton, &QPushButton::clicked, &guideDialog, &QDialog::accept);
    guideDialog.exec();
}

void MainWindow::on_errorGuide_triggered()
{
    QDialog dialog(this);
    dialog.setWindowTitle(m_isEnglish ? "Error Guide" : "报错说明");
    dialog.resize(800, 620);

    QVBoxLayout* mainLayout = new QVBoxLayout(&dialog);
    QTextBrowser* viewer = new QTextBrowser(&dialog);
    viewer->setOpenExternalLinks(true);
    viewer->setStyleSheet(
        "QTextBrowser {"
        "  background: #ffffff;"
        "  border: 1px solid #d7dde7;"
        "  border-radius: 8px;"
        "  padding: 12px;"
        "  line-height: 1.6;"
        "}"
    );

    if (m_isEnglish) {
        viewer->setHtml(
            "<h2 style='color:#1f2937;'>Stitching Error Guide</h2>"
            "<p style='color:#6b7280;'>Below are all possible error messages, their causes, and suggested solutions.</p>"

            "<h3 style='color:#dc2626;'>1. No Image Files Found</h3>"
            "<p><b>Cause:</b> No image files matching the extension were found in the specified directory.</p>"
            "<ul>"
            "<li>Extension mismatch (e.g. actual is <code>.JPG</code> but specified <code>*.jpg</code>)</li>"
            "<li>Directory path is wrong or empty</li>"
            "</ul>"
            "<p><b>Solution:</b> Check the image directory and extension settings, confirm files exist.</p>"

            "<h3 style='color:#dc2626;'>2. Need More Images</h3>"
            "<p><b>Cause:</b> Insufficient feature matching between images, stitcher cannot establish valid image connections.</p>"
            "<ul>"
            "<li>Overlap between adjacent images is too small (recommend 30%+ overlap)</li>"
            "<li>Image texture is too uniform (e.g. solid sky, water surface), too few feature points</li>"
            "<li>Matcher threshold is too strict, discarding too many matches</li>"
            "</ul>"
            "<p><b>Solution:</b> Increase image overlap, or lower the match threshold (matchThreshold) and retry.</p>"

            "<h3 style='color:#dc2626;'>3. Homography Estimation Failed</h3>"
            "<p><b>Cause:</b> Feature matches between two images cannot produce a valid Homography matrix.</p>"
            "<ul>"
            "<li><b>Too many outliers:</b> Fewer than 4 correct matches, RANSAC cannot converge. ORB binary descriptors have lower discriminability and are more prone to this</li>"
            "<li><b>Collinear degeneracy:</b> All match points approximately lie on a line, matrix is singular</li>"
            "<li><b>Excessive transformation:</b> Beyond the applicable range of the homography model (e.g. angle difference > 30 degrees)</li>"
            "</ul>"
            "<p><b>Solution:</b> Switch to SIFT or SuperPoint detector, or increase image count and overlap.</p>"

            "<h3 style='color:#dc2626;'>4. Camera Parameters Adjustment Failed</h3>"
            "<p><b>Cause:</b> Bundle Adjustment optimization did not converge.</p>"
            "<p>Stitching pipeline: Feature matching -> Homography estimation -> Decompose into focal length/rotation -> Optimize reprojection error.</p>"
            "<ul>"
            "<li><b>Poor upstream matching:</b> Inaccurate Homography -> negative or NaN initial focal length -> optimization starts from bad point, cannot converge</li>"
            "<li><b>Pure translation between images:</b> Cannot estimate focal length, parameters degenerate</li>"
            "<li><b>Numerical instability:</b> Rotation matrix loses orthogonality</li>"
            "</ul>"
            "<p><b>Solution:</b> This is a cascading effect of poor matching. Focus on improving matching (change detector/tune threshold) rather than adjusting camera parameters.</p>"

            "<h3 style='color:#dc2626;'>5. Unknown Error / Exception</h3>"
            "<p><b>Cause:</b> Program caught a C++ exception.</p>"
            "<ul>"
            "<li><b>ONNX inference failed:</b> Model file corrupted, path contains special characters, input dimension mismatch</li>"
            "<li><b>Out of memory:</b> Image resolution too high, memory exhausted during stitching</li>"
            "<li><b>OpenCV internal exception:</b> Matrix dimension mismatch, empty matrix operation</li>"
            "</ul>"
            "<p><b>Solution:</b> Check the exception message in the log, verify model files and image sizes.</p>"

            "<hr style='margin:16px 0;'>"
        );
    } else {
        viewer->setHtml(
            "<h2 style='color:#1f2937;'>拼接报错说明</h2>"
            "<p style='color:#6b7280;'>以下列出了所有可能的报错信息、触发原因及解决建议。</p>"

            "<h3 style='color:#dc2626;'>❶ 未找到任何图像文件</h3>"
            "<p><b>原因：</b>指定目录下没有符合扩展名的图像文件。</p>"
            "<ul>"
            "<li>扩展名不匹配（如实际是 <code>.JPG</code> 但填写了 <code>*.jpg</code>）</li>"
            "<li>目录路径错误或目录为空</li>"
            "</ul>"
            "<p><b>解决：</b>检查图像目录和扩展名设置，确认文件存在。</p>"

            "<h3 style='color:#dc2626;'>❷ 需要更多图像</h3>"
            "<p><b>原因：</b>图像之间特征匹配不足，拼接器无法建立有效的图像连接关系。</p>"
            "<ul>"
            "<li>相邻图像重叠区域太小（建议重叠 30% 以上）</li>"
            "<li>图像纹理过于单一（如纯色天空、水面），特征点太少</li>"
            "<li>匹配器阈值过于严格，丢弃了过多匹配</li>"
            "</ul>"
            "<p><b>解决：</b>增加图像重叠度，或降低匹配阈值（matchThreshold）重试。</p>"

            "<h3 style='color:#dc2626;'>❸ 单应性矩阵估计失败</h3>"
            "<p><b>原因：</b>两幅图像之间的特征匹配无法求出有效的 Homography 矩阵。</p>"
            "<ul>"
            "<li><b>匹配外点（outlier）过多：</b>正确匹配少于 4 对，RANSAC 无法收敛。ORB 二进制描述子区分度低，更容易出现此问题</li>"
            "<li><b>匹配点共线退化：</b>所有匹配点近似分布在一条直线上，矩阵奇异</li>"
            "<li><b>图像间变换过大：</b>超出单应性模型的适用范围（如拍摄角度差异 > 30°）</li>"
            "</ul>"
            "<p><b>解决：</b>更换为 SIFT 或 SuperPoint 检测器，或增加图像数量和重叠度。</p>"

            "<h3 style='color:#dc2626;'>❹ 相机参数调整失败</h3>"
            "<p><b>原因：</b>Bundle Adjustment（光束法平差）优化不收敛。</p>"
            "<p>拼接流程为：特征匹配 → 单应性估计 → 分解为相机焦距/旋转 → 优化重投影误差。</p>"
            "<ul>"
            "<li><b>上游匹配质量差：</b>Homography 不准确 → 分解出的初始焦距为负数或 NaN → 优化从坏起点出发，无法收敛</li>"
            "<li><b>图像间纯平移：</b>无法估计焦距，参数退化</li>"
            "<li><b>数值不稳定：</b>旋转矩阵失去正交性</li>"
            "</ul>"
            "<p><b>解决：</b>这是匹配质量差的连锁反应。优先改善匹配（换检测器/调阈值），而非调整相机参数。</p>"

            "<h3 style='color:#dc2626;'>❺ 未知错误 / 异常</h3>"
            "<p><b>原因：</b>程序捕获到 C++ 异常。</p>"
            "<ul>"
            "<li><b>ONNX 推理失败：</b>模型文件损坏、路径含中文、输入维度不匹配</li>"
            "<li><b>内存不足：</b>图像分辨率过高，拼接过程内存耗尽</li>"
            "<li><b>OpenCV 内部异常：</b>矩阵运算维度不匹配、空矩阵操作</li>"
            "</ul>"
            "<p><b>解决：</b>查看日志中的异常信息，检查模型文件和图像尺寸。</p>"

            "<hr style='margin:16px 0;'>"
        );
    }

    QPushButton* closeBtn = new QPushButton(m_isEnglish ? "Close" : "关闭", &dialog);
    closeBtn->setMinimumWidth(100);
    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    btnLayout->addWidget(closeBtn);
    mainLayout->addWidget(viewer);
    mainLayout->addLayout(btnLayout);

    connect(closeBtn, &QPushButton::clicked, &dialog, &QDialog::accept);
    dialog.exec();
}

void MainWindow::on_paramGuide_triggered()
{
    QDialog dialog(this);
    dialog.setWindowTitle(m_isEnglish ? "Parameter Guide" : "参数说明");
    dialog.resize(780, 580);

    QVBoxLayout* mainLayout = new QVBoxLayout(&dialog);
    QTextBrowser* viewer = new QTextBrowser(&dialog);
    viewer->setOpenExternalLinks(true);
    viewer->setStyleSheet(
        "QTextBrowser {"
        "  background: #ffffff;"
        "  border: 1px solid #d7dde7;"
        "  border-radius: 8px;"
        "  padding: 12px;"
        "  line-height: 1.6;"
        "}"
    );

    if (m_isEnglish) {
        viewer->setHtml(
            "<h2 style='color:#1f2937;'>Parameter Guide</h2>"

            "<h3 style='color:#2563eb;'>Image Settings</h3>"
            "<table cellpadding='6' cellspacing='0' style='width:100%;'>"
            "<tr><td><b>Image Directory</b></td><td>Folder path containing images to stitch</td></tr>"
            "<tr><td><b>File Extension</b></td><td>Image filter format, e.g. <code>*.jpg</code>, <code>*.png</code></td></tr>"
            "<tr><td><b>Split Image</b></td><td>Splits each input image horizontally into three parts before stitching, useful for wide scan images</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;'>Stitching Parameters</h3>"
            "<table cellpadding='6' cellspacing='0' style='width:100%;'>"
            "<tr><td><b>Feature Detector</b></td>"
            "<td><b>SuperPoint</b> - Deep learning, highest accuracy, requires ONNX model<br>"
            "<b>SIFT</b> - Classical algorithm, good robustness, moderate speed<br>"
            "<b>ORB</b> - Binary descriptor, fastest, lowest accuracy<br>"
            "<b>SURF</b> - Haar wavelet features, faster than SIFT</td></tr>"
            "<tr><td><b>Feature Matcher</b></td>"
            "<td><b>LightGlue</b> - Deep learning matching, only works with SuperPoint<br>"
            "<b>BFMatcher</b> - Brute-force matching + Lowe's ratio test, works with all detectors</td></tr>"
            "<tr><td><b>Stitching Mode</b></td>"
            "<td><b>Panorama</b> - Panorama mode, uses Homography transform (for rotational shooting)<br>"
            "<b>Scans</b> - Scan mode, uses affine transform (for flat scanned documents)</td></tr>"
            "<tr><td><b>Stitching Direction</b></td>"
            "<td><b>Auto</b> - Determined automatically by algorithm<br>"
            "<b>Horizontal</b> - Force horizontal stitching (left-right arrangement)<br>"
            "<b>Vertical</b> - Force vertical stitching (top-bottom arrangement)</td></tr>"
            "<tr><td><b>Match Threshold</b></td>"
            "<td>Range 0~1, default 0.6. Controls the strictness of feature matching filtering.<br>"
            "Higher values = looser matching, more matches but also more false matches.<br>"
            "Recommended: SuperPoint 0.3~0.5, SIFT/SURF 0.5~0.7, ORB 0.6~0.8</td></tr>"
            "<tr><td><b>Confidence Threshold</b></td>"
            "<td>Range 0~1, default 0.2. Controls the minimum confidence for image pairs to participate in stitching.<br>"
            "Higher values = stricter quality requirement, low-quality pairs will be excluded.<br>"
            "Recommended: Generally keep default 0.1~0.3</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;'>Output Settings</h3>"
            "<table cellpadding='6' cellspacing='0' style='width:100%;'>"
            "<tr><td><b>Output Folder</b></td><td>Directory for saving stitching results, auto-generated based on algorithm parameters</td></tr>"
            "<tr><td><b>Output Filename</b></td><td>Filename for stitching result, format like <code>sp_lg_pano_mt0.3_ct0.2.jpg</code></td></tr>"
            "<tr><td><b>Save Matching Images</b></td><td>When checked, saves feature matching visualization for each image pair (match_0_1.jpg)</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;'>Output Naming Convention</h3>"
            "<p>Folders and filenames are auto-generated in <code>{detector}_{matcher}_{mode}_mt{matchThreshold}_ct{confidence}</code> format:</p>"
            "<table cellpadding='4' cellspacing='0'>"
            "<tr><td>sp</td><td>SuperPoint</td><td>sift</td><td>SIFT</td></tr>"
            "<tr><td>orb</td><td>ORB</td><td>surf</td><td>SURF</td></tr>"
            "<tr><td>lg</td><td>LightGlue</td><td>bf</td><td>BFMatcher</td></tr>"
            "<tr><td>pano</td><td>Panorama</td><td>scan</td><td>Scans</td></tr>"
            "</table>"
        );
    } else {
        viewer->setHtml(
            "<h2 style='color:#1f2937;'>参数说明</h2>"

            "<h3 style='color:#2563eb;'>图像设置</h3>"
            "<table cellpadding='6' cellspacing='0' style='width:100%;'>"
            "<tr><td><b>图像目录</b></td><td>待拼接图像所在的文件夹路径</td></tr>"
            "<tr><td><b>文件扩展名</b></td><td>图像过滤格式，如 <code>*.jpg</code>、<code>*.png</code></td></tr>"
            "<tr><td><b>分割图像</b></td><td>将每张输入图像水平切分为三份再拼接，适用于单张宽幅扫描图</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;'>拼接参数</h3>"
            "<table cellpadding='6' cellspacing='0' style='width:100%;'>"
            "<tr><td><b>特征检测器</b></td>"
            "<td><b>SuperPoint</b> — 深度学习，精度最高，需 ONNX 模型<br>"
            "<b>SIFT</b> — 经典算法，鲁棒性好，速度适中<br>"
            "<b>ORB</b> — 二进制描述子，速度最快，精度最低<br>"
            "<b>SURF</b> — Haar 小波特征，速度快于 SIFT</td></tr>"
            "<tr><td><b>特征匹配器</b></td>"
            "<td><b>LightGlue</b> — 深度学习匹配，仅配合 SuperPoint 使用<br>"
            "<b>BFMatcher</b> — 暴力匹配 + Lowe's 比率测试，适用于所有检测器</td></tr>"
            "<tr><td><b>拼接模式</b></td>"
            "<td><b>Panorama</b> — 全景模式，使用 Homography 变换（适用于旋转拍摄）<br>"
            "<b>Scans</b> — 扫描模式，使用仿射变换（适用于平面扫描件）</td></tr>"
            "<tr><td><b>拼接方向</b></td>"
            "<td><b>自动</b> — 由算法自动判断<br>"
            "<b>水平</b> — 强制水平拼接（左右排列）<br>"
            "<b>垂直</b> — 强制垂直拼接（上下排列）</td></tr>"
            "<tr><td><b>匹配阈值</b></td>"
            "<td>范围 0~1，默认 0.6。控制特征匹配的筛选严格程度。<br>"
            "值越大，匹配条件越宽松，匹配数量越多但误匹配也越多。<br>"
            "建议：SuperPoint 0.3~0.5，SIFT/SURF 0.5~0.7，ORB 0.6~0.8</td></tr>"
            "<tr><td><b>置信度阈值</b></td>"
            "<td>范围 0~1，默认 0.2。控制图像对是否参与拼接的最低可信度。<br>"
            "值越大，要求匹配质量越高，低质量图像对会被排除。<br>"
            "建议：一般保持默认 0.1~0.3 即可</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;'>输出设置</h3>"
            "<table cellpadding='6' cellspacing='0' style='width:100%;'>"
            "<tr><td><b>输出文件夹</b></td><td>拼接结果保存目录，会根据算法参数自动生成</td></tr>"
            "<tr><td><b>输出文件名</b></td><td>拼接结果的文件名，格式如 <code>sp_lg_pano_mt0.3_ct0.2.jpg</code></td></tr>"
            "<tr><td><b>保存匹配图</b></td><td>勾选后会在输出目录保存每对图像的特征匹配可视化图（match_0_1.jpg）</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;'>输出命名规则</h3>"
            "<p>文件夹和文件名自动按 <code>{检测器}_{匹配器}_{模式}_mt{匹配阈值}_ct{置信度}</code> 格式生成：</p>"
            "<table cellpadding='4' cellspacing='0'>"
            "<tr><td>sp</td><td>SuperPoint</td><td>sift</td><td>SIFT</td></tr>"
            "<tr><td>orb</td><td>ORB</td><td>surf</td><td>SURF</td></tr>"
            "<tr><td>lg</td><td>LightGlue</td><td>bf</td><td>BFMatcher</td></tr>"
            "<tr><td>pano</td><td>Panorama</td><td>scan</td><td>Scans</td></tr>"
            "</table>"
        );
    }

    QPushButton* closeBtn = new QPushButton(m_isEnglish ? "Close" : "关闭", &dialog);
    closeBtn->setMinimumWidth(100);
    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    btnLayout->addWidget(closeBtn);
    mainLayout->addWidget(viewer);
    mainLayout->addLayout(btnLayout);

    connect(closeBtn, &QPushButton::clicked, &dialog, &QDialog::accept);
    dialog.exec();
}

void MainWindow::on_switchLanguage_triggered()
{
    m_isEnglish = !m_isEnglish;
    retranslateUi();
}

void MainWindow::retranslateUi()
{
    if (m_isEnglish) {
        setWindowTitle("PVStitch");

        // Group boxes
        ui->groupBox->setTitle("Image Settings");
        ui->groupBox_3->setTitle("Stitching Parameters");
        ui->groupBox_4->setTitle("Output Settings");
        ui->groupBox_inputImages->setTitle("Input Images");
        ui->groupBox_5->setTitle("Result Preview");

        // Labels
        ui->label->setText("Image Folder:");
        ui->label_2->setText("Image Extension:");
        ui->label_detector->setText("Feature Detector:");
        ui->label_matcher->setText("Feature Matcher:");
        ui->label_5->setText("Stitching Mode:");
        ui->label_direction->setText("Stitching Direction:");
        ui->label_6->setText("Match Threshold:");
        ui->label_7->setText("Confidence Threshold:");
        ui->label_8->setText("Output Folder:");
        ui->label_9->setText("Output Filename:");

        // Buttons
        ui->pushButton_startStitching->setText("Start Stitching");
        ui->pushButton_reset->setText("Reset");
        ui->pushButton_browseImageDir->setText("Browse...");
        ui->pushButton_browseOutputDir->setText("Browse...");

        // Placeholders
        ui->lineEdit_imageDir->setPlaceholderText("Select image folder");
        ui->lineEdit_outputDir->setPlaceholderText("Select output folder");

        // Checkboxes
        ui->checkBox_divideImages->setText("Split image (divide each image into 3 parts)");
        ui->checkBox_gpu->setText("LightGlue GPU acceleration (requires NVIDIA GPU)");
        ui->checkBox_showMatching->setText("Save matching result images");

        // Combo box items
        ui->comboBox_mode->setItemText(0, "Panorama");
        ui->comboBox_mode->setItemText(1, "Scans");
        ui->comboBox_direction->setItemText(0, "Auto");
        ui->comboBox_direction->setItemText(1, "Horizontal");
        ui->comboBox_direction->setItemText(2, "Vertical");

        // Menu items
        ui->menu->setTitle("File");
        ui->menu_settings->setTitle("Settings");
        ui->menu_2->setTitle("Help");
        ui->action_openImages->setText("Open Image Folder");
        ui->action_exit->setText("Exit");
        ui->action_about->setText("About");
        ui->action_modelSettings->setText("Model Settings...");
        m_exportLogAction->setText("Export Log...");
        m_operationGuideAction->setText("Operation Guide");
        m_errorGuideAction->setText("Error Guide");
        m_paramGuideAction->setText("Parameter Guide");
        m_langAction->setText("中文");

        // Status bar
        ui->statusbar->showMessage("Ready");
        m_statusLabel->setText("● Ready");

    } else {
        setWindowTitle("PVStitch");

        // Group boxes
        ui->groupBox->setTitle("图像设置");
        ui->groupBox_3->setTitle("拼接参数");
        ui->groupBox_4->setTitle("输出设置");
        ui->groupBox_inputImages->setTitle("待拼接图片");
        ui->groupBox_5->setTitle("结果预览");

        // Labels
        ui->label->setText("图像文件夹：");
        ui->label_2->setText("图像扩展名：");
        ui->label_detector->setText("特征检测器：");
        ui->label_matcher->setText("特征匹配器：");
        ui->label_5->setText("拼接模式：");
        ui->label_direction->setText("拼接方向：");
        ui->label_6->setText("匹配阈值：");
        ui->label_7->setText("拼接置信度阈值：");
        ui->label_8->setText("输出文件夹：");
        ui->label_9->setText("输出文件名：");

        // Buttons
        ui->pushButton_startStitching->setText("开始拼接");
        ui->pushButton_reset->setText("重置任务");
        ui->pushButton_browseImageDir->setText("浏览...");
        ui->pushButton_browseOutputDir->setText("浏览...");

        // Placeholders
        ui->lineEdit_imageDir->setPlaceholderText("请选择待拼接图像文件夹");
        ui->lineEdit_outputDir->setPlaceholderText("请选择输出文件夹");

        // Checkboxes
        ui->checkBox_divideImages->setText("分割图像（将每张图像分成三部分）");
        ui->checkBox_gpu->setText("LightGlue使用GPU加速（需NVIDIA显卡）");
        ui->checkBox_showMatching->setText("保存匹配结果图像");

        // Combo box items
        ui->comboBox_mode->setItemText(0, "Panorama（全景）");
        ui->comboBox_mode->setItemText(1, "Scans（扫描）");
        ui->comboBox_direction->setItemText(0, "自动");
        ui->comboBox_direction->setItemText(1, "水平");
        ui->comboBox_direction->setItemText(2, "垂直");

        // Menu items
        ui->menu->setTitle("文件");
        ui->menu_settings->setTitle("设置");
        ui->menu_2->setTitle("帮助");
        ui->action_openImages->setText("打开图像文件夹");
        ui->action_exit->setText("退出");
        ui->action_about->setText("关于");
        ui->action_modelSettings->setText("模型设置...");
        m_exportLogAction->setText("导出日志...");
        m_operationGuideAction->setText("操作指南");
        m_errorGuideAction->setText("报错说明");
        m_paramGuideAction->setText("参数说明");
        m_langAction->setText("English");

        // Status bar
        ui->statusbar->showMessage("就绪");
        m_statusLabel->setText("● 就绪");
    }
}

void MainWindow::on_stitchingFinished()
{
    updateUIState(false);
    ui->statusbar->showMessage(m_isEnglish ? "Stitching complete" : "拼接完成");
    // 只有在没有错误发生时才显示任务完成消息
    if (!ui->textEdit_log->toPlainText().contains(m_isEnglish ? "Error" : "错误")) {
        appendLog(m_isEnglish ? QString("%1 Stitching task completed").arg(m_currentRunTag) : QString("%1 拼接任务完成").arg(m_currentRunTag));
    }
}

void MainWindow::on_stitchingProgress(int value)
{
    ui->progressBar->setValue(value);
}

void MainWindow::on_stitchingLog(const QString& message)
{
    appendLog(QString("%1 %2").arg(m_currentRunTag, message));
}

void MainWindow::on_stitchingResult()
{
    double elapsed = m_elapsedTimer.elapsed() / 1000.0;
    setStatusIndicator(StateSuccess, elapsed);
    // 从 worker 成员变量读取结果，避免 cv::Mat 跨线程传输
    if (m_worker)
        displayImage(m_worker->lastResult());
    generateReport(elapsed);
    appendLog(m_isEnglish
        ? QString("%1 Stitching result displayed, elapsed %2s").arg(m_currentRunTag).arg(elapsed, 0, 'f', 2)
        : QString("%1 拼接结果已显示，耗时 %2s").arg(m_currentRunTag).arg(elapsed, 0, 'f', 2));
}

void MainWindow::on_stitchingError(const QString& error)
{
    double elapsed = m_elapsedTimer.elapsed() / 1000.0;
    setStatusIndicator(StateError, elapsed);
    QMessageBox::critical(this, m_isEnglish ? "Error" : "错误", error);
    appendLog(m_isEnglish
        ? QString("%1 Error: %2, elapsed %3s").arg(m_currentRunTag, error).arg(elapsed, 0, 'f', 2)
        : QString("%1 错误: %2，耗时 %3s").arg(m_currentRunTag, error).arg(elapsed, 0, 'f', 2));
    ui->statusbar->showMessage(m_isEnglish ? "Stitching failed, please check parameters and log" : "拼接失败，请检查参数和日志");
    updateUIState(false);
}

void MainWindow::updateUIState(bool processing)
{
    ui->pushButton_startStitching->setEnabled(!processing);
    ui->pushButton_reset->setEnabled(!processing);
    ui->groupBox->setEnabled(!processing);
    ui->groupBox_3->setEnabled(!processing);
    ui->groupBox_4->setEnabled(!processing);
}

void MainWindow::displayImage(const cv::Mat& image)
{
    if (image.empty()) {
        return;
    }

    cv::Mat rgbImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, rgbImage, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(image, rgbImage, cv::COLOR_GRAY2RGB);
    }

    QImage qImage(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step, QImage::Format_RGB888);
    m_currentResult = QPixmap::fromImage(qImage);

    m_resultScene->clear();
    m_resultPixmapItem = m_resultScene->addPixmap(m_currentResult);

    // 场景矩形设为图片矩形（无多余边界，避免滚动条干扰）
    m_resultScene->setSceneRect(m_currentResult.rect());

    // 重置变换参数
    m_rotationAngle = 0;
    m_zoomFactor = 1.0;

    // 初始 fitInView 并记录此时视图中心在场景中的位置
    ui->graphicsView_result->resetTransform();
    ui->graphicsView_result->fitInView(m_resultPixmapItem, Qt::KeepAspectRatio);

    // 根据 fitInView 后的实际缩放反算 m_zoomFactor
    QTransform fitTransform = ui->graphicsView_result->transform();
    m_zoomFactor = fitTransform.m11();  // 假设均匀缩放

    // 记录视图中心对应的场景坐标
    m_viewCenter = ui->graphicsView_result->mapToScene(
        ui->graphicsView_result->viewport()->rect().center());
}

void MainWindow::loadInputImages(const QString& dir)
{
    // 清空旧缩略图
    QLayout* layout = ui->hLayout_inputImages;
    while (QLayoutItem* item = layout->takeAt(0)) {
        if (item->widget()) {
            delete item->widget();
        }
        delete item;
    }

    if (dir.isEmpty()) return;

    // 获取文件列表
    std::vector<std::string> files;
    libpano::get_filenames_with_absolute_path(dir.toStdString(), files, ui->lineEdit_extension->text().toStdString());

    // 根据滚动区域实际可用高度计算缩略图高度
    int targetHeight = ui->scrollArea_input->viewport()->height() - 8;
    if (targetHeight < 40) targetHeight = 120;

    for (const auto& filePath : files) {
        QString qPath = QString::fromStdString(filePath);
        QImageReader reader(qPath);
        reader.setAutoTransform(true);
        QImage img = reader.read();
        if (img.isNull()) continue;

        QPixmap pixmap = QPixmap::fromImage(img);
        QPixmap scaled = pixmap.scaledToHeight(targetHeight, Qt::SmoothTransformation);

        QLabel* thumbLabel = new QLabel();
        thumbLabel->setPixmap(scaled);
        thumbLabel->setToolTip(QFileInfo(qPath).fileName());
        thumbLabel->setFixedSize(scaled.size());
        thumbLabel->setStyleSheet("border: 1px solid #d0d8e3; border-radius: 4px; margin: 2px;");
        layout->addWidget(thumbLabel);
    }

    // 添加弹性空间，让缩略图靠左排列
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
}

void MainWindow::applyTransform()
{
    if (!m_resultPixmapItem) return;

    // 在视图上构建变换：先缩放，再旋转
    QTransform t;
    t.scale(m_zoomFactor, m_zoomFactor);
    t.rotate(m_rotationAngle);
    ui->graphicsView_result->setTransform(t);

    // 视图中心对准 m_viewCenter（场景坐标）
    ui->graphicsView_result->centerOn(m_viewCenter);
}

void MainWindow::setStatusIndicator(StitchState state, double elapsedSec)
{
    m_stitchState = state;
    if (elapsedSec >= 0)
        m_lastElapsedSec = elapsedSec;

    QString text, bg, border, color;
    switch (state) {
    case StateReady:
        text = m_isEnglish ? "● Ready" : "● 就绪";
        bg = "#eafaf1"; border = "#27ae60"; color = "#27ae60"; break;
    case StateProcessing:
        text = m_isEnglish ? "● Processing" : "● 拼接中";
        bg = "#fef9e7"; border = "#f39c12"; color = "#f39c12"; break;
    case StateSuccess:
        text = m_isEnglish ? "● Done" : "● 完成";
        bg = "#eafaf1"; border = "#27ae60"; color = "#27ae60"; break;
    case StateError:
        text = m_isEnglish ? "● Failed" : "● 失败";
        bg = "#fdedec"; border = "#e74c3c"; color = "#e74c3c"; break;
    }
    m_statusLabel->setText(text);
    m_statusLabel->setStyleSheet(
        QString("QLabel { background-color: %1; border: 1.5px solid %2; border-radius: 6px; "
                "padding: 2px 10px; font-size: 14px; font-weight: bold; color: %3; }")
        .arg(bg, border, color));

    if (m_lastElapsedSec >= 0 && state != StateProcessing)
        m_timeLabel->setText(m_isEnglish
            ? QString("⏱ Elapsed %1 s").arg(m_lastElapsedSec, 0, 'f', 2)
            : QString("⏱ 耗时 %1 s").arg(m_lastElapsedSec, 0, 'f', 2));
    else if (state == StateProcessing)
        m_timeLabel->setText(m_isEnglish ? "⏱ Timing..." : "⏱ 计时中...");
    else
        m_timeLabel->setText("");
}


void MainWindow::generateReport(double elapsedSec)
{
    QStringList detectors = {"superpoint", "sift", "orb", "surf"};
    QStringList matchers = {"lightglue", "bfmatcher"};
    QStringList modes = {"panorama", "scans"};
    QStringList directions = {"auto", "horiz", "vert"};

    QJsonObject input;
    input[m_isEnglish ? "folder" : "文件夹"] = ui->lineEdit_imageDir->text();
    input[m_isEnglish ? "image_count" : "图片数量"] = ui->hLayout_inputImages->count() - 1;

    QJsonObject params;
    params[m_isEnglish ? "feature_detector" : "特征检测器"] = detectors[ui->comboBox_detector->currentIndex()];
    params[m_isEnglish ? "feature_matcher" : "特征匹配器"] = matchers[ui->comboBox_matcher->currentIndex()];
    params[m_isEnglish ? "stitching_mode" : "拼接模式"] = modes[ui->comboBox_mode->currentIndex()];
    params[m_isEnglish ? "stitching_direction" : "拼接方向"] = directions[ui->comboBox_direction->currentIndex()];
    params[m_isEnglish ? "match_threshold" : "匹配阈值"] = ui->doubleSpinBox_matchThreshold->value();
    params[m_isEnglish ? "confidence_threshold" : "置信度阈值"] = ui->doubleSpinBox_confidenceThreshold->value();
    params[m_isEnglish ? "save_matching_images" : "保存匹配图"] = ui->checkBox_showMatching->isChecked();

    QJsonObject output;
    output[m_isEnglish ? "folder" : "文件夹"] = ui->lineEdit_outputDir->text();
    output[m_isEnglish ? "result_file" : "结果文件"] = ui->lineEdit_outputName->text();

    QJsonObject result;
    result[m_isEnglish ? "status" : "状态"] = m_isEnglish ? "Success" : "成功";
    result[m_isEnglish ? "elapsed_seconds" : "耗时_秒"] = elapsedSec;

    QJsonObject report;
    report[m_isEnglish ? "stitching_report" : "拼接报告"] = m_isEnglish ? "PVStitch auto-generated" : "SuperStitch 自动生成";
    report[m_isEnglish ? "generated_at" : "生成时间"] = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    report[m_isEnglish ? "input" : "输入"] = input;
    report[m_isEnglish ? "algorithm_parameters" : "算法参数"] = params;
    report[m_isEnglish ? "output" : "输出"] = output;
    report[m_isEnglish ? "result" : "结果"] = result;

    QString tag = buildOutputTag();
    QString baseDir = ui->lineEdit_imageDir->text();
    QString reportPath = baseDir + "/" + tag + "/" + tag + ".json";

    QFile file(reportPath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QJsonDocument doc(report);
        file.write(doc.toJson(QJsonDocument::Indented));
        file.close();
        appendLog(m_isEnglish ? "Stitching report saved: " + reportPath : "拼接报告已保存: " + reportPath);
    } else {
        appendLog(m_isEnglish ? "Warning: Cannot save stitching report: " + reportPath : "警告: 无法保存拼接报告: " + reportPath);
    }
}

void MainWindow::on_zoomIn_clicked()
{
    m_zoomFactor *= 1.25;
    m_zoomFactor = qBound(0.01, m_zoomFactor, 100.0);
    applyTransform();
}

void MainWindow::on_zoomOut_clicked()
{
    m_zoomFactor /= 1.25;
    m_zoomFactor = qBound(0.01, m_zoomFactor, 100.0);
    applyTransform();
}

void MainWindow::on_rotateLeft_clicked()
{
    m_rotationAngle = (m_rotationAngle - 90) % 360;
    applyTransform();
}

void MainWindow::on_rotateRight_clicked()
{
    m_rotationAngle = (m_rotationAngle + 90) % 360;
    applyTransform();
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    if (obj != ui->graphicsView_result->viewport() || !m_resultPixmapItem)
        return QMainWindow::eventFilter(obj, event);

    // 左键按下：记录位置，切换为抓手光标
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent* me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::LeftButton) {
            m_lastMousePos = me->pos();
            ui->graphicsView_result->setCursor(Qt::ClosedHandCursor);
            return true;
        }
        if (me->button() == Qt::RightButton) {
            m_lastMousePos = me->pos();
            return true;
        }
    }

    // 鼠标移动：左键拖拽平移，右键拖拽旋转
    if (event->type() == QEvent::MouseMove) {
        QMouseEvent* me = static_cast<QMouseEvent*>(event);
        if (me->buttons() & Qt::LeftButton) {
            // 将视图像素 delta 转换为场景坐标 delta
            QPointF oldScenePos = ui->graphicsView_result->mapToScene(m_lastMousePos);
            QPointF newScenePos = ui->graphicsView_result->mapToScene(me->pos());
            m_viewCenter += (oldScenePos - newScenePos);
            m_lastMousePos = me->pos();
            applyTransform();
            return true;
        }
        if (me->buttons() & Qt::RightButton) {
            int dx = me->pos().x() - m_lastMousePos.x();
            m_rotationAngle = (m_rotationAngle + dx) % 360;
            m_lastMousePos = me->pos();
            applyTransform();
            return true;
        }
    }

    // 左键释放：恢复光标
    if (event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent* me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::LeftButton) {
            ui->graphicsView_result->setCursor(Qt::ArrowCursor);
            return true;
        }
    }

    // 滚轮缩放：以鼠标位置为中心缩放
    if (event->type() == QEvent::Wheel) {
        QWheelEvent* we = static_cast<QWheelEvent*>(event);
        const double factor = 1.15;
        double oldZoom = m_zoomFactor;

        if (we->angleDelta().y() > 0)
            m_zoomFactor *= factor;
        else
            m_zoomFactor /= factor;

        m_zoomFactor = qBound(0.01, m_zoomFactor, 100.0);

        // 以鼠标位置为锚点缩放：保持鼠标下方的场景点不动
        QPointF scenePos = ui->graphicsView_result->mapToScene(we->position().toPoint());
        double ratio = m_zoomFactor / oldZoom;
        m_viewCenter = scenePos + (m_viewCenter - scenePos) / ratio;

        applyTransform();
        return true;
    }

    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::appendLog(const QString& message)
{
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    ui->textEdit_log->appendPlainText(QString("[%1] %2").arg(timestamp, message));
}

bool MainWindow::validateParameters()
{
    if (ui->lineEdit_imageDir->text().isEmpty()) {
        QMessageBox::warning(this, m_isEnglish ? "Warning" : "警告", m_isEnglish ? "Please select an image folder!" : "请选择图像文件夹！");
        return false;
    }

    bool isSuperPoint = (ui->comboBox_detector->currentIndex() == 0);
    bool isLightGlue = (ui->comboBox_matcher->currentIndex() == 0);

    // SuperPoint 检测器需要模型路径
    if (isSuperPoint && m_superPointPath.isEmpty()) {
        QMessageBox::warning(this, m_isEnglish ? "Warning" : "警告", m_isEnglish ? "Please configure the SuperPoint model file in Settings -> Model Settings!" : "请在 设置→模型设置 中配置SuperPoint模型文件！");
        return false;
    }

    // LightGlue 匹配器需要模型路径
    if (isLightGlue && m_lightGluePath.isEmpty()) {
        QMessageBox::warning(this, m_isEnglish ? "Warning" : "警告", m_isEnglish ? "Please configure the LightGlue model file in Settings -> Model Settings!" : "请在 设置→模型设置 中配置LightGlue模型文件！");
        return false;
    }

    if (ui->lineEdit_outputDir->text().isEmpty()) {
        QMessageBox::warning(this, m_isEnglish ? "Warning" : "警告", m_isEnglish ? "Please select an output folder!" : "请选择输出文件夹！");
        return false;
    }

    if (ui->lineEdit_outputName->text().isEmpty()) {
        QMessageBox::warning(this, m_isEnglish ? "Warning" : "警告", m_isEnglish ? "Please enter an output filename!" : "请输入输出文件名！");
        return false;
    }

    return true;
}
