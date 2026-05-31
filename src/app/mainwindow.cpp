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

StitchingWorker::StitchingWorker(QObject* parent)
    : QObject(parent)
    , m_divideImages(false)
    , m_mode(cv::Stitcher::PANORAMA)
    , m_waveCorrectKind(cv::detail::WAVE_CORRECT_AUTO)
    , m_matchThreshold(0.2f)
    , m_confidenceThreshold(0.1f)
    , m_saveMatching(false)
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
    const std::string& matcher)
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
        return cv::ORB::create();
    else if (m_detector == "surf")
        return cv::makePtr<SurfDetector>();
    throw std::runtime_error("不支持的检测器: " + m_detector);
}

cv::Ptr<cv::detail::FeaturesMatcher> StitchingWorker::createMatcher()
{
    if (m_matcher == "lightglue")
        return cv::makePtr<LightGlue>(m_lightGluePath, m_mode, m_matchThreshold);
    else
        return cv::makePtr<ClassicalMatcher>(m_mode, m_matchThreshold);
}

void StitchingWorker::process()
{
    try {
        emit logMessage("开始...");

        emit progressChanged(10);
        emit logMessage("正在读取图像...");

        std::vector<cv::Mat> imgs = readImages(m_imageDir, m_extension, m_divideImages);

        if (imgs.empty()) {
            emit errorOccurred("未找到任何图像文件！");
            emit finished();
            return;
        }

        emit logMessage(QString("成功读取 %1 张图像").arg(imgs.size()));
        emit progressChanged(30);

        {
            QMutexLocker locker(&m_mutex);
            if (m_stopRequested) {
                emit logMessage("拼接已停止");
                emit finished();
                return;
            }
        }

        emit logMessage("正在初始化特征检测器: " + QString::fromStdString(m_detector) + "...");
        cv::Ptr<cv::Feature2D> detector = createDetector();

        emit logMessage("正在初始化匹配器: " + QString::fromStdString(m_matcher) + "...");
        cv::Ptr<cv::detail::FeaturesMatcher> matcher = createMatcher();

        emit progressChanged(50);

        {
            QMutexLocker locker(&m_mutex);
            if (m_stopRequested) {
                emit logMessage("拼接已停止");
                emit finished();
                return;
            }
        }

        emit logMessage("正在创建拼接器...");
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(m_mode);
        stitcher->setPanoConfidenceThresh(m_confidenceThreshold);
        stitcher->setFeaturesFinder(detector);
        stitcher->setFeaturesMatcher(matcher);
        stitcher->setWaveCorrectKind(static_cast<cv::detail::WaveCorrectKind>(m_waveCorrectKind));

        emit logMessage("正在执行...");
        cv::Mat pano;
        cv::Stitcher::Status status = stitcher->stitch(imgs, pano);

        emit progressChanged(80);

        if (status == cv::Stitcher::OK) {
            emit logMessage("拼接成功！");

            // 使用用户指定的输出目录（默认为 m_imageDir + "/superstitch"，但也可能被用户修改）
            std::string resDir = m_outputDir;
            if (libpano::create_directory(resDir)) {
                emit logMessage("创建输出目录: " + QString::fromStdString(resDir));
            }

            std::string outputPath = resDir + "/" + m_outputName;
            if (cv::imwrite(outputPath, pano)) {
                emit logMessage("保存拼接结果: " + QString::fromStdString(outputPath));
            } else {
                emit logMessage("警告: 保存拼接结果失败");
            }

            if (m_saveMatching) {
                emit logMessage("正在保存匹配结果...");

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
                    emit logMessage("保存匹配结果: " + QString::fromStdString(matchPath));
                }
            }

            emit progressChanged(100);
            emit resultReady(pano);
        } else {
            QString errorMsg;
            switch (status) {
                case cv::Stitcher::ERR_NEED_MORE_IMGS:
                    errorMsg = "错误: 需要更多图像";
                    break;
                case cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
                    errorMsg = "错误: 单应性矩阵估计失败";
                    break;
                case cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
                    errorMsg = "错误: 相机参数调整失败";
                    break;
                default:
                    errorMsg = "错误: 拼接失败 (错误代码: " + QString::number(status) + ")";
                    break;
            }
            emit errorOccurred(errorMsg);
        }

    } catch (const std::exception& e) {
        emit errorOccurred(QString("异常: %1").arg(e.what()));
    } catch (...) {
        emit errorOccurred("未知错误");
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

    setWindowTitle("SuperStitch光伏场景的全景创建");

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
    ui->horizontalLayout_main->setStretch(0, 5);
    ui->horizontalLayout_main->setStretch(1, 7);

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

    QAction* exportLogMenuAction = new QAction("导出日志...", this);
    ui->menu->clear();
    ui->menu->addAction(ui->action_openImages);
    ui->menu->addAction(exportLogMenuAction);
    ui->menu->addSeparator();
    ui->menu->addAction(ui->action_exit);
    connect(exportLogMenuAction, &QAction::triggered, this, &MainWindow::on_exportLog_triggered);

    QAction* operationGuideAction = new QAction("操作指南", this);
    ui->menu_2->insertAction(ui->action_about, operationGuideAction);
    connect(operationGuideAction, &QAction::triggered, this, &MainWindow::on_operationGuide_triggered);
    
    // 预设模型路径（相对于可执行文件目录）
    QString appDir = QCoreApplication::applicationDirPath();
    m_superPointPath = appDir + "/model/superpoint.onnx";
    m_lightGluePath = appDir + "/model/superpoint_lightglue.onnx";

    initializeConnections();

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

    // 工具栏按钮（居中）：放大、缩小、左旋、右旋
    QPushButton* zoomInBtn = new QPushButton("放大", this);
    QPushButton* zoomOutBtn = new QPushButton("缩小", this);
    QPushButton* rotateLeftBtn = new QPushButton("左旋90°", this);
    QPushButton* rotateRightBtn = new QPushButton("右旋90°", this);
    QHBoxLayout* toolbarLayout = new QHBoxLayout();
    toolbarLayout->addStretch();
    toolbarLayout->addWidget(zoomInBtn);
    toolbarLayout->addSpacing(6);
    toolbarLayout->addWidget(zoomOutBtn);
    toolbarLayout->addSpacing(16);
    toolbarLayout->addWidget(rotateLeftBtn);
    toolbarLayout->addSpacing(6);
    toolbarLayout->addWidget(rotateRightBtn);
    toolbarLayout->addStretch();
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

    appendLog("SuperStitch GUI 已启动");
}

MainWindow::~MainWindow()
{
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
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
    QString dir = QFileDialog::getExistingDirectory(this, "选择图像文件夹", "");
    if (!dir.isEmpty()) {
        ui->lineEdit_imageDir->setText(dir);
        appendLog("选择图像文件夹: " + dir);

        // 根据当前算法参数自动生成输出文件夹名
        QString tag = buildOutputTag();
        QString defaultOutputDir = dir + "/" + tag;
        ui->lineEdit_outputDir->setText(defaultOutputDir);
        ui->lineEdit_outputName->setText(tag + ".jpg");
    }
}

void MainWindow::on_browseOutputDir_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "选择输出文件夹", "");
    if (!dir.isEmpty()) {
        ui->lineEdit_outputDir->setText(dir);
        ui->lineEdit_outputName->setText(QFileInfo(dir).fileName() + ".jpg");
        appendLog("选择输出文件夹: " + dir);
    }
}

void MainWindow::on_modelSettings_triggered()
{
    QDialog dialog(this);
    dialog.setWindowTitle("模型设置");
    dialog.setMinimumWidth(500);

    QVBoxLayout* mainLayout = new QVBoxLayout(&dialog);

    // SuperPoint 模型
    QHBoxLayout* spLayout = new QHBoxLayout();
    QLabel* spLabel = new QLabel("SuperPoint模型：", &dialog);
    QLineEdit* spEdit = new QLineEdit(&dialog);
    spEdit->setText(m_superPointPath);
    spEdit->setPlaceholderText("请选择 superpoint.onnx");
    QPushButton* spBrowse = new QPushButton("浏览...", &dialog);
    spLayout->addWidget(spLabel);
    spLayout->addWidget(spEdit, 1);
    spLayout->addWidget(spBrowse);
    mainLayout->addLayout(spLayout);

    // LightGlue 模型
    QHBoxLayout* lgLayout = new QHBoxLayout();
    QLabel* lgLabel = new QLabel("LightGlue模型：", &dialog);
    QLineEdit* lgEdit = new QLineEdit(&dialog);
    lgEdit->setText(m_lightGluePath);
    lgEdit->setPlaceholderText("请选择 lightglue.onnx");
    QPushButton* lgBrowse = new QPushButton("浏览...", &dialog);
    lgLayout->addWidget(lgLabel);
    lgLayout->addWidget(lgEdit, 1);
    lgLayout->addWidget(lgBrowse);
    mainLayout->addLayout(lgLayout);

    // 浏览按钮连接
    connect(spBrowse, &QPushButton::clicked, [&]() {
        QString file = QFileDialog::getOpenFileName(&dialog, "选择SuperPoint模型文件", "", "ONNX Files (*.onnx)");
        if (!file.isEmpty()) spEdit->setText(file);
    });
    connect(lgBrowse, &QPushButton::clicked, [&]() {
        QString file = QFileDialog::getOpenFileName(&dialog, "选择LightGlue模型文件", "", "ONNX Files (*.onnx)");
        if (!file.isEmpty()) lgEdit->setText(file);
    });

    // 确定/取消按钮
    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    QPushButton* okBtn = new QPushButton("确定", &dialog);
    QPushButton* cancelBtn = new QPushButton("取消", &dialog);
    btnLayout->addWidget(okBtn);
    btnLayout->addWidget(cancelBtn);
    mainLayout->addLayout(btnLayout);

    connect(okBtn, &QPushButton::clicked, &dialog, &QDialog::accept);
    connect(cancelBtn, &QPushButton::clicked, &dialog, &QDialog::reject);

    if (dialog.exec() == QDialog::Accepted) {
        m_superPointPath = spEdit->text();
        m_lightGluePath = lgEdit->text();
        appendLog("模型路径已更新");
    }
}

void MainWindow::on_startStitching_clicked()
{
    if (!validateParameters()) {
        return;
    }

    if (m_workerThread && m_workerThread->isRunning()) {
        appendLog("已有拼接任务正在运行，请等待当前任务结束");
        return;
    }

    updateUIState(true);
    ui->progressBar->setValue(0);

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
    // 根据当前参数刷新输出目录和文件名，确保名称反映实际使用的参数
    QString tag = buildOutputTag();
    QString baseDir = ui->lineEdit_imageDir->text();
    if (!baseDir.isEmpty()) {
        QString autoDir = baseDir + "/" + tag;
        ui->lineEdit_outputDir->setText(autoDir);
        ui->lineEdit_outputName->setText(tag + ".jpg");
    }

    std::string outputDir = ui->lineEdit_outputDir->text().toStdString();
    std::string outputName = ui->lineEdit_outputName->text().toStdString();
    bool saveMatching = ui->checkBox_showMatching->isChecked();

    // 检测器和匹配器选择
    static const QStringList detectors = {"superpoint", "sift", "orb", "surf"};
    static const QStringList matchers = {"lightglue", "bfmatcher"};
    std::string detector = detectors[ui->comboBox_detector->currentIndex()].toStdString();
    std::string matcher = matchers[ui->comboBox_matcher->currentIndex()].toStdString();

    const QString inputDir = ui->lineEdit_imageDir->text();
    const QString inputFolderName = QFileInfo(inputDir).fileName();
    const QString runId = QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz");
    m_currentRunTag = QString("[任务:%1|输入:%2]").arg(runId, inputFolderName.isEmpty() ? inputDir : inputFolderName);
    appendLog("==================================================");
    appendLog(QString("%1 开始拼接").arg(m_currentRunTag));
    appendLog(QString("%1 输入文件夹: %2").arg(m_currentRunTag, inputDir));
    ui->statusbar->showMessage("正在拼接...");

    m_worker->setParameters(
        imageDir, extension, divideImages,
        superPointPath, lightGluePath, mode,
        waveCorrectKind,
        matchThreshold, confidenceThreshold,
        outputDir, outputName, saveMatching,
        detector, matcher);

    m_worker->moveToThread(m_workerThread);

    connect(m_workerThread, &QThread::started, m_worker, &StitchingWorker::process);
    connect(m_worker, &StitchingWorker::finished, this, &MainWindow::on_stitchingFinished);
    connect(m_worker, &StitchingWorker::progressChanged, this, &MainWindow::on_stitchingProgress);
    connect(m_worker, &StitchingWorker::logMessage, this, &MainWindow::on_stitchingLog);
    connect(m_worker, &StitchingWorker::resultReady, this, &MainWindow::on_stitchingResult);
    connect(m_worker, &StitchingWorker::errorOccurred, this, &MainWindow::on_stitchingError);
    connect(m_worker, &StitchingWorker::finished, m_workerThread, &QThread::quit);
    connect(m_workerThread, &QThread::finished, m_workerThread, &QThread::deleteLater);
    connect(m_workerThread, &QThread::finished, m_worker, &StitchingWorker::deleteLater);
    connect(m_workerThread, &QThread::finished, this, [this]() {
        m_workerThread = nullptr;
        m_worker = nullptr;
    });

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
    ui->statusbar->showMessage("已重置，可重新选择图像文件夹");
    
    appendLog("--- 重置完成，请选择新的图像文件夹 ---");
}

void MainWindow::on_exportLog_triggered()
{
    const QString logText = ui->textEdit_log->toPlainText();
    if (logText.isEmpty()) {
        QMessageBox::information(this, "提示", "当前没有可导出的日志内容。");
        return;
    }

    const QString defaultName = QString("SuperStitch_Log_%1.txt")
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    const QString filePath = QFileDialog::getSaveFileName(
        this,
        "导出日志",
        defaultName,
        "Text Files (*.txt);;All Files (*)");

    if (filePath.isEmpty()) {
        return;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "导出失败", "无法写入日志文件，请检查路径权限。");
        return;
    }

    QTextStream out(&file);
    out.setEncoding(QStringConverter::Utf8);
    out << logText;
    file.close();

    appendLog(QString("日志已导出: %1").arg(filePath));
    ui->statusbar->showMessage("日志导出成功", 3000);
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
    QMessageBox::about(this, "关于 StitchGUI",
        "StitchGUI - 图像拼接工具\n\n"
        "支持特征检测器: SuperPoint, SIFT, ORB, SURF\n"
        "支持匹配器: LightGlue, BFMatcher\n"
        "版本: 1.1\n\n"
        "使用Qt 6.5.3 + OpenCV 4.10 + ONNX Runtime开发");
}

void MainWindow::on_operationGuide_triggered()
{
    QDialog guideDialog(this);
    guideDialog.setWindowTitle("操作指南");
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

    docViewer->setHtml(
        "<h2 style='color:#1f2937;margin-bottom:8px;'>StitchGUI 操作指南</h2>"
        "<p style='color:#4b5563;'>本指南用于帮助新手快速完成首次拼接任务。</p>"
        "<h3 style='color:#2563eb;'>一、准备阶段</h3>"
        "<ol>"
        "<li>在<b>图像设置</b>中选择待拼接图像文件夹。</li>"
        "<li>确认图像扩展名（如 <code>*.jpg</code>）。</li>"
        "<li>按需启用‘分割图像’选项。</li>"
        "</ol>"
        "<h3 style='color:#2563eb;'>二、算法与参数</h3>"
        "<ol>"
        "<li>在<b>拼接参数</b>中选择<b>特征检测器</b>：SuperPoint（深度学习）、SIFT、ORB 或 SURF（经典算法）。</li>"
        "<li>选择<b>特征匹配器</b>：LightGlue（仅SuperPoint可用）或 BFMatcher（通用）。</li>"
        "<li>如需对比不同检测器效果，建议统一使用 BFMatcher 保证匹配器一致。</li>"
        "<li>SuperPoint 模式需在<b>模型设置</b>中指定 ONNX 模型路径。</li>"
        "<li>根据图像质量微调匹配阈值与置信度阈值。</li>"
        "</ol>"
        "<h3 style='color:#2563eb;'>三、输出与执行</h3>"
        "<ol>"
        "<li>在<b>输出设置</b>中选择输出目录与输出文件名。</li>"
        "<li>点击<b>开始拼接</b>执行任务，并观察下方日志。</li>"
        "<li>结束后可在<b>文件 -> 导出日志...</b>中保存运行记录。</li>"
        "</ol>"
        "<h3 style='color:#2563eb;'>快捷提示</h3>"
        "<ul>"
        "<li><b>提示 1：</b>第二次拼接前可点击‘重置任务’快速清空路径输入。</li>"
        "<li><b>提示 2：</b>出现错误时优先查看日志中带任务标识的报错行。</li>"
        "<li><b>提示 3：</b>若结果异常，建议先用 Panorama 模式并降低阈值尝试。</li>"
        "</ul>"
    );

    QPushButton* closeButton = new QPushButton("我已了解", &guideDialog);
    closeButton->setMinimumWidth(120);

    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);

    mainLayout->addWidget(docViewer);
    mainLayout->addLayout(buttonLayout);

    connect(closeButton, &QPushButton::clicked, &guideDialog, &QDialog::accept);
    guideDialog.exec();
}

void MainWindow::on_stitchingFinished()
{
    updateUIState(false);
    ui->statusbar->showMessage("拼接完成");
    // 只有在没有错误发生时才显示任务完成消息
    if (!ui->textEdit_log->toPlainText().contains("错误")) {
        appendLog(QString("%1 拼接任务完成").arg(m_currentRunTag));
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

void MainWindow::on_stitchingResult(const cv::Mat& result)
{
    displayImage(result);
    appendLog(QString("%1 拼接结果已显示").arg(m_currentRunTag));
}

void MainWindow::on_stitchingError(const QString& error)
{
    QMessageBox::critical(this, "错误", error);
    appendLog(QString("%1 错误: %2").arg(m_currentRunTag, error));
    ui->statusbar->showMessage("拼接失败，请检查参数和日志");
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
        QMessageBox::warning(this, "警告", "请选择图像文件夹！");
        return false;
    }

    bool isSuperPoint = (ui->comboBox_detector->currentIndex() == 0);
    bool isLightGlue = (ui->comboBox_matcher->currentIndex() == 0);

    // SuperPoint 检测器需要模型路径
    if (isSuperPoint && m_superPointPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请在 设置→模型设置 中配置SuperPoint模型文件！");
        return false;
    }

    // LightGlue 匹配器需要模型路径
    if (isLightGlue && m_lightGluePath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请在 设置→模型设置 中配置LightGlue模型文件！");
        return false;
    }

    if (ui->lineEdit_outputDir->text().isEmpty()) {
        QMessageBox::warning(this, "警告", "请选择输出文件夹！");
        return false;
    }

    if (ui->lineEdit_outputName->text().isEmpty()) {
        QMessageBox::warning(this, "警告", "请输入输出文件名！");
        return false;
    }

    return true;
}
