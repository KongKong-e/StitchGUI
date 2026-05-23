#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QMutex>
#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <opencv2/opencv.hpp>
#include "superpoint.h"
#include "lightglue.h"
#include "classical_matcher.h"
#include "surf_wrapper.h"
#include "utils.h"
#include "file_utils.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class StitchingWorker : public QObject
{
    Q_OBJECT

public:
    StitchingWorker(QObject* parent = nullptr);

    void requestStop();

    void setParameters(
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
        const std::string& matcher);

public slots:
    void process();

signals:
    void finished();
    void progressChanged(int value);
    void logMessage(const QString& message);
    void resultReady(const cv::Mat& result);
    void errorOccurred(const QString& error);

private:
    std::string m_imageDir;
    std::string m_extension;
    bool m_divideImages;
    std::wstring m_superPointPath;
    std::wstring m_lightGluePath;
    cv::Stitcher::Mode m_mode;
    int m_waveCorrectKind;
    float m_matchThreshold;
    float m_confidenceThreshold;
    std::string m_outputDir;
    std::string m_outputName;
    bool m_saveMatching;
    std::string m_detector;
    std::string m_matcher;
    bool m_stopRequested;
    QMutex m_mutex;

    cv::Ptr<cv::Feature2D> createDetector();
    cv::Ptr<cv::detail::FeaturesMatcher> createMatcher();
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_browseImageDir_clicked();
    void on_browseOutputDir_clicked();
    void on_startStitching_clicked();
    void on_reset_clicked();
    void on_exportLog_triggered();
    void on_openImages_triggered();
    void on_exit_triggered();
    void on_operationGuide_triggered();
    void on_about_triggered();
    void on_modelSettings_triggered();
    void on_rotateLeft_clicked();
    void on_rotateRight_clicked();
    void on_stitchingFinished();
    void on_stitchingProgress(int value);
    void on_stitchingLog(const QString& message);
    void on_stitchingResult(const cv::Mat& result);
    void on_stitchingError(const QString& error);

private:
    Ui::MainWindow *ui;

    QThread* m_workerThread;
    StitchingWorker* m_worker;
    QString m_currentRunTag;
    QString m_superPointPath;
    QString m_lightGluePath;

    void initializeConnections();
    void updateUIState(bool processing);
    void displayImage(const cv::Mat& image);
    void appendLog(const QString& message);
    bool validateParameters();

    // 输入图片预览
    void loadInputImages(const QString& dir);

    // 结果预览缩放/旋转
    QGraphicsScene* m_resultScene;
    QGraphicsPixmapItem* m_resultPixmapItem;
    QPixmap m_currentResult;
    int m_rotationAngle;
    double m_zoomFactor;
    QPoint m_lastMousePos;

    bool eventFilter(QObject* obj, QEvent* event) override;
};

#endif // MAINWINDOW_H
