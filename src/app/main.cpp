#include "mainwindow.h"

#include <QApplication>
#include <QIcon>
#include <QMetaType>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char *argv[])
{
#ifdef _WIN32
    // 设置控制台代码页为 UTF-8，解决 qDebug 中文乱码
    SetConsoleOutputCP(65001);
#endif

    // 注册 cv::Mat 用于跨线程信号/槽传输（queued connection）
    qRegisterMetaType<cv::Mat>("cv::Mat");

    QApplication a(argc, argv);
    a.setApplicationName("PVStitch");
    a.setWindowIcon(QIcon(":/images/1.png"));
    MainWindow w;
    w.show();
    return a.exec();
}
