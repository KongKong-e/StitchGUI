#include "mainwindow.h"

#include <QApplication>
#include <QIcon>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setApplicationName("SuperStitch光伏场景的全景创建");
    a.setWindowIcon(QIcon(":/images/1.png"));
    MainWindow w;
    w.show();
    return a.exec();
}
