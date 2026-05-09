QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

DEFINES += _CRT_SECURE_NO_WARNINGS
DEFINES += _GLIBCXX_USE_CXX11_ABI=0

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    superpoint.cpp \
    lightglue.cpp \
    utils.cpp \
    file_utils.cpp

HEADERS += \
    mainwindow.h \
    superpoint.h \
    lightglue.h \
    utils.h \
    file_utils.h \
    dlldefine.h

FORMS += \
    mainwindow.ui

RESOURCES += \
    resources.qrc

# OpenCV and ONNX Runtime paths
OPENCV_PATH = "D:/code/DATA/3rdParty/opencv-sdk"
ONNXRUNTIME_PATH = "D:/code/DATA/3rdParty/onnx-sdk"

# Include paths
INCLUDEPATH += \
    $$OPENCV_PATH/include/opencv2 \
    $$OPENCV_PATH/include \
    $$ONNXRUNTIME_PATH/include

# Library paths
CONFIG(debug, debug|release) {
    # Debug mode
    LIBS += -L$$OPENCV_PATH/lib \
            -lopencv_world4100d \
            -L$$ONNXRUNTIME_PATH/lib \
            -lonnxruntime
} else {
    # Release mode
    LIBS += -L$$OPENCV_PATH/lib \
            -lopencv_world4100 \
            -L$$ONNXRUNTIME_PATH/lib \
            -lonnxruntime
}

# Copy DLLs to build directory (Windows only)
win32 {
    OPENCV_BIN = $$replace(OPENCV_PATH, /, \\)\\bin
    ONNX_LIB = $$replace(ONNXRUNTIME_PATH, /, \\)\\lib
    
    CONFIG(debug, debug|release) {
        DESTNUM = debug
        DLL_SUFFIX = d
    } else {
        DESTNUM = release
        DLL_SUFFIX = 
    }
    
    DESTDIR_WIN = $${OUT_PWD}/$${DESTNUM}
    DESTDIR_WIN ~= s,/,\\,g
    
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /y /d "$${OPENCV_BIN}\\opencv_world4100$${DLL_SUFFIX}.dll" "$${DESTDIR_WIN}\\" &)
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /y /d "$${OPENCV_BIN}\\opencv_videoio_ffmpeg4100_64.dll" "$${DESTDIR_WIN}\\" &)
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /y /d "$${ONNX_LIB}\\onnxruntime.dll" "$${DESTDIR_WIN}\\" &)
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
