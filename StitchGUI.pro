QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

DEFINES += _CRT_SECURE_NO_WARNINGS
DEFINES += _GLIBCXX_USE_CXX11_ABI=0

# OpenMP support (MSVC)
win32-msvc {
    QMAKE_CXXFLAGS += /openmp
}

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    superpoint.cpp \
    lightglue.cpp \
    classical_matcher.cpp \
    utils.cpp \
    file_utils.cpp \
    surf_wrapper.cpp \
    Surf.cpp

HEADERS += \
    mainwindow.h \
    superpoint.h \
    lightglue.h \
    classical_matcher.h \
    utils.h \
    file_utils.h \
    dlldefine.h \
    surf_wrapper.h \
    Surf.h \
    parallelsurf/Image.h \
    parallelsurf/KeyPoint.h \
    parallelsurf/KeyPointDetector.h \
    parallelsurf/KeyPointDescriptor.h \
    parallelsurf/KeyPointDescriptorContext.h \
    parallelsurf/BoxFilter.h \
    parallelsurf/WaveFilter.h \
    parallelsurf/MathStuff.h

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
    $$ONNXRUNTIME_PATH/include \
    $$PWD \
    $$PWD/parallelsurf

# Library paths
CONFIG(debug, debug|release) {
    # Debug mode
    LIBS += $$OPENCV_PATH/lib/opencv_world4100d.lib
    LIBS += $$ONNXRUNTIME_PATH/lib/onnxruntime.lib
} else {
    # Release mode
    LIBS += $$OPENCV_PATH/lib/opencv_world4100.lib
    LIBS += $$ONNXRUNTIME_PATH/lib/onnxruntime.lib
}

# Copy DLLs to build directory (Windows only)
win32 {
    OPENCV_BIN = $$OPENCV_PATH/bin
    ONNX_LIB = $$ONNXRUNTIME_PATH/lib

    CONFIG(debug, debug|release) {
        DESTNUM = debug
        DLL_SUFFIX = d
    } else {
        DESTNUM = release
        DLL_SUFFIX =
    }

    DESTDIR_WIN = $${OUT_PWD}/$${DESTNUM}
    DESTDIR_WIN ~= s,/,\\,g

    OPENCV_BIN ~= s,/,\\,g
    ONNX_LIB ~= s,/,\\,g

    QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$${OPENCV_BIN}\\opencv_world4100$${DLL_SUFFIX}.dll\" \"$${DESTDIR_WIN}\\\" &)
    QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$${OPENCV_BIN}\\opencv_videoio_ffmpeg4100_64.dll\" \"$${DESTDIR_WIN}\\\" &)
    QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$${ONNX_LIB}\\onnxruntime.dll\" \"$${DESTDIR_WIN}\\\" &)

    # Copy model directory to build output
    MODEL_SRC = $$PWD/model
    MODEL_SRC ~= s,/,\\,g
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /E /I /Y \"$${MODEL_SRC}\" \"$${DESTDIR_WIN}\\model\\\" &)
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
