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
    src/app/main.cpp \
    src/app/mainwindow.cpp \
    src/core/superpoint.cpp \
    src/core/lightglue.cpp \
    src/core/classical_matcher.cpp \
    src/core/surf_wrapper.cpp \
    src/core/Surf.cpp \
    src/utils/utils.cpp \
    src/utils/file_utils.cpp

HEADERS += \
    src/app/mainwindow.h \
    src/core/superpoint.h \
    src/core/lightglue.h \
    src/core/classical_matcher.h \
    src/core/surf_wrapper.h \
    src/core/Surf.h \
    src/core/dlldefine.h \
    src/utils/utils.h \
    src/utils/file_utils.h \
    src/parallelsurf/Image.h \
    src/parallelsurf/KeyPoint.h \
    src/parallelsurf/KeyPointDetector.h \
    src/parallelsurf/KeyPointDescriptor.h \
    src/parallelsurf/KeyPointDescriptorContext.h \
    src/parallelsurf/BoxFilter.h \
    src/parallelsurf/WaveFilter.h \
    src/parallelsurf/MathStuff.h

FORMS += \
    src/app/mainwindow.ui

RESOURCES += \
    resources/resources.qrc

# OpenCV and ONNX Runtime paths
OPENCV_PATH = "D:/code/DATA/3rdParty/opencv-sdk"
ONNXRUNTIME_PATH = "D:/code/DATA/3rdParty/onnx-sdk"

# Include paths
INCLUDEPATH += \
    $$OPENCV_PATH/include/opencv2 \
    $$OPENCV_PATH/include \
    $$ONNXRUNTIME_PATH/include \
    $$PWD/src \
    $$PWD/src/core \
    $$PWD/src/utils \
    $$PWD/src/parallelsurf

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

# CUDA support (optional, auto-detect)
CUDA_LIB = $$ONNXRUNTIME_PATH/lib/onnxruntime_providers_cuda.lib
exists($$CUDA_LIB) {
    LIBS += $$ONNXRUNTIME_PATH/lib/onnxruntime_providers_shared.lib
    LIBS += $$CUDA_LIB
    DEFINES += ONNX_CUDA_AVAILABLE
    message("CUDA provider found, GPU support enabled")
} else {
    message("CUDA provider not found, GPU support disabled")
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
    exists($$ONNX_LIB/onnxruntime_providers_cuda.dll) {
        QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$${ONNX_LIB}\\onnxruntime_providers_shared.dll\" \"$${DESTDIR_WIN}\\\" &)
        QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$${ONNX_LIB}\\onnxruntime_providers_cuda.dll\" \"$${DESTDIR_WIN}\\\" &)
    }

    # Copy cuDNN DLLs if bundled in project
    CUDNN_SRC = $$PWD/cudnn
    CUDNN_SRC ~= s,/,\\,g
    exists($$CUDNN_SRC) {
        QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$${CUDNN_SRC}\\cudnn*.dll\" \"$${DESTDIR_WIN}\\\" &)
    }

    # Copy CUDA runtime DLLs from CUDA Toolkit (auto-detect via CUDA_PATH)
    CUDA_PATH = $$(CUDA_PATH)
    !isEmpty(CUDA_PATH) {
        CUDA_PATH ~= s,/,\\,g
        CUDA_BIN = $$CUDA_PATH\\bin
        exists($$CUDA_BIN) {
            QMAKE_POST_LINK += $$quote(cmd /c for /f \"tokens=*\" %%i in ('dir /b /a-d \"$$CUDA_BIN\\cudart64_*.dll\"') do copy /y \"$$CUDA_BIN\\%%i\" \"$$DESTDIR_WIN\\\" &)
            QMAKE_POST_LINK += $$quote(cmd /c for /f \"tokens=*\" %%i in ('dir /b /a-d \"$$CUDA_BIN\\cublas64_*.dll\"') do copy /y \"$$CUDA_BIN\\%%i\" \"$$DESTDIR_WIN\\\" &)
            QMAKE_POST_LINK += $$quote(cmd /c for /f \"tokens=*\" %%i in ('dir /b /a-d \"$$CUDA_BIN\\cublasLt64_*.dll\"') do copy /y \"$$CUDA_BIN\\%%i\" \"$$DESTDIR_WIN\\\" &)
        }
        # Copy cuDNN DLLs from NVIDIA cuDNN install path
        CUDNN_INSTALL = C:\\Program Files\\NVIDIA\\CUDNN\\v9.23\\bin\\12.9\\x64
        exists($$CUDNN_INSTALL) {
            CUDNN_INSTALL ~= s,/,\\,g
            QMAKE_POST_LINK += $$quote(cmd /c copy /y \"$$CUDNN_INSTALL\\cudnn*.dll\" \"$$DESTDIR_WIN\\\" &)
        }
    }

    # Copy model directory to build output
    MODEL_SRC = $$PWD/model
    MODEL_SRC ~= s,/,\\,g
    QMAKE_POST_LINK += $$quote(cmd /c xcopy /E /I /Y \"$${MODEL_SRC}\" \"$${DESTDIR_WIN}\\model\\\" &)
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
