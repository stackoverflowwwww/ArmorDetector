TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        ArmorDetector.cpp \
        main.cpp
# OpenCV
INCLUDEPATH += /usr/local/include/opencv2
LIBS += $(shell pkg-config opencv --libs)

HEADERS += \
    ArmorDetector.hpp
