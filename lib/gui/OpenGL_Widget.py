import sys
import os
import tkinter as tk
import cv2 as cv2
import time

from OpenGL.GL import *

from PySide2.QtCore import (
    QBasicTimer
)

from PySide2.QtOpenGL import (
    QGLWidget
)

from PySide2.QtWidgets import (
    QApplication,
)

from PySide2.QtGui import (
    QIcon,
)

import lib.core.PoseModule as pm

_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width


class OpenGLWidget(QGLWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512,
                 winTitle='My Window', iconPath=''):
        super().__init__()
        self._timer = QBasicTimer()  # creating timer
        self._timer.start(1000 / 60, self)  # setting up timer ticks to 60 fps
        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.setWindowTitle(winTitle)  # Set Window Title
        self.setWindowIcon(QIcon(iconPath))  # Set Window Icon
        self.setGeometry(_INT_SCREEN_WIDTH / 4, _INT_SCREEN_HEIGHT / 4, w, h)  # Set Window Geometry
        self.setMinimumWidth(minW)  # Set Window Minimum Width
        self.setMinimumHeight(minH)  # Set Window Minimum Height
        if maxW is not None:
            self.setMaximumWidth(maxW)  # Set Window Maximum Width
        if maxH is not None:
            self.setMaximumHeight(maxH)  # Set Window Maximum Width

        self.imgToView = None
        self.isInputFromCamera = False
        self.videoCapture = None

        # --------------------------- #
        # ----- Pose Estimation ----- #
        # --------------------------- #
        self.detector = pm.PoseDetector()
        self.pTime = 0

    def initializeGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        if self.isInputFromCamera:
            ret, frame = self.videoCapture.read()

            frame = self.detector.MediaPipe_findPose(frame)
            # frame = self.detector.OpenPose_findPose(frame)
            if frame is not None and frame.any():
                _ = self.detector.MediaPipe_findPosition(frame, draw=True)
                cTime = time.time()
                fps = 1 / (cTime - self.pTime)
                self.pTime = cTime

                frame = cv2.flip(frame, 1)
                cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 0), 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = cv2.flip(frame, 0)

            self.imgToView = frame

        if self.imgToView is not None:
            # print(self.imgToView.shape)
            glColor3f(0.0, 0.0, 0.0)
            w = self.imgToView.shape[1]
            h = self.imgToView.shape[0]
            original_ratio = w / h
            designer_ratio = self.width() / self.height()
            if original_ratio > designer_ratio:
                designer_height = self.width() / original_ratio
                scale = designer_height / h
            elif original_ratio < designer_ratio:
                designer_width = self.height() * original_ratio
                scale = designer_width / w
            else:
                scale = 1

            pos_w = scale * w
            pos_w = 1.0 - ((self.width() - pos_w) / self.width())
            glRasterPos2f(-pos_w, -1.0)
            pos_h = scale * h
            pos_h = 1.0 - ((self.height() - pos_h) / self.height())
            glRasterPos2f(-pos_w, -pos_h)

            glPixelZoom(scale, scale)
            glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, self.imgToView)
            # cv2.imshow('hello', self.imgToView)
        else:
            glColor4f(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glLoadIdentity()
        # Make the display area proportional to the size of the view
        glOrtho(-w / self.width(), w / self.width(), -h / self.height(), h / self.height(), -1.0, 1.0)

    def timerEvent(self, QTimerEvent):
        self.update()  # refreshing the widget

    def setImg(self, imgPath):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        # img = cv2.rotate(img, cv2.cv2.ROTATE_180)

        if img is not None and img.any():
            _ = self.detector.MediaPipe_findPosition(img, draw=True)
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

        self.imgToView = img

    def clearImg(self):
        self.imgToView = None

    def setIsInputFromCamera(self, state: bool):
        self.isInputFromCamera = state
        if state:
            self.videoCapture = cv2.VideoCapture(0)
            self.videoCapture.set(cv2.CAP_PROP_FPS, 24)
            # self.videoCapture.set(3, 800)
            # self.videoCapture.set(4, 800)
        else:
            if self.videoCapture is not None:
                self.videoCapture.release()
                self.videoCapture = None
                self.imgToView = None
            self.pTime = 0

    def stopImageFromCameraAndKeepImage(self):
        img = self.imgToView
        self.setIsInputFromCamera(False)
        self.imgToView = img

    def getImgToView(self):
        return self.imgToView


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #

def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = OpenGLWidget(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                               winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='OpenGLWidget', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
