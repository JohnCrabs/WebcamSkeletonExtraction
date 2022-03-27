import sys
import os
import tkinter as tk
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem
from PySide2.QtGui import QIcon

from lib.gui.OpenGL_Widget import OpenGLWidget

_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width


class CentralWidget(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512,
                 winTitle='My Window', iconPath=''):
        super().__init__()
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

        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # --------------------------- #
        # ----- ImageViewerArea ----- #
        # --------------------------- #
        self.imageViewer = OpenGLWidget(w=256, h=256,
                                        minW=16, minH=16,
                                        maxW=2048, maxH=2048)

        # -------------------------- #
        # ----- Set PushButton ----- #
        # -------------------------- #
        self.buttonOpenCamera = QPushButton('Open Camera')
        self.buttonOpenCamera.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)
        self.buttonCloseCamera = QPushButton('Close Camera')
        self.buttonCloseCamera.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)

    def setWidget(self):
        # Set buttons in hbox
        hbox_buttons = QHBoxLayout()  # Create Horizontal Layout
        # hbox_buttons.addSpacerItem(QSpacerItem(_INT_MAX_STRETCH, 0))  # Add Spacer
        hbox_buttons.addWidget(self.buttonOpenCamera)  # Add the OK Button
        hbox_buttons.addWidget(self.buttonCloseCamera)  # Add the CANCEL Button

        vbox_final = QVBoxLayout()
        vbox_final.addWidget(self.imageViewer)
        vbox_final.addLayout(hbox_buttons)

        self.vbox_main_layout.addLayout(vbox_final)

    # ------------------- #
    # ----- Actions ----- #
    # ------------------- #
    def setActions_(self):
        self.buttonOpenCamera.clicked.connect(self.actionOpenCamera)
        self.buttonCloseCamera.clicked.connect(self.actionCloseCamera)

    def actionOpenCamera(self):
        self.imageViewer.setIsInputFromCamera(True)

    def actionCloseCamera(self):
        self.imageViewer.setIsInputFromCamera(False)


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #

def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = CentralWidget(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                               winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.setWidget()
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='CentralWidget', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
