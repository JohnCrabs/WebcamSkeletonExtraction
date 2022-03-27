import sys
import os
import tkinter as tk
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem
from PySide2.QtGui import QIcon

_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width


class WidgetTemplate(QWidget):
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

        # -------------------------- #
        # ----- Set PushButton ----- #
        # -------------------------- #
        self.buttonOk = QPushButton('Ok')
        self.buttonOk.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)
        self.buttonCancel = QPushButton('Cancel')
        self.buttonCancel.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)

    def setWidget(self):
        # Set buttons in hbox
        hbox_buttons = QHBoxLayout()  # Create Horizontal Layout
        hbox_buttons.addSpacerItem(QSpacerItem(_INT_MAX_STRETCH, 0))  # Add Spacer
        hbox_buttons.addWidget(self.buttonOk)  # Add the OK Button
        hbox_buttons.addWidget(self.buttonCancel)  # Add the CANCEL Button

        self.vbox_main_layout.addLayout(hbox_buttons)

    # ------------------- #
    # ----- Actions ----- #
    # ------------------- #
    def setActions_(self):
        self.buttonOk.clicked.connect(self.actionButtonOk)
        self.buttonCancel.clicked.connect(self.actionButtonOk)

    def actionButtonOk(self):
        # -----> Write here code for ok <-----
        self.close()  # Close the window

    def actionButtonCancel(self):
        self.close()  # Close the window


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #

def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = WidgetTemplate(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                               winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
