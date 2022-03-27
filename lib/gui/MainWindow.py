import sys
import os
import tkinter as tk
import qdarkstyle
from PySide2.QtCore import (
    Qt
)
from PySide2.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QAction,
    QStatusBar,
)
from PySide2.QtGui import (
    QIcon
)

from lib.gui.CentralWidget import CentralWidget

_STR_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 512  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width
_INT_SPACES = 10  # Set Spaces for Menu Items

# Icon Paths
_ICON_PATH_LOGO_32x32 = _STR_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png'
_ICON_PATH_OPEN_128x128 = _STR_PROJECT_FOLDER + '/icon/open_128x128.png'
_ICON_PATH_SETTINGS_48x48 = _STR_PROJECT_FOLDER + '/icon/settings_48_48.png'
_ICON_PATH_EXIT_APP_48x48 = _STR_PROJECT_FOLDER + '/icon/exit_app_48x48.png'
_ICON_PATH_CALENDAR_48x48 = _STR_PROJECT_FOLDER + '/icon/calendar_48x48.png'


class MainWindowTemplate(QMainWindow):
    def __init__(self, app, w=512, h=512, minW=256, minH=256, winTitle='My Window', iconPath='', parent=None):
        super(MainWindowTemplate, self).__init__(parent)  # super().__init__()
        self.app = app

        # ----------------------------- #
        # ----- Set Other Widgets ----- #
        # ----------------------------- #
        self._centralWidget = CentralWidget(w=512, h=512, minW=64, minH=64, maxW=2048, maxH=2048)
        self._centralWidget.setWidget()
        self.setCentralWidget(self._centralWidget)

        # -------------------------- #
        # ----- Set MainWindow ----- #
        # -------------------------- #
        # self.setStyle_()
        self.setWindowTitle(winTitle)  # Set Window Title
        self.setWindowIcon(QIcon(iconPath))  # Set Window Icon
        self.setGeometry(_INT_SCREEN_WIDTH / 4, _INT_SCREEN_HEIGHT / 4, w, h)  # Set Window Geometry
        self.setMinimumWidth(minW)  # Set Window Minimum Width
        self.setMinimumHeight(minH)  # Set Window Minimum Height

        # ----------------------- #
        # ----- Set MenuBar ----- #
        # ----------------------- #
        self.mainMenu = self.menuBar()  # Set the Menu Bar

        # ***** ACTIONS ***** #
        self.actionExit = QAction(QIcon(_ICON_PATH_EXIT_APP_48x48), 'Exit' + self.setSpaces(_INT_SPACES))  # Exit
        self.actionExit.setShortcut('Ctrl+Q')  # Ctrl + Q
        self.actionExit.setToolTip('Application exit.')  # ToolTip
        # ******************* #

        self.createMenuBar()  # Create all Menu/Sub-Menu/Actions

        # ---------------------------- #
        # ----- Set Main Content ----- #
        # ---------------------------- #

        # ------------------------- #
        # ----- Set StatusBar ----- #
        # ------------------------- #
        self.statusBar = QStatusBar()  # Create Status Bar

        # ------------------------------- #
        # ----- Set Actions Signals ----- #
        # ------------------------------- #
        self.setActions_SignalSlots()  # Contains all the actions

    # -------------------------- #
    # ----- Static Methods ----- #
    # -------------------------- #
    @staticmethod
    def widgetDialogParams(widget: QWidget):
        widget.setWindowModality(Qt.ApplicationModal)
        # widget.setWindowFlags(Qt.WindowStaysOnTopHint)

    @staticmethod
    def setSpaces(number):
        return number * ' '

    # ---------------------------- #
    # ----- Override Methods ----- #
    # ---------------------------- #

    def closeEvent(self, event):
        self.actionExit_func_()

    # ------------------------------ #
    # ----- Non-Static Methods ----- #
    # ------------------------------ #

    def setStyle_(self):
        self.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside2'))

    def createMenuBar(self):
        """
        This function runs all the createMenuBar* menu functions to create each menu.
        By default the template have a Menu File and a Menu Tool
        :return: Nothing
        """
        # Create Menu
        self.createMenuBarFile()  # File
        self.createMenuBarTools()  # Tools

    def createMenuBarFile(self):
        """
        Use this function to create the Menu File.
        Useful Commands:
        menuMain = self.mainMenu.addMenu('NewMenuName')
        menuMain.addAction(self.Action)  # add action created in def __init__()
        menuMain.addSeparator()  # add a separator line between Actions/Menus
        menuNewMenu = menuMain.addMenu("NewMenu")  # create a new Menu inside menuMain
        :return: Nothing
        """
        menuFile = self.mainMenu.addMenu('File')  # File
        # Set Actions and Menus to menuFile
        # Project Actions (New Project, Open, Save, etc)
        menuFile.addSeparator()  # Separator
        # Action Exit
        menuFile.addAction(self.actionExit)

    def createMenuBarTools(self):
        """
        Use this function to create the Menu File.
        Useful Commands:
        menuMain = self.mainMenu.addMenu('NewMenuName')
        menuMain.addAction(self.Action)  # add action created in def __init__()
        menuMain.addSeparator()  # add a separator line between Actions/Menus
        menuNewMenu = menuMain.addMenu("NewMenu")  # create a new Menu inside menuMain
        :return: Nothing
        """
        menuTools = self.mainMenu.addMenu('Tools')  # File
        # Set Actions and Menus to menuTools
        # Project Menu/Actions (Calendar, Machine Learning, )
        menuTools.addSeparator()

    # ------------------- #
    # ----- Actions ----- #
    # ------------------- #
    def setActions_SignalSlots(self):
        """
        A function for storing all the trigger connections
        :return: Nothing
        """
        # ----------------- #
        # Triggered Actions #
        # ----------------- #
        self._centralWidget.setActions_()

        # ********* #
        # Menu FILE #
        # ********* #
        self.actionExit.triggered.connect(self.actionExit_func_)  # actionExit

    # ************ #
    # *** File *** #
    # ************ #
    def actionExit_func_(self):
        self.close()  # close the application
        QApplication.closeAllWindows()


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #


def exec_app(w=512, h=512, minW=256, minH=256, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    mainWin = MainWindowTemplate(myApp, w=w, h=h, minW=minW, minH=minH, winTitle=winTitle,
                                 iconPath=iconPath)  # Create MainWindow
    mainWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


# ****************************************************** #
# ********************   __main__   ******************** #
# ****************************************************** #
if __name__ == "__main__":
    exec_app(w=512, h=512, minW=512, minH=256,
             winTitle='SPACE', iconPath=_ICON_PATH_LOGO_32x32)
