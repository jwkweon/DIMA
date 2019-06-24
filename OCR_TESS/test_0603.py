#for UI

import sys
import os

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QAction, qApp, QLabel, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from PIL import Image

class MyWidget(QWidget):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.initUI()

    def initUI(self):

        label = QLabel(self)
        pixmap = QPixmap(self.filename)
        pixmap = pixmap.scaledToWidth(800)
        label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())



class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        exitAction = QAction(QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        openFile = QAction('Open file', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open file')
        openFile.triggered.connect(self.file_open)

        runFile = QAction('Run file', self)
        runFile.setShortcut('Ctrl+R')
        runFile.setStatusTip('Run file')
        runFile.triggered.connect(self.file_run)

        self.statusBar().showMessage('Open your file!', 1000)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&Open file')
        Run = menubar.addMenu('&Run')
        Quit = menubar.addMenu('&Quit')

        fileMenu.addAction(openFile)
        Run.addAction(runFile)
        Quit.addAction(exitAction)

        self.setWindowTitle('OCR-FUNNEL')
        self.setWindowIcon(QIcon('dima.png'))
        self.setGeometry(500, 300, 900, 1300)

        self.show()

    def file_open(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        self.f_path = name[0]

        self.statusBar().showMessage(name[0])
        wg = MyWidget(name[0])
        self.setCentralWidget(wg)

    def file_run(self):
        print(self.f_path.split('/')[-1])
        len_filename = len(self.f_path.split('/')[-1])
        os.system("python test_0527.py --filename="+self.f_path.split('/')[-1])

        wg = MyWidget(self.f_path[:-len_filename] + 'res-' + self.f_path.split('/')[-1])
        self.setCentralWidget(wg)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()

    sys.exit(app.exec_())
