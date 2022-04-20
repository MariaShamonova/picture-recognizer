import pathlib
from PyQt5 import QtWidgets, QtGui, QtCore

from feature_getters import FeatureGetter
from design import Ui_MainWindow
from faces_repository import *
from controller import FaceRecognizer
path = str(pathlib.Path(__file__).parent.resolve())

from frontend import *

class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.buttonRun = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.buttonRun.hide()

        self.connect_functions()

    def connect_functions(self):
        self.buttonSelectImage.clicked.connect(self.select_image)
        self.buttonRun.clicked.connect(self.start_computing)
        print('connect function')

    def select_image(self):
        fname = self.get_path_image_from_folder()

        display_selected_image(self)
        pixmap = QtGui.QPixmap(fname).scaled(176, 179, QtCore.Qt.KeepAspectRatio)
        self.selected_image_block.setPixmap(pixmap)

    def get_path_image_from_folder(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image',
                                                      str(path) + '/pictures')
        return fname[0]

    def start_computing(self):
        display_example_features(self)
        display_answer_image(self)
        display_received_score(self)
        display_scores(self)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    # window.parallel_computing()
    sys.exit(app.exec_())



if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
