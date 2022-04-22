import pathlib
import random
from multiprocessing import Pool
from functools import reduce
from PyQt5 import QtWidgets, QtGui, QtCore

from feature_getters import FeatureGetter, Histogram, DCT, DFT, Random, WaveletTransform, SURF, PCAanalisys, Haralick
from design import Ui_MainWindow
from faces_repository import *
from controller import FaceRecognizer
path = str(pathlib.Path(__file__).parent.resolve())

from frontend import *

class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.selected_picture = ''
        self.url_selected_pictures = ''

        self.methods = {
            'WaveletTransform': {'name': 'step', 'value': 5},
            'PCAanalisys': {'name': 'P/Q', 'value': 20},
            'Haralick': {'name': 'P/D', 'value': 10}
        }

        self.authors = ['Pablo Picasso', 'Sandro Botticelli', 'Vincent van Gogh', 'Ilya Repin',
                        'Rembrandt van Rijn', 'Rubens Piter Paul', 'Ivan Aivazovskiy',
                        'Pierre-Auguste Renoir', 'Karl Brullov', 'Francois Boucher' ]

        self.buttonRun = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.buttonRun.hide()

        self.connect_functions()

    def get_path_plot_picture_for_selected_image(self, classifier):
        self.selected_picture = cv2.cvtColor(cv2.imread(self.url_selected_pictures), cv2.COLOR_BGR2GRAY) / 255
        path_to_image_plot = classifier.plot(self.selected_picture )

        return path_to_image_plot

    def get_features_for_selected_image(self, classifier):
        self.selected_picture = cv2.cvtColor(cv2.imread(self.url_selected_pictures), cv2.COLOR_BGR2GRAY) / 255
        picture_features = classifier.get_feature(self.selected_picture )

        return picture_features

    def start_computing(self):
        num_faces_for_train = 7

        data_pictures, data_target = get_pictures_data(10, 10)
        x_train, y_train, x_test, y_test = split_data(data_pictures, data_target, num_faces_for_train)

        # Обучаем модель на 3х разных методах и методом голосования решаем какой из 3-х ответов наиболее частый и берем это за ответ алгоритма, считаем скор.
        scores_futures = []
        methods_features=[]
        answers=[]
        scores_params = []
        scores_folds = []
        for idx, method in enumerate(self.methods):
            classifier = eval(method)()
            classifier.set_param(self.methods[method]['value'])

            face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                             classifier=classifier)

            face_recognizer.teach_recognizer()
            methods_features.append((method, self.get_path_plot_picture_for_selected_image(classifier)))
            scores_futures.append(face_recognizer.get_recognize_score)
            answer_id, answer_image = face_recognizer.recognize_face(self.get_features_for_selected_image(classifier))
            answers.append((answer_id, answer_image))

            # Подбор наилучшего параметра метода
            scores_params.append(face_recognizer.get_list_params())
            best_param, max_score = max(scores_params[idx], key=lambda x: x[1])
            scores_folds.append(face_recognizer.cross_validation(data_pictures, data_target, best_param))


        # Здесь должен быть кусок отвечающий за вывод признаков для загруженной картинки
        display_example_features(self, methods_features)

        # Находим минимальное рассстояние для загруженного изображения и вычисляем ответ.
        display_answer_image(self, answers,  self.authors)

        display_scores(self, scores_params,  'Подбор параметра')
        display_scores(self, scores_folds, 'Кросс-валидация')
        # График кросс-валидации для разных фолдов на лучшем параметре

    @staticmethod
    def parallelize(n_workers, functions):
        with Pool(n_workers) as pool:
            futures = [pool.apply_async(t) for t in functions]
            results = [fut.get() for fut in futures]
        return results

    def connect_functions(self):
        self.buttonSelectImage.clicked.connect(self.select_image)
        self.buttonRun.clicked.connect(self.start_computing)


    def select_image(self):
        fname = self.get_path_image_from_folder()
        self.url_selected_pictures = fname

        display_selected_image(self)

        pixmap = QtGui.QPixmap(fname).scaled(176, 179, QtCore.Qt.KeepAspectRatio)
        self.selected_image_block.setPixmap(pixmap)

    def get_path_image_from_folder(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image',
                                                      str(path) + '/pictures')
        return fname[0]


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    # window.parallel_computing()
    sys.exit(app.exec_())



if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
