import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt

_translate = QtCore.QCoreApplication.translate


def clear_blocks(self):

	if self.verticalLayout_5.count() > 4:
		self.verticalLayout_5.removeItem(self.verticalLayoutWrapper)
		# self.verticalLayout_5.removeItem(self.verticalLayoutRecievedScore)
		# self.verticalLayout_5.removeItem(self.label_result)
		self.verticalLayout_5.removeItem(self.horizontalLayoutAnwers)
		self.verticalLayout_5.removeItem(self.horizontalLayoutExampleFeatures)
		self.verticalLayout_5.removeItem(self.verticalLayout_2)
		self.verticalLayout_5.removeItem(self.horizontalLayout)


def remove_spacer(self):
	self.verticalLayout_5.removeItem(self.spacerItem2)


def add_spacer(self):
	self.verticalLayout_5.addItem(self.spacerItem2)


def display_selected_image(self):
	remove_spacer(self)
	self.verticalLayout_2 = QtWidgets.QVBoxLayout()
	self.verticalLayout_2.setObjectName("verticalLayout_2")
	self.label_selected_image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_selected_image.setObjectName("label_selected_image")
	self.label_selected_image.setText(_translate("MainWindow", "Selected Image"))
	self.verticalLayout_2.addWidget(self.label_selected_image)
	self.selected_image_block = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.selected_image_block.setMinimumSize(QtCore.QSize(150, 150))
	self.selected_image_block.setMaximumSize(QtCore.QSize(150, 150))
	self.selected_image_block.setText(_translate("MainWindow", "TextLabel"))
	self.verticalLayout_2.addWidget(self.selected_image_block)
	self.verticalLayout_5.addLayout(self.verticalLayout_2)
	self.horizontalLayout = QtWidgets.QHBoxLayout()
	self.horizontalLayout.setObjectName("horizontalLayout")
	self.buttonRun.show()
	self.buttonRun.setMinimumSize(QtCore.QSize(100, 0))
	self.buttonRun.setMaximumSize(QtCore.QSize(16777215, 20))
	self.buttonRun.setStyleSheet("background-color: rgba(0, 130, 0, 145);\n"
								 "border-radius: 4px;\n"
								 "")
	self.buttonRun.setObjectName("buttonRun")
	self.buttonRun.setText(_translate("MainWindow", "Run"))
	self.horizontalLayout.addWidget(self.buttonRun)
	spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
	self.horizontalLayout.addItem(spacerItem2)
	self.verticalLayout_5.addLayout(self.horizontalLayout)

	add_spacer(self)

def display_example_features(self, methods):
	self.horizontalLayoutExampleFeatures = QtWidgets.QHBoxLayout()
	self.horizontalLayoutExampleFeatures.setObjectName("horizontalLayout_3")

	for method in methods:
		verticalLayout = QtWidgets.QVBoxLayout()

		label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		label.setText(_translate("MainWindow", method[0]))
		verticalLayout.addWidget(label, alignment=Qt.AlignCenter)

		feature_plot = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		feature_plot.setMinimumSize(QtCore.QSize(150, 150))

		if method[0] != 'WaveletTransform':
			pixmap = QtGui.QPixmap(method[1]).scaled(330, 330, QtCore.Qt.KeepAspectRatio)
		else:
			pixmap = QtGui.QPixmap(method[1])


		feature_plot.setPixmap(pixmap)

		verticalLayout.addWidget(feature_plot)
		self.horizontalLayoutExampleFeatures.addLayout(verticalLayout)

	self.verticalLayout_5.addLayout(self.horizontalLayoutExampleFeatures)

def create_path_to_image(image):
	min_val, max_val = image.min(), image.max()
	image = 255.0 * (image - min_val) / (max_val - min_val)
	image = image.astype(np.uint8)

	path = 'answer.png'
	cv2.imwrite(path, image)

	return path

def display_answer_image(self, answers, authors):
	self.horizontalLayoutAnwers = QtWidgets.QHBoxLayout()
	for answer in answers:

		verticalLayout = QtWidgets.QVBoxLayout()
		label_answer = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		label_answer.setStyleSheet("margin-top: 20px;")
		label_answer.setText(_translate("MainWindow", "Answer"))

		verticalLayout.addWidget(label_answer)
		image_answer = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		image_answer.setMinimumSize(QtCore.QSize(50, 50))
		label_answer.setText(_translate("MainWindow", authors[answer[0] - 1]))

		verticalLayout.addWidget(image_answer)
		self.horizontalLayoutAnwers.addLayout(verticalLayout)

	self.horizontalLayoutAnwers.addLayout(verticalLayout)
	self.verticalLayout_5.addLayout(self.horizontalLayoutAnwers)


def display_received_score(self, methods):
	self.verticalLayoutRecievedScore = QtWidgets.QVBoxLayout()

	for idx, method in enumerate(methods):
		horizontalLayout = QtWidgets.QHBoxLayout()
		self.score_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		self.score_label.setMaximumSize(QtCore.QSize(16777215, 30))
		self.score_label.setObjectName("score_label")
		self.score_label.setText(_translate("MainWindow", "Score of method " + method + ' for ' + methods[method]['name'] + "=" + str(methods[method]['value'])))

		horizontalLayout.addWidget(self.score_label)
		self.score_value = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		self.score_value.setMaximumSize(QtCore.QSize(16777215, 30))
		self.score_value.setStyleSheet("")
		self.score_value.setObjectName("score_value")
		self.score_value.setText(_translate("MainWindow", str(methods[method]['score'])))
		horizontalLayout.addWidget(self.score_value)
		self.verticalLayoutRecievedScore.addLayout(horizontalLayout)

	self.verticalLayout_5.addLayout(self.verticalLayoutRecievedScore)
	self.label_result = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_result.setStyleSheet("margin-top: 25px;")
	self.label_result.setObjectName("label_result")
	self.label_result.setText(_translate("MainWindow", "Results"))
	self.verticalLayout_5.addWidget(self.label_result)


def build_line_plot(data, name):
	plt.figure(figsize=(20, 10), dpi=80)
	ax = plt.gca()
	plt.xticks(fontsize=50)
	plt.yticks(fontsize=50)
	ax.grid(linewidth=5)
	plt.plot(*zip(*data), linewidth=6)
	save_path = name + '.png'
	plt.savefig(save_path)

	return save_path


def display_scores(self, scores, titleBlock = 'Точность распознавания'):
	self.verticalLayoutWrapper = QtWidgets.QVBoxLayout()
	title = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	title.setMaximumSize(QtCore.QSize(16777215, 30))
	title.setText(_translate("MainWindow", titleBlock))
	self.verticalLayoutWrapper.addWidget(title)

	horizontalLayout = QtWidgets.QHBoxLayout()

	for idx in range(3):
		verticalLayout = QtWidgets.QVBoxLayout()

		label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
		verticalLayout.addWidget(label)

		chart = QtWidgets.QLabel(self.scrollAreaWidgetContents)

		path_to_line_chart = build_line_plot(scores[idx], 'scores_folds')

		pixmap = QtGui.QPixmap(path_to_line_chart).scaled(310, 300, QtCore.Qt.KeepAspectRatio)
		chart.setPixmap(pixmap)
		verticalLayout.addWidget(chart)
		horizontalLayout.addLayout(verticalLayout)

	self.verticalLayoutWrapper.addLayout(horizontalLayout)
	self.verticalLayout_5.addLayout(self.verticalLayoutWrapper)