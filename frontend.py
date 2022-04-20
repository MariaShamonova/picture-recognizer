from PyQt5 import QtWidgets, QtGui, QtCore

_translate = QtCore.QCoreApplication.translate


def remove_spacer(self, order=3):
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

def display_example_features(self):
	self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_3.setObjectName("horizontalLayout_3")
	self.verticalLayout_7 = QtWidgets.QVBoxLayout()
	self.verticalLayout_7.setObjectName("verticalLayout_7")
	self.label_method_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_method_1.setObjectName("label_method_1")
	self.verticalLayout_7.addWidget(self.label_method_1)
	self.feature_method_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.feature_method_1.setMinimumSize(QtCore.QSize(150, 150))
	self.feature_method_1.setObjectName("feature_method_1")
	self.verticalLayout_7.addWidget(self.feature_method_1)
	self.horizontalLayout_3.addLayout(self.verticalLayout_7)
	self.verticalLayout_13 = QtWidgets.QVBoxLayout()
	self.verticalLayout_13.setObjectName("verticalLayout_13")
	self.label_method_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_method_2.setObjectName("label_method_2")
	self.verticalLayout_13.addWidget(self.label_method_2)
	self.feature_method_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.feature_method_2.setMinimumSize(QtCore.QSize(150, 150))
	self.feature_method_2.setObjectName("feature_method_2")
	self.verticalLayout_13.addWidget(self.feature_method_2)
	self.horizontalLayout_3.addLayout(self.verticalLayout_13)
	self.verticalLayout_12 = QtWidgets.QVBoxLayout()
	self.verticalLayout_12.setObjectName("verticalLayout_12")
	self.label_method_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_method_3.setObjectName("label_method_3")
	self.verticalLayout_12.addWidget(self.label_method_3)
	self.feature_method_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.feature_method_3.setMinimumSize(QtCore.QSize(150, 150))
	self.feature_method_3.setObjectName("feature_method_3")
	self.verticalLayout_12.addWidget(self.feature_method_3)
	self.horizontalLayout_3.addLayout(self.verticalLayout_12)
	self.verticalLayout_5.addLayout(self.horizontalLayout_3)


def display_answer_image(self):
	self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_7.setObjectName("horizontalLayout_7")
	self.verticalLayout_17 = QtWidgets.QVBoxLayout()
	self.verticalLayout_17.setObjectName("verticalLayout_17")
	self.label_selected_method_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_selected_method_1.setObjectName("label_selected_method_1")
	self.label_selected_method_1.setText(_translate("MainWindow", "Selected image"))
	self.verticalLayout_17.addWidget(self.label_selected_method_1)
	self.selected_image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.selected_image.setMinimumSize(QtCore.QSize(150, 150))
	self.selected_image.setObjectName("selected_image")
	self.verticalLayout_17.addWidget(self.selected_image)
	self.horizontalLayout_7.addLayout(self.verticalLayout_17)
	self.verticalLayout_18 = QtWidgets.QVBoxLayout()
	self.verticalLayout_18.setObjectName("verticalLayout_18")
	self.label_answer_algorithm = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_answer_algorithm.setObjectName("label_answer_algorithm")
	self.label_answer_algorithm.setText(_translate("MainWindow", "Answer"))
	self.verticalLayout_18.addWidget(self.label_answer_algorithm)
	self.answer_algorithm = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.answer_algorithm.setMinimumSize(QtCore.QSize(150, 150))
	self.answer_algorithm.setObjectName("answer_algorithm")
	self.verticalLayout_18.addWidget(self.answer_algorithm)
	self.horizontalLayout_7.addLayout(self.verticalLayout_18)
	spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
	self.horizontalLayout_7.addItem(spacerItem3)
	self.verticalLayout_5.addLayout(self.horizontalLayout_7)


def display_received_score(self):
	self.verticalLayout = QtWidgets.QVBoxLayout()
	self.verticalLayout.setObjectName("verticalLayout")
	self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_2.setObjectName("horizontalLayout_2")
	self.score_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.score_label.setMaximumSize(QtCore.QSize(16777215, 30))
	self.score_label.setStyleSheet("")
	self.score_label.setObjectName("score_label")
	self.score_label.setText(_translate("MainWindow", "Score of parallel system"))

	self.horizontalLayout_2.addWidget(self.score_label)
	self.score_value = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.score_value.setMaximumSize(QtCore.QSize(16777215, 30))
	self.score_value.setStyleSheet("")
	self.score_value.setObjectName("score_value")
	self.score_value.setText(_translate("MainWindow", "Score"))
	self.horizontalLayout_2.addWidget(self.score_value)
	self.verticalLayout.addLayout(self.horizontalLayout_2)
	self.verticalLayout_5.addLayout(self.verticalLayout)
	self.label_result = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_result.setStyleSheet("margin-top: 25px;")
	self.label_result.setObjectName("label_result")
	self.label_result.setText(_translate("MainWindow", "Results"))
	self.verticalLayout_5.addWidget(self.label_result)


def display_scores(self):
	self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_6.setObjectName("horizontalLayout_6")
	self.verticalLayout_14 = QtWidgets.QVBoxLayout()
	self.verticalLayout_14.setObjectName("verticalLayout_14")
	self.label_result_method_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_result_method_1.setObjectName("label_result_method_1")
	self.verticalLayout_14.addWidget(self.label_result_method_1)
	self.parameters_method_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.parameters_method_1.setMinimumSize(QtCore.QSize(150, 150))
	self.parameters_method_1.setObjectName("parameters_method_1")
	self.verticalLayout_14.addWidget(self.parameters_method_1)
	self.folds_method_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.folds_method_1.setMinimumSize(QtCore.QSize(150, 150))
	self.folds_method_1.setObjectName("folds_method_1")
	self.verticalLayout_14.addWidget(self.folds_method_1)
	self.horizontalLayout_6.addLayout(self.verticalLayout_14)
	self.verticalLayout_15 = QtWidgets.QVBoxLayout()
	self.verticalLayout_15.setObjectName("verticalLayout_15")
	self.label_result_method_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_result_method_2.setObjectName("label_result_method_2")
	self.verticalLayout_15.addWidget(self.label_result_method_2)
	self.parameters_method_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.parameters_method_2.setMinimumSize(QtCore.QSize(150, 150))
	self.parameters_method_2.setObjectName("parameters_method_2")
	self.verticalLayout_15.addWidget(self.parameters_method_2)
	self.folds_method_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.folds_method_2.setMinimumSize(QtCore.QSize(150, 150))
	self.folds_method_2.setObjectName("folds_method_2")
	self.verticalLayout_15.addWidget(self.folds_method_2)
	self.horizontalLayout_6.addLayout(self.verticalLayout_15)
	self.verticalLayout_16 = QtWidgets.QVBoxLayout()
	self.verticalLayout_16.setObjectName("verticalLayout_16")
	self.label_result_method_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_result_method_3.setObjectName("label_result_method_3")
	self.verticalLayout_16.addWidget(self.label_result_method_3)
	self.parameters_method_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.parameters_method_3.setMinimumSize(QtCore.QSize(150, 150))
	self.parameters_method_3.setObjectName("parameters_method_3")
	self.verticalLayout_16.addWidget(self.parameters_method_3)
	self.folds_method_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.folds_method_3.setMinimumSize(QtCore.QSize(150, 150))
	self.folds_method_3.setObjectName("folds_method_3")
	self.verticalLayout_16.addWidget(self.folds_method_3)
	self.horizontalLayout_6.addLayout(self.verticalLayout_16)
	self.verticalLayout_5.addLayout(self.horizontalLayout_6)
