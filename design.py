# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'recognizer.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 0, 781, 541))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 779, 539))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_count_faces_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_count_faces_2.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_count_faces_2.setStyleSheet("font-size: 18px;\n"
"font-weight: 500;")
        self.label_count_faces_2.setObjectName("label_count_faces_2")
        self.horizontalLayout_5.addWidget(self.label_count_faces_2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_count_faces = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_count_faces.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_count_faces.setObjectName("label_count_faces")
        self.horizontalLayout_4.addWidget(self.label_count_faces)
        self.buttonSelectImage = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.buttonSelectImage.setEnabled(True)
        self.buttonSelectImage.setMinimumSize(QtCore.QSize(120, 20))
        self.buttonSelectImage.setStyleSheet("color: rgb(0, 0, 0);\n"
"margin-left: 15px;\n"
"border-radius: 4px;\n"
"background-color: rgb(178, 178, 178);")
        self.buttonSelectImage.setCheckable(False)
        self.buttonSelectImage.setAutoDefault(True)
        self.buttonSelectImage.setDefault(False)
        self.buttonSelectImage.setObjectName("buttonSelectImage")
        self.horizontalLayout_4.addWidget(self.buttonSelectImage)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.line = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_5.addWidget(self.line)

        self.spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(self.spacerItem2)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_count_faces_2.setText(_translate("MainWindow", "Picture recognizer "))
        self.label_count_faces.setText(_translate("MainWindow", "Select Image"))
        self.buttonSelectImage.setText(_translate("MainWindow", "Select"))