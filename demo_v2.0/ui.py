# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(952, 552)
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 30, 851, 491))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 60, 293, 18))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.radioButton_3 = QtWidgets.QRadioButton(self.layoutWidget1)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_2.addWidget(self.radioButton_3)
        self.radioButton_5 = QtWidgets.QRadioButton(self.layoutWidget1)
        self.radioButton_5.setObjectName("radioButton_5")
        self.horizontalLayout_2.addWidget(self.radioButton_5)
        self.radioButton_4 = QtWidgets.QRadioButton(self.layoutWidget1)
        self.radioButton_4.setObjectName("radioButton_4")
        self.horizontalLayout_2.addWidget(self.radioButton_4)
        self.layoutWidget2 = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 30, 192, 18))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.radioButton = QtWidgets.QRadioButton(self.layoutWidget2)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.layoutWidget2)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout.addWidget(self.radioButton_2)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_3 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.layoutWidget3 = QtWidgets.QWidget(self.groupBox_3)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 30, 282, 54))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget3)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget3)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit_model = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_model.setObjectName("lineEdit_model")
        self.gridLayout.addWidget(self.lineEdit_model, 0, 1, 1, 1)
        self.button_model = QtWidgets.QPushButton(self.layoutWidget3)
        self.button_model.setObjectName("button_model")
        self.gridLayout.addWidget(self.button_model, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_image = QtWidgets.QLineEdit(self.layoutWidget3)
        self.lineEdit_image.setObjectName("lineEdit_image")
        self.gridLayout.addWidget(self.lineEdit_image, 1, 1, 1, 1)
        self.button_image = QtWidgets.QPushButton(self.layoutWidget3)
        self.button_image.setObjectName("button_image")
        self.gridLayout.addWidget(self.button_image, 1, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.layoutWidget4 = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget4.setGeometry(QtCore.QRect(10, 30, 302, 22))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.spin_conf = QtWidgets.QDoubleSpinBox(self.layoutWidget4)
        self.spin_conf.setObjectName("spin_conf")
        self.horizontalLayout_3.addWidget(self.spin_conf)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_3.addWidget(self.label_6)
        self.spin_iou = QtWidgets.QDoubleSpinBox(self.layoutWidget4)
        self.spin_iou.setObjectName("spin_iou")
        self.horizontalLayout_3.addWidget(self.spin_iou)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_3.addWidget(self.label_7)
        self.spin_other = QtWidgets.QDoubleSpinBox(self.layoutWidget4)
        self.spin_other.setObjectName("spin_other")
        self.horizontalLayout_3.addWidget(self.spin_other)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.lineEdit_reserve = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_reserve.setGeometry(QtCore.QRect(10, 30, 113, 20))
        self.lineEdit_reserve.setObjectName("lineEdit_reserve")
        self.verticalLayout_2.addWidget(self.groupBox_4)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.button_save = QtWidgets.QPushButton(self.layoutWidget)
        self.button_save.setObjectName("button_save")
        self.gridLayout_2.addWidget(self.button_save, 0, 0, 1, 1)
        self.button_run = QtWidgets.QPushButton(self.layoutWidget)
        self.button_run.setObjectName("button_run")
        self.gridLayout_2.addWidget(self.button_run, 0, 1, 1, 1)
        self.button_reserve1 = QtWidgets.QPushButton(self.layoutWidget)
        self.button_reserve1.setObjectName("button_reserve1")
        self.gridLayout_2.addWidget(self.button_reserve1, 0, 2, 1, 1)
        self.button_reserve2 = QtWidgets.QPushButton(self.layoutWidget)
        self.button_reserve2.setObjectName("button_reserve2")
        self.gridLayout_2.addWidget(self.button_reserve2, 0, 3, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.image_label = QtWidgets.QLabel(self.layoutWidget)
        self.image_label.setText("")
        self.image_label.setPixmap(QtGui.QPixmap("images/HFUT.png"))
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)
        self.output_text = QtWidgets.QTextBrowser(self.layoutWidget)
        self.output_text.setObjectName("output_text")
        self.verticalLayout.addWidget(self.output_text)
        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "模型/文件类型选择"))
        self.label_4.setText(_translate("Form", "文件类型："))
        self.radioButton_3.setText(_translate("Form", "image "))
        self.radioButton_5.setText(_translate("Form", "video "))
        self.radioButton_4.setText(_translate("Form", "image folder"))
        self.label_3.setText(_translate("Form", "模型类型："))
        self.radioButton.setText(_translate("Form", "Yolov5"))
        self.radioButton_2.setText(_translate("Form", "Yolov8"))
        self.groupBox_3.setTitle(_translate("Form", "模型/文件输入"))
        self.label.setText(_translate("Form", "模型位置："))
        self.button_model.setText(_translate("Form", "浏览"))
        self.label_2.setText(_translate("Form", "文件位置："))
        self.button_image.setText(_translate("Form", "浏览"))
        self.groupBox_2.setTitle(_translate("Form", "参数设置"))
        self.label_5.setText(_translate("Form", "Conf："))
        self.label_6.setText(_translate("Form", "IoU："))
        self.label_7.setText(_translate("Form", "other："))
        self.groupBox_4.setTitle(_translate("Form", "保留配置"))
        self.lineEdit_reserve.setText(_translate("Form", "保留配置"))
        self.button_save.setText(_translate("Form", "保存配置"))
        self.button_run.setText(_translate("Form", "开始运行"))
        self.button_reserve1.setText(_translate("Form", "保留1"))
        self.button_reserve2.setText(_translate("Form", "保留2"))
from PyQt5.QtWidgets import QApplication,QMainWindow
import sys
if __name__ == '__main__':
    # 只有直接运行这个脚本，才会往下执行
    # 别的脚本文件执行，不会调用这个条件句

    # 实例化，传参
    app = QApplication(sys.argv)

    # 创建对象
    mainWindow = QMainWindow()

    # 创建ui，引用demo1文件中的Ui_MainWindow类
    ui = Ui_Form()
    # 调用Ui_MainWindow类的setupUi，创建初始组件
    ui.setupUi(mainWindow)
    # 创建窗口
    mainWindow.show()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())