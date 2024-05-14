import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from deploy import *
import cv2 as cv
envpath = '/home/zency/anaconda3/envs/qt/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

class Zency(QWidget):
    def __init__(self):
        super(Zency, self).__init__()
        self.model_name = 'yolov5m.onnx'
        self.image_name = 'images/bus.jpg'
        self.Conf = 0.25
        self.IoU = 0.45
        self.other = 0
        self.initUI()
    def initUI(self):
        #self.resize(800,450)
        self.setWindowTitle('Zency Demo')

        # 创建布局
        main_layout = QHBoxLayout()
        layout_set = QVBoxLayout()
        layout_show = QVBoxLayout()
        layout1 = QGridLayout()
        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()
        layout4 = QHBoxLayout()

        # 放置控件
        group_box1 = QGroupBox('模型/文件输入')
        group_box1.setFixedSize(300,100)
        group_box2 = QGroupBox('参数设置')
        group_box2.setFixedSize(300, 100)
        group_box3 = QGroupBox('待完善设置')
        group_box3.setFixedSize(300, 100)

        self.label1 = QLabel("模型位置：")
        self.label2 = QLabel("图像位置：")
        self.label3 = QLabel("Conf:")
        self.label4 = QLabel("IoU: ")
        self.label5 = QLabel("other:")

        #self.label.setText("test")
        # self.label.move(20, 20)
        self.image_label_org = QLabel("图片显示")
        self.image_label_org.setFixedSize(640,640)
        self.image_label_org.setPixmap(QPixmap("controls/images/HFUT.png"))

        self.lineEdit1 = QLineEdit("yolov5m.onnx")
        self.lineEdit2 = QLineEdit("images/bus.jpg")
        self.lineEdit3 = QLineEdit("待完成设置")
        # self.lineEdit.inputRejected()
        # self.lineEdit.move(40, 20)

        self.output_text = QTextBrowser(None)
        self.output_text.setPlainText("输出窗口")

        self.spin1 = QDoubleSpinBox(None)
        self.spin1 .setValue(0.50)
        self.spin1.setRange(0.00,1.00)
        self.spin1.setSingleStep(0.01)
        self.spin2 = QDoubleSpinBox(None)
        self.spin2 .setValue(0.50)
        self.spin2.setRange(0.00,1.00)
        self.spin2.setSingleStep(0.01)
        self.spin3 = QDoubleSpinBox(None)

        self.button1 = QPushButton('浏览')
        self.button2 = QPushButton('浏览')
        self.button3 = QPushButton('保存配置')
        self.button4 = QPushButton('运行程序')
        self.button5 = QPushButton('保留按键')

        # 连接信号槽
        self.button1.clicked.connect(self.button1_click)
        self.button2.clicked.connect(self.button2_click)
        self.button3.clicked.connect(self.button3_click)
        self.button4.clicked.connect(self.button4_click)

        # 把控件添加到布局里
        layout1.addWidget(self.label1, 0, 0)
        layout1.addWidget(self.lineEdit1, 0, 1)
        layout1.addWidget(self.button1, 0, 2)
        layout1.addWidget(self.label2, 1, 0)
        layout1.addWidget(self.lineEdit2, 1, 1)
        layout1.addWidget(self.button2, 1, 2)
        group_box1.setLayout(layout1)
        layout2.addWidget(self.label3)
        layout2.addWidget(self.spin1)
        layout2.addWidget(self.label4)
        layout2.addWidget(self.spin2)
        layout2.addWidget(self.label5)
        layout2.addWidget(self.spin3)
        group_box2.setLayout(layout2)
        layout3.addWidget(self.lineEdit3)
        group_box3.setLayout(layout3)
        layout4.addWidget(self.button3)
        layout4.addWidget(self.button4)
        layout4.addWidget(self.button5)
        layout_set.addWidget(group_box1)
        layout_set.addWidget(group_box2)
        layout_set.addWidget(group_box3)
        layout_set.addLayout(layout4)
        layout_show.addWidget(self.image_label_org)
        layout_show.addWidget(self.output_text)

        main_layout.addLayout(layout_set)
        main_layout.addLayout(layout_show)

        # 应用于布局
        self.setLayout(main_layout)


    def button1_click(self):
        self.model_name, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', ' 所有文件 (*.*)')
        self.lineEdit1.setText(self.model_name)
        print(self.model_name)
        self.output_text.append('选择模型：'+self.model_name)


    def button2_click(self):
        self.image_name , _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '图形文件 (*.jpg *.png *.bmp *.jpeg *.gif);;所有文件 (*.*);;')
        self.lineEdit2.setText(self.image_name)
        print(self.image_name)
        self.output_text.append('选择图像：' + self.image_name)

    def button3_click(self):
        # 设置参数值
        self.Conf = self.spin1.value()
        self.IoU = self.spin2.value()
        self.other = self.spin3.value()
        # 输出日志信息
        self.output_text.append('------------以下配置已保存------------')
        self.output_text.append('模型位置：' + self.model_name)
        self.output_text.append('图像位置：' + self.image_name)
        self.output_text.append('Conf：' + str(self.Conf))
        self.output_text.append('IoU：' + str(self.IoU))
        self.output_text.append('other：' + str(self.other))

        img = cv.imread(self.image_name)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, byteValue = img.shape
        bytePerLine = 3 * width
        q_image = QImage(img.data, width, height, bytePerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label_org.setPixmap(pixmap)

    def button4_click(self):
        save_path = "runs"
        res_img = ONNX_img_qt(self.model_name, self.image_name, save_path,self.Conf, self.IoU)

        self.output_text.append('运行结果已保存至runs文件夹')
        # 显示结果
        img = cv.cvtColor(res_img, cv.COLOR_BGR2RGB)
        height, width, byteValue = img.shape
        bytePerLine = 3 * width
        q_image = QImage(img.data, width, height, bytePerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label_org.setPixmap(pixmap)

'''
    def demo_fun(self):
        res = demo(self.fname)
        height, width = res.shape
        q_image = QImage(res.data, width, height, width, QImage.Format_Indexed8)
        for i in range(256):
            q_image.setColor(i, qRgb(i, i, i))
        pixmap = QPixmap.fromImage(q_image)
        self.image_label_org.setPixmap(pixmap)
'''
        
if __name__ == '__main__':
    # 创建QApplication类的实例
    app = QApplication(sys.argv)

    # 创建对象
    main = Zency()
    # 创建窗口
    main.show()

    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())

