import sys
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2 as cv
import os
from ui import Ui_Form
from yolov5_deploy import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

envpath = '/home/zency/anaconda3/envs/qt/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

# 界面v2.0版本
class MyForm(QWidget):
    def __init__(self, title):
        super(MyForm, self).__init__()
        # 定义类内变量
        self.model_path = 'controls/images/butterfly.png'
        self.image_path = 'controls/images/butterfly.png'
        self.video_path = None
        self.cap = None
        self.playing = False
        self.Conf = 0.25
        self.IoU = 0.45
        self.other = 0
        self.time = None
        self.model_type = "Yolov8"
        self.data_type = "image "

        # 初始化类内UI，并配置UI
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle(title)
        self.ui.radioButton.setChecked(True)
        self.ui.radioButton_3.setChecked(True)
        self.ui.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.spin_conf.setValue(0.25)
        self.ui.spin_conf.setRange(0.00,1.00)
        self.ui.spin_conf.setSingleStep(0.01)
        self.ui.spin_iou.setValue(0.45)
        self.ui.spin_iou.setRange(0.00,1.00)
        self.ui.spin_iou.setSingleStep(0.01)

        # 配置定时器
        self.timer = QTimer(self)

        # 链接槽函数
        self.ui.radioButton.toggled.connect(self.radioButton_model_state)
        self.ui.radioButton_2.toggled.connect(self.radioButton_model_state)
        self.ui.radioButton_3.toggled.connect(self.radioButton_file_state)
        self.ui.radioButton_4.toggled.connect(self.radioButton_file_state)
        self.ui.radioButton_5.toggled.connect(self.radioButton_file_state)
        self.ui.button_model.clicked.connect(self.select_model)
        self.ui.button_image.clicked.connect(self.select_file)
        self.ui.button_save.clicked.connect(self.save_config)
        self.ui.button_run.clicked.connect(self.run_program)
        self.ui.button_reserve1.clicked.connect(self.reserve1)
        self.ui.button_reserve2.clicked.connect(self.reserve2)
        self.timer.timeout.connect(self.updateFrame)

    def radioButton_model_state(self):
        button_selected = self.sender()
        if button_selected.isChecked() == True:
            self.model_type = button_selected.text()
            self.ui.output_text.append("已选择模型类别："+self.model_type)
            print(self.model_type, "被选中")
        # else:
        #     print(button_selected.text(), "被取消选中")

    def radioButton_file_state(self):
        button_selected = self.sender()
        if button_selected.isChecked() == True:
            self.data_type = button_selected.text()
            self.ui.output_text.append("已选择数据类别："+self.data_type)
            print(self.data_type)
            # print(self.data_type, "被选中")
        # else:
        #     print(button_selected.text(), "被取消选中")

    def select_model(self):
        # self.model_path, _ = QFileDialog.getOpenFileName(self, '打开模型文件', '.', ' 所有文件 (*.*)')
        self.model_path, _ = QFileDialog.getOpenFileName(self, '打开模型文件', '.','ONNX文件 (*.onnx);;所有文件 (*.*);;')
        self.ui.lineEdit_model.setText(self.model_path)
        print(self.model_path)
        self.ui.output_text.append('选择模型：' + self.model_path)
        # 判空逻辑
        if self.model_path == '':
            QMessageBox.information(self, "提示", "请选择模型！", QMessageBox.Ok)

    def select_file(self):
        # 根据文件类型选择文件
        if self.data_type == 'image ':
            self.image_path, _ = QFileDialog.getOpenFileName(self, '打开图片', '.',
                                                             '图形文件 (*.jpg *.png *.bmp *.jpeg *.gif);;所有文件 (*.*);;')
        elif self.data_type == 'video ':
            self.image_path, _ = QFileDialog.getOpenFileName(self, '打开视频', '.', '视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*)')
        elif self.data_type == 'image folder':
            self.image_path = QFileDialog.getExistingDirectory(self, "选择文件夹", '.')
        else:
            print("ERROR")
        # 判空逻辑
        if self.image_path == '':
            QMessageBox.information(self, "提示", "请选择文件！", QMessageBox.Ok)
        self.ui.lineEdit_image.setText(self.image_path)
        print(self.image_path)
        self.ui.output_text.append('选择图像：' + self.image_path)

    def save_config(self):
        # 设置参数值
        self.Conf = self.ui.spin_conf.value()
        self.IoU = self.ui.spin_iou.value()
        self.other = self.ui.spin_other.value()
        # 输出日志信息
        self.ui.output_text.append('------------以下配置已保存------------')
        self.ui.output_text.append('模型类型：' + self.model_type)
        self.ui.output_text.append('数据类型：' + self.data_type)
        self.ui.output_text.append('模型位置：' + self.model_path)
        self.ui.output_text.append('图像位置：' + self.image_path)
        self.ui.output_text.append('Conf：' + str(self.Conf))
        self.ui.output_text.append('IoU：' + str(self.IoU))
        self.ui.output_text.append('other：' + str(self.other))
        self.ui.output_text.append('--------------------------------------')

        img = cv.imread(self.image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, byteValue = img.shape
        bytePerLine = 3 * width
        q_image = QImage(img.data, width, height, bytePerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.image_label.setPixmap(pixmap)

    def run_program(self):
        # print("程序开始运行")
        self.ui.output_text.append("程序开始运行")

        res_img = ONNX_img(self.model_path, self.image_path,self.Conf, self.IoU)
        self.ui.output_text.append('运行结果已保存至runs文件夹')
        # 显示结果
        img = cv.cvtColor(res_img, cv.COLOR_BGR2RGB)
        height, width, byteValue = img.shape
        bytePerLine = 3 * width
        q_image = QImage(img.data, width, height, bytePerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.ui.image_label.setPixmap(pixmap)

    # 载入视频数据
    def reserve1(self):
        print("保留按键1按下")
        # folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", '.')
        video_path, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*)')

        if video_path:
            self.video_path = video_path
            self.cap = cv.VideoCapture(self.video_path)
            fps = int(self.cap.get(cv.CAP_PROP_FPS))
            # total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
            timer_interval = int(1000 / fps)
            self.timer.setInterval(timer_interval)
        else:
            print("未选择路径")

    def reserve2(self):
        print("保留按键2按下")
        # 切换播放/停止状态
        self.playing = not self.playing

        if self.playing:
            # 启动定时器
            self.timer.start()
            self.ui.button_reserve2.setText('停止')
        else:
            # 停止定时器
            self.timer.stop()
            self.ui.button_reserve2.setText('播放')

    def updateFrame(self):
        # 读取视频帧
        ret, frame = self.cap.read()
        if ret:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            # 将 QImage 显示在 QLabel 上
            # self.image_label_org.setPixmap(QPixmap.fromImage(q_img))
            self.ui.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.ui.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # 当视频播放完毕时停止定时器
            self.timer.stop()

    def closeEvent(self, event):
        # 在窗口关闭时释放资源
        if self.cap is not None:
            self.cap.release()


if __name__ == '__main__':
    # 创建QApplication类的实例
    app = QApplication(sys.argv)

    # 创建对象
    window = MyForm("Yolo Demo v2.0")
    # 创建窗口
    window.show()

    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())