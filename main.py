import sys
import time
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2 as cv
import os
from ui import Ui_Form
from deploy_yolov5 import Yolov5ONNX, Yolov5VINO, filter_box
from deploy_yolov8 import YOLOv8_ONNX, YOLOv8_VINO
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
envpath = '/home/zency/anaconda3/envs/qt/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

# 界面v3.0版本
class MyForm(QWidget):
    def __init__(self, title):
        super(MyForm, self).__init__()
        # 定义类内变量
        self.model_path = 'yolov5m.onnx'
        self.image_path = 'images/zidane.jpg'
        self.video_path = None
        self.cap = None
        self.playing = False
        self.play_end = False
        self.Conf = 0.50
        self.IoU = 0.50
        self.other = 0
        self.time = None
        self.model_type = "Yolov5"
        self.deploy_type = "ONNX"
        self.data_type = "image"
        self.camera = False
        self.stop = False

        # 初始化类内UI，并配置UI
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon("images/logo_HFUT.png"))        # 设置图标
        self.ui.lineEdit_model.setText(self.model_path)
        self.ui.lineEdit_image.setText(self.image_path)
        self.ui.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.spin_conf.setValue(0.50)
        self.ui.spin_conf.setRange(0.00,1.00)
        self.ui.spin_conf.setSingleStep(0.01)
        self.ui.spin_iou.setValue(0.50)
        self.ui.spin_iou.setRange(0.00,1.00)
        self.ui.spin_iou.setSingleStep(0.01)

        # 配置定时器
        self.timer = QTimer(self)

        # 链接槽函数
        self.ui.radioButton_6.toggled.connect(self.radioButton_deploy_state)
        self.ui.radioButton_7.toggled.connect(self.radioButton_deploy_state)
        self.ui.radioButton.toggled.connect(self.radioButton_model_state)
        self.ui.radioButton_2.toggled.connect(self.radioButton_model_state)
        self.ui.radioButton_3.toggled.connect(self.radioButton_file_state)
        self.ui.radioButton_4.toggled.connect(self.radioButton_file_state)
        self.ui.radioButton_5.toggled.connect(self.radioButton_file_state)
        self.ui.checkBox.stateChanged.connect(self.checkBox_camera)
        self.ui.button_model.clicked.connect(self.select_model)
        self.ui.button_image.clicked.connect(self.select_file)
        self.ui.button_reserve1.clicked.connect(self.reserve1)
        self.ui.button_reserve2.clicked.connect(self.reserve2)
        self.ui.button_save.clicked.connect(self.save_config)
        self.ui.button_run.clicked.connect(self.run_program)
        self.ui.button_stop.clicked.connect(self.stop_program)
        self.ui.button_exit.clicked.connect(self.exit_window)
        self.timer.timeout.connect(self.updateFrame)
    def radioButton_model_state(self):
        button_selected = self.sender()
        if button_selected.isChecked() == True:
            self.model_type = button_selected.text()
            self.ui.output_text.append("已选择模型类别："+self.model_type)
            self.model_path = ''
            self.ui.lineEdit_model.setText(self.model_path)
            print(self.model_type, "被选中")
        # else:
        #     print(button_selected.text(), "被取消选中")
    def radioButton_deploy_state(self):
        button_selected = self.sender()
        if button_selected.isChecked() == True:
            self.deploy_type = button_selected.text()
            self.ui.output_text.append("已选择部署类别："+self.deploy_type)
            self.model_path = ''
            self.ui.lineEdit_model.setText(self.model_path)
            print(self.deploy_type, "被选中")
        # else:
        #     print(button_selected.text(), "被取消选中")

    def radioButton_file_state(self):
        button_selected = self.sender()
        if button_selected.isChecked() == True:
            self.data_type = button_selected.text()
            self.ui.output_text.append("已选择数据类别："+self.data_type)
            self.image_path = ''                            # 更改选项后，清空选项
            self.ui.lineEdit_image.setText(self.image_path)
            if self.data_type == 'video':
                self.ui.checkBox.setEnabled(True)
                if self.ui.checkBox.isChecked() == True:
                    self.ui.label_2.setEnabled(False)
                    self.ui.lineEdit_image.setEnabled(False)
                    self.ui.button_image.setEnabled(False)
            else:
                self.ui.checkBox.setEnabled(False)
                self.ui.label_2.setEnabled(True)
                self.ui.lineEdit_image.setEnabled(True)
                self.ui.button_image.setEnabled(True)
            print(self.data_type)
            # print(self.data_type, "被选中")
        # else:
        #     print(button_selected.text(), "被取消选中")
    def checkBox_camera(self):
        checker = self.sender()
        if checker.isChecked() == True:
            print("camera is checked!")
            self.ui.output_text.append('Camera Selected!')
            self.camera = True
            self.image_path = ''
            self.ui.label_2.setEnabled(False)
            self.ui.lineEdit_image.setEnabled(False)
            self.ui.button_image.setEnabled(False)
        else:
            print("camera is  not checked!")
            self.ui.output_text.append('Camera Deselected!')
            self.camera = False
            self.ui.label_2.setEnabled(True)
            self.ui.lineEdit_image.setEnabled(True)
            self.ui.button_image.setEnabled(True)

    def select_model(self):
        self.model_path = ''
        # self.model_path, _ = QFileDialog.getOpenFileName(self, '打开模型文件', '.', ' 所有文件 (*.*)')
        if self.deploy_type == 'ONNX':
            self.model_path, _ = QFileDialog.getOpenFileName(self, '打开模型文件', '.','ONNX文件 (*.onnx);;所有文件 (*.*);;')
        elif self.deploy_type == 'OpenVino':
            self.model_path, _ = QFileDialog.getOpenFileName(self, '打开模型文件', '.', 'Openvino文件 (*.xml);;所有文件 (*.*);;')
        else:
            print("ERROR")
        self.ui.lineEdit_model.setText(self.model_path)
        print(self.model_path)
        self.ui.output_text.append('选择模型：' + self.model_path)
        # 判空逻辑
        if self.model_path == '':
            QMessageBox.information(self, "提示", "请选择模型！", QMessageBox.Ok)

    def select_file(self):
        # 根据文件类型选择文件
        self.image_path = ''
        if self.data_type == 'image':
            self.image_path, _ = QFileDialog.getOpenFileName(self, '打开图片', '.',
                                                             '图形文件 (*.jpg *.png *.bmp *.jpeg *.gif);;所有文件 (*.*);;')
        elif self.data_type == 'video':
            self.image_path, _ = QFileDialog.getOpenFileName(self, '打开视频', '.', '视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*)')
        elif self.data_type == 'folder':
            self.image_path = QFileDialog.getExistingDirectory(self, "选择文件夹", '.')
        else:
            print(self.image_path)
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
        self.ui.output_text.append('部署类型：' + self.deploy_type)
        self.ui.output_text.append('文件类型：' + self.data_type)
        self.ui.output_text.append('模型位置：' + self.model_path)
        if self.camera:
            self.ui.output_text.append('Use Camera as Video')
        else:
            self.ui.output_text.append('图像位置：' + self.image_path)
        self.ui.output_text.append('Conf：' + str(self.Conf))
        self.ui.output_text.append('IoU：' + str(self.IoU))
        self.ui.output_text.append('other：' + str(self.other))
        self.ui.output_text.append('--------------------------------------')

        if self.data_type == 'image':
            img = cv.imread(self.image_path)
            self.show_image(img)


    def show_image(self, img_out):
        self.ui.image_label.setPixmap(QPixmap.fromImage(QImage(img_out.data, img_out.shape[1], img_out.shape[0], img_out.shape[1]*3, QImage.Format_RGB888).rgbSwapped()))
        QApplication.processEvents()
    def run_program(self):
        self.ui.output_text.append("程序开始运行...")
        if self.model_type == 'Yolov5' and self.deploy_type == 'ONNX' and self.data_type == 'image':
            print("Yolov5+ONNX+image")
            self.yolov5_onnx_image()
        elif self.model_type == 'Yolov5' and self.deploy_type == 'ONNX' and self.data_type == 'video':
            print("Yolov5+ONNX+video")
            self.yolov5_onnx_video()
        elif self.model_type == 'Yolov5' and self.deploy_type == 'ONNX' and self.data_type == 'folder':
            print("Yolov5+ONNX+folder")
            self.yolov5_onnx_folder()
        elif self.model_type == 'Yolov5' and self.deploy_type == 'OpenVino' and self.data_type == 'image':
            print("Yolov5+OpenVino+image")
            self.yolov5_vino_image()
        elif self.model_type == 'Yolov5' and self.deploy_type == 'OpenVino' and self.data_type == 'video':
            print("Yolov5+OpenVino+video")
            self.yolov5_vino_video()
        elif self.model_type == 'Yolov5' and self.deploy_type == 'OpenVino' and self.data_type == 'folder':
            print("Yolov5+OpenVino+folder")
            self.yolov5_vino_folder()
        elif self.model_type == 'Yolov8' and self.deploy_type == 'ONNX' and self.data_type == 'image':
            print("Yolov8+ONNX+image")
            self.yolov8_onnx_image()
        elif self.model_type == 'Yolov8' and self.deploy_type == 'ONNX' and self.data_type == 'video':
            print("Yolov8+ONNX+video")
            self.yolov8_onnx_video()
        elif self.model_type == 'Yolov8' and self.deploy_type == 'ONNX' and self.data_type == 'folder':
            print("Yolov8+ONNX+folder")
            self.yolov8_onnx_folder()
        elif self.model_type == 'Yolov8' and self.deploy_type == 'OpenVino' and self.data_type == 'image':
            print("Yolov8+OpenVino+image")
            self.yolov8_vino_image()
        elif self.model_type == 'Yolov8' and self.deploy_type == 'OpenVino' and self.data_type == 'video':
            print("Yolov8+OpenVino+video")
            self.yolov8_vino_video()
        elif self.model_type == 'Yolov8' and self.deploy_type == 'OpenVino' and self.data_type == 'folder':
            print("Yolov8+OpenVino+folder")
            self.yolov8_vino_folder()
        else:
            print("ERROR in run_program")

        self.ui.output_text.append('------------程序运行结束------------')
        ''''
        img_out = ONNX_img_qt(self.model_path, self.image_path, self.Conf, self.IoU)
        height, width, byteValue = img_out.shape
        bytePerLine = 3 * width
        q_image = QImage(img_out.data, width, height, bytePerLine, QImage.Format_RGB888).rgbSwapped()
        self.ui.image_label.setPixmap(QPixmap.fromImage(q_image))
        '''

    def stop_program(self):
        self.stop = True
        print("按下停止键")
    def exit_window(self):
        self.close()

    # 读取摄像头
    def reserve1(self):
        print("保留按键1按下")
        self.cap = cv.VideoCapture(0)
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        # total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        timer_interval = int(1000 / fps)
        self.timer.setInterval(timer_interval)
        self.timer.start()

    '''
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
    '''
    def reserve2(self):
        print("保留按键2按下")
        # 切换播放/停止状态
        self.playing = not self.playing
        # 播放还未结束
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
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)     # 实现视频循环播放的功能
            #self.timer.stop()                          # 视频播放单次


    def closeEvent(self, event):
        # 在窗口关闭时释放资源
        if self.cap is not None:
            self.cap.release()

    # 以下为部署调用函数
    def yolov5_onnx_image(self):
        model = Yolov5ONNX(self.model_path)
        t0 = time.time()
        output, or_img = model.inference(self.image_path)          # 1.推理
        infer_time = time.time() - t0
        print("ONNX inferance time:", infer_time)
        outbox = filter_box(output, self.Conf, self.IoU)        # 2.滤框
        '''
        # 注释掉，防止中断程序
        if len(outbox) == 0:
            print('没有发现物体')
            sys.exit(0)
        '''
        if np.size(outbox) != 0:
            or_img = model.draw(or_img, outbox)                 # 3.绘制结果并显示
        or_img = cv.putText(or_img, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img_name = os.path.basename(self.image_path)
        file_name, file_extension = os.path.splitext(img_name)
        cv.imwrite('./results/{}'.format(file_name + '_yolov5_onnx' + file_extension), or_img)
        self.show_image(or_img)
        self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time,'results'))

    def yolov5_onnx_video(self):
        model = Yolov5ONNX(self.model_path)
        if self.image_path == "":
            video = cv.VideoCapture(0)  # 调用摄像头
            video_save_path = './results/camera_yolov5_onnx.mp4'
        else:
            video = cv.VideoCapture(self.image_path)  # 读取视频文件
            base_name = os.path.basename(self.image_path)
            file_name, file_extension = os.path.splitext(base_name)
            video_save_path = './results/' + file_name + '_yolov5_onnx' + file_extension
        # 保存结果使用
        video_fps = int(round(video.get(cv.CAP_PROP_FPS)))
        if video_save_path != "":
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            size = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
            # size = (640,640)
            # _, frame = video.read()
            # size = (letterbox(frame)[0].shape[1], letterbox(frame)[0].shape[0])
            out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        # 判断是否成功创建视频流
        t0 = time.time()
        while video.isOpened():
            t1 = time.time()
            ret, frame = video.read()
            if ret is True:
                # frame = cv.flip(frame, 1)       # 调用摄像头时使用
                output, or_img = model.inference(frame)
                outbox = filter_box(output, self.Conf, self.IoU)
                '''
                # 防止异常退出
                if len(outbox) == 0:
                    print('没有发现物体')
                    sys.exit(0)
                '''
                if np.size(outbox) != 0:
                    or_img = model.draw(or_img, outbox)

                fps = (fps + (1. / (time.time() - t1))) / 2
                # print("fps= %.2f" % (fps))
                or_img = cv.putText(or_img, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv.imshow("Video", or_img)
                self.show_image(or_img)

                if video_save_path != "":  # 保存结果
                    out.write(or_img)
                # 按下q退出, 改成按键槽函数退出
                #if cv.waitKey(1) & 0xFF == ord('q'):
                #    break
                if self.stop:
                    self.stop = False
                    break
            else:
                break
        infer_time = time.time() - t0
        video.release()  # 释放并关闭窗口
        if video_save_path != "":  # 保存结果
            #print("Save processed video to the path :" + video_save_path)
            self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time, 'results'))
            out.release()

    def yolov5_onnx_folder(self):
        image_files = [f for f in os.listdir(self.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not image_files:
            print(f"{self.image_path} 中没有图片文件。")
            sys.exit(0)

        model = Yolov5ONNX(self.model_path)
        #for image_file in image_files:
        for index, image_file in enumerate(image_files):
            image_path = os.path.join(self.image_path, image_file)
            t0 = time.time()
            output, or_img = model.inference(image_path)        # 1.推理
            infer_time = time.time() - t0
            print("ONNX inferance time:", infer_time)

            outbox = filter_box(output, self.Conf, self.IoU)    # 2.滤框

            if np.size(outbox) != 0:
                or_img = model.draw(or_img, outbox)             # 3.绘制结果并显示
            or_img = cv.putText(or_img, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            img_name = os.path.basename(image_path)
            file_name, file_extension = os.path.splitext(img_name)
            cv.imwrite('./results/{}'.format(file_name + '_yolov5_onnx' + file_extension), or_img)
            self.show_image(or_img)
            self.ui.output_text.append("进度{}/{}，推理时长{:.4f}，结果保存于{}".format(index+1, len(image_files),infer_time, 'results'))
            if self.stop:
                self.stop = False
                break

    def yolov5_vino_image(self):
        model = Yolov5VINO(self.model_path)
        t0 = time.time()
        output, or_img = model.inference(self.image_path)   # 1.推理
        infer_time = time.time() - t0
        print("VINO inferrance time:", infer_time)

        outbox = filter_box(output, self.Conf, self.IoU)    # 2.滤框
        '''
        if len(outbox) == 0:
            print('没有发现物体')
            sys.exit(0)
        '''
        if np.size(outbox) != 0:
            or_img = model.draw(or_img, outbox)                 # 3.绘制结果并显示
        or_img = cv.putText(or_img, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img_name = os.path.basename(self.image_path)
        file_name, file_extension = os.path.splitext(img_name)
        cv.imwrite('./results/{}'.format(file_name+'_yolov5_vino'+file_extension), or_img)
        self.show_image(or_img)
        self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time,'results'))

    def yolov5_vino_video(self):
        video_save_path = 'results/test_vino.mp4'
        model = Yolov5VINO(self.model_path)
        if self.image_path == "":
            video = cv.VideoCapture(0)  # 调用摄像头
            video_save_path = './results/camera_yolov5_vino.mp4'
        else:
            video = cv.VideoCapture(self.image_path)  # 读取视频文件
            base_name = os.path.basename(self.image_path)
            file_name, file_extension = os.path.splitext(base_name)
            video_save_path = './results/' + file_name + '_yolov5_vino' + file_extension
        # 保存结果使用
        video_fps = int(round(video.get(cv.CAP_PROP_FPS)))
        if video_save_path != "":
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            size = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
            # size = (640,640)
            # _, frame = video.read()
            # size = (letterbox(frame)[0].shape[1], letterbox(frame)[0].shape[0])
            out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        # 判断是否成功创建视频流
        t0 = time.time()
        while video.isOpened():
            t1 = time.time()
            ret, frame = video.read()
            if ret is True:
                # frame = cv.flip(frame, 1)       # 调用摄像头时使用
                output, or_img = model.inference(frame)
                outbox = filter_box(output, self.Conf, self.IoU)
                '''
                # 防止异常退出
                if len(outbox) == 0:
                    print('没有发现物体')
                    sys.exit(0)
                '''
                if np.size(outbox) != 0:
                    or_img = model.draw(or_img, outbox)

                fps = (fps + (1. / (time.time() - t1))) / 2
                # print("fps= %.2f" % (fps))
                or_img = cv.putText(or_img, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv.imshow("Video", or_img)
                self.show_image(or_img)

                if video_save_path != "":  # 保存结果
                    out.write(or_img)
                # 按下q退出, 改成按键槽函数退出
                #if cv.waitKey(1) & 0xFF == ord('q'):
                #    break
                if self.stop:
                    self.stop = False
                    break
            else:
                break
        infer_time = time.time() - t0
        video.release()  # 释放并关闭窗口
        if video_save_path != "":  # 保存结果
            #print("Save processed video to the path :" + video_save_path)
            self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time, 'results'))
            out.release()

    def yolov5_vino_folder(self):
        image_files = [f for f in os.listdir(self.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not image_files:
            print(f"{self.image_path} 中没有图片文件。")
            sys.exit(0)

        model = Yolov5VINO(self.model_path)
        #for image_file in image_files:
        for index, image_file in enumerate(image_files):
            image_path = os.path.join(self.image_path, image_file)
            t0 = time.time()
            output, or_img = model.inference(image_path)        # 1.推理
            infer_time = time.time() - t0
            print("VINO inferance time:", infer_time)

            outbox = filter_box(output, self.Conf, self.IoU)    # 2.滤框

            if np.size(outbox) != 0:
                or_img = model.draw(or_img, outbox)             # 3.绘制结果并显示
            or_img = cv.putText(or_img, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            img_name = os.path.basename(image_path)
            file_name, file_extension = os.path.splitext(img_name)
            cv.imwrite('./results/{}'.format(file_name + '_yolov5_vino' + file_extension), or_img)
            self.show_image(or_img)
            self.ui.output_text.append("进度{}/{}，推理时长{:.4f}，结果保存于{}".format(index+1, len(image_files),infer_time, 'results'))
            if self.stop:
                self.stop = False
                break

    def yolov8_onnx_image(self):
        detection = YOLOv8_ONNX(self.model_path, self.image_path, self.Conf, self.IoU)
        t0 = time.time()
        output_image = detection.main()
        infer_time = time.time() - t0
        print("ONNX inferrance time:", infer_time)
        # 当图中无目标时，重新读取图片
        if len(output_image[0]) == 0:
            output_image = cv.imread(self.image_path)
        output_image = cv.putText(output_image, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img_name = os.path.basename(self.image_path)
        file_name, file_extension = os.path.splitext(img_name)
        cv.imwrite('./results/{}'.format(file_name + '_yolov8_onnx' + file_extension), output_image)
        self.show_image(output_image)
        self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time,'results'))

    def yolov8_onnx_video(self):
        detection = YOLOv8_ONNX(self.model_path, self.image_path, self.Conf, self.IoU)
        if self.image_path == "":
            video = cv.VideoCapture(0)  # 调用摄像头
            video_save_path = './results/camera_yolov8_onnx.mp4'
        else:
            video = cv.VideoCapture(self.image_path)  # 读取视频文件
            base_name = os.path.basename(self.image_path)
            file_name, file_extension = os.path.splitext(base_name)
            video_save_path = './results/' + file_name + '_yolov8_onnx' + file_extension
        # 保存结果使用
        video_fps = int(round(video.get(cv.CAP_PROP_FPS)))

        if video_save_path != "":
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            size = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
            # size = (640,640)
            # _, frame = video.read()
            # size = (letterbox(frame)[0].shape[1], letterbox(frame)[0].shape[0])
            # print(size)
            out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        t0 = time.time()
        # 判断是否成功创建视频流
        while video.isOpened():
            t1 = time.time()
            ret, frame = video.read()
            if ret is True:
                # frame = cv.flip(frame, 1)       # 调用摄像头时使用
                # --------------------------------------
                output_image = detection.main(frame)
                if len(output_image[0]) == 0:
                    output_image = frame
                # print(output_image.shape)
                # -------------------------------------------
                fps = (fps + (1. / (time.time() - t1))) / 2
                # print("fps= %.2f" % (fps))
                output_image = cv.putText(output_image, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                                          (0, 255, 0), 2)

                # cv.imshow("Video", output_image)
                self.show_image(output_image)
                if video_save_path != "":  # 保存结果
                    out.write(output_image)
                if self.stop:
                    self.stop = False
                    break
            else:
                break

        infer_time = time.time() - t0
        video.release()  # 释放并关闭窗口
        if video_save_path != "":  # 保存结果
            # print("Save processed video to the path :" + video_save_path)
            self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time, 'results'))
            out.release()
        cv.destroyAllWindows()

    def yolov8_onnx_folder(self):
        image_files = [f for f in os.listdir(self.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not image_files:
            print(f"{self.image_path} 中没有图片文件。")
            sys.exit(0)

        detection = YOLOv8_ONNX(self.model_path, self.image_path, self.Conf, self.IoU)
        for index, image_file in enumerate(image_files):
            image_path = os.path.join(self.image_path, image_file)
            detection.input_image = image_path
            t0 = time.time()
            output_image = detection.main()
            infer_time = time.time() - t0
            print("ONNX inferance time:", infer_time)
            if len(output_image[0]) == 0:
                output_image = cv.imread(image_path)
            output_image = cv.putText(output_image, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            img_name = os.path.basename(image_path)
            file_name, file_extension = os.path.splitext(img_name)
            cv.imwrite('./results/{}'.format(file_name + '_yolov8_onnx' + file_extension), output_image)
            self.show_image(output_image)
            self.ui.output_text.append("进度{}/{}，推理时长{:.4f}，结果保存于{}".format(index+1, len(image_files), infer_time, 'results'))

    def yolov8_vino_image(self):
        detection = YOLOv8_VINO(self.model_path, self.image_path, self.Conf, self.IoU)
        t0 = time.time()
        output_image = detection.main()
        infer_time = time.time() - t0
        print("VINO inferrance time:", infer_time)
        # 当图中无目标时，重新读取图片
        if len(output_image[0]) == 0:
            output_image = cv.imread(self.image_path)
        output_image = cv.putText(output_image, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img_name = os.path.basename(self.image_path)
        file_name, file_extension = os.path.splitext(img_name)
        cv.imwrite('./results/{}'.format(file_name + '_yolov8_vino' + file_extension), output_image)
        self.show_image(output_image)
        self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time,'results'))

    def yolov8_vino_video(self):
        detection = YOLOv8_VINO(self.model_path, self.image_path, self.Conf, self.IoU)
        if self.image_path == "":
            video = cv.VideoCapture(0)  # 调用摄像头
            video_save_path = './results/camera_yolov8_vino.mp4'
        else:
            video = cv.VideoCapture(self.image_path)  # 读取视频文件
            base_name = os.path.basename(self.image_path)
            file_name, file_extension = os.path.splitext(base_name)
            video_save_path = './results/' + file_name + '_yolov8_vino' + file_extension
        # 保存结果使用
        video_fps = int(round(video.get(cv.CAP_PROP_FPS)))

        if video_save_path != "":
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            size = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
            # size = (640,640)
            # _, frame = video.read()
            # size = (letterbox(frame)[0].shape[1], letterbox(frame)[0].shape[0])
            # print(size)
            out = cv.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        t0 = time.time()
        # 判断是否成功创建视频流
        while video.isOpened():
            t1 = time.time()
            ret, frame = video.read()
            if ret is True:
                # frame = cv.flip(frame, 1)       # 调用摄像头时使用
                # --------------------------------------
                output_image = detection.main(frame)
                if len(output_image[0]) == 0:
                    output_image = frame
                # print(output_image.shape)
                # -------------------------------------------
                fps = (fps + (1. / (time.time() - t1))) / 2
                # print("fps= %.2f" % (fps))
                output_image = cv.putText(output_image, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                                          (0, 255, 0), 2)

                # cv.imshow("Video", output_image)
                self.show_image(output_image)
                if video_save_path != "":  # 保存结果
                    out.write(output_image)
                if self.stop:
                    self.stop = False
                    break
            else:
                break

        infer_time = time.time() - t0
        video.release()  # 释放并关闭窗口
        if video_save_path != "":  # 保存结果
            # print("Save processed video to the path :" + video_save_path)
            self.ui.output_text.append("推理时长{:.4f}，结果保存于{}".format(infer_time, 'results'))
            out.release()
        cv.destroyAllWindows()

    def yolov8_vino_folder(self):
        image_files = [f for f in os.listdir(self.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if not image_files:
            print(f"{self.image_path} 中没有图片文件。")
            sys.exit(0)

        detection = YOLOv8_VINO(self.model_path, self.image_path, self.Conf, self.IoU)
        for index, image_file in enumerate(image_files):
            image_path = os.path.join(self.image_path, image_file)
            detection.input_image = image_path
            t0 = time.time()
            output_image = detection.main()
            infer_time = time.time() - t0
            print("VINO inferance time:", infer_time)
            if len(output_image[0]) == 0:
                output_image = cv.imread(image_path)
            output_image = cv.putText(output_image, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            img_name = os.path.basename(image_path)
            file_name, file_extension = os.path.splitext(img_name)
            cv.imwrite('./results/{}'.format(file_name + '_yolov8_vino' + file_extension), output_image)
            self.show_image(output_image)
            self.ui.output_text.append("进度{}/{}，推理时长{:.4f}，结果保存于{}".format(index+1, len(image_files), infer_time, 'results'))


if __name__ == '__main__':
    # 创建QApplication类的实例
    app = QApplication(sys.argv)
    # 创建对象
    window = MyForm("Yolo Demo v3.0")
    # 创建窗口
    window.show()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())

# TODO
# 1.保存路径的问题（保存路径可指定功能）
# 2.显示图像的后处理（图片显示居中，过宽/高图片的缩放功能）
# 3.选择模型的问题（如，选择的是yolov5+onnx，但是用户选择了yolov8的模型）
