# YOLO-Deploy-QT_Interface
用于部署YOLOv5和YOLOv8的可视化QT界面，可实现图片、文件夹、视频、摄像头的ONNX与OpenVino部署
![图片描述](assets/Demo.gif)

## 1.环境配置
本程序使用Conda创建环境，依次按照下面的命令创建并激活环境：
```bash
conda create -n qtenv python=3.8
conda activate qtenv
```
激活环境之后，进入本项目文件夹，并为环境安装必要的包：
```bash
pip install -r requirements.txt
```
至此，环境配置已完成，将该环境作为项目的环境即可。

## 2.数据与模型准备
- 在使用软件之前，需要准备好需要检测的图片、文件夹、视频、摄像头等
- 下载本项目提供的YOLOv5、YOLOv8的ONNX、OpenVINO的[模型文件](https://github.com/Zency-Sun/YOLO-Deploy-QT_Interface/releases/tag/v1.0.0)（注意解压），放在项目文件夹下
- 注意：本项目提供的模型文件是YOLOv5/8的COCO预训练模型直接转换过来的，若需要自定义类别，需要更改为自己的模型，并修改deploy_yolov5.py和deploy_yolov8.py的12行和32行的类别


## 3.使用指南
- 打开并运行main.py
- 依次选择“模型类型”、“部署类型”、“文件类型”、“模型位置”、“文件位置”等选项，并设置参数
- 点击“保存配置”，再点击“开始运行”即可运行，在“输出日志”中可以看到程序运行状态
- “停止运行”可以停止程序运行，“退出”可以退出程序
- 程序运行结果存放在./results文件夹下

## 4.声明
**使用代码请注明出处，拒绝白嫖，一起发扬开源精神**
