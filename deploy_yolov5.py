import os
import sys
import onnx
import onnxruntime as ort
from openvino.runtime import Core
import cv2 as cv
import numpy as np
import time
# from main import MyForm


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']  # coco80类别

class Yolov5ONNX(object):
    def __init__(self, onnx_path):
        """检查onnx模型并初始化onnx"""
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        else:
            print("Model correct")

        # options = ort.SessionOptions()
        # options.enable_profiling = True
        # self.onnx_session = ort.InferenceSession(onnx_path, sess_options=options,
        #                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()  # ['images']
        self.output_name = self.get_output_name()  # ['output0']

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy

        return input_feed

    # dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
    # thresh: 阈值
    def draw(self, image, box_data):
        # -------------------------------------------------------
        #	取整，方便画框
        # -------------------------------------------------------

        boxes = box_data[..., :4].astype(np.int32)  # x1 y1 x2 y2
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)
        for box, score, cl in zip(boxes, scores, classes):
            # top, left, right, bottom = box
            top, left, right, bottom = map(int, [box[0]*self.x_factor, box[1]*self.y_factor, box[2]*self.x_factor, box[3]*self.y_factor])
            # print('class: {}, score: {}'.format(CLASSES[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                       (top, left),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 255), 2)
        return image
    def inference(self, img_path):
        """ 1.cv2读取图像并resize
         2.图像转BGR2RGB和HWC2CHW(因为yolov5的onnx模型输入为 RGB：1 × 3 × 640 × 640)
         3.图像归一化
         4.图像增加维度
         5.onnx_session 推理 """
        # t_pre = time.time()
        if isinstance(img_path, np.ndarray):
            or_img = img_path
        else:
            or_img = cv.imread(img_path)

        # img = cv.resize(or_img, (640, 640))                 # resize后的原图 (640, 640, 3)【注，还有更好的图像处理方法，见博客】
        img = letterbox(or_img)[0]
        self.x_factor = or_img.shape[1] / img.shape[1]
        self.y_factor = or_img.shape[0] / img.shape[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)            # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)                  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)                   # [3, 640, 640]扩展为[1, 3, 640, 640]
        # print("预处理时间", time.time() - t_pre)
        # t_run = time.time()
        # img尺寸(1, 3, 640, 640) (NCHW)
        input_feed = self.get_input_feed(img)               # dict:{ input_name: input_value }
        pred = self.onnx_session.run(None, input_feed)[0]   # <class 'numpy.ndarray'>(1, 25200, 9)
        # print("ONNX推理时间", time.time() - t_run)
        return pred, or_img

class Yolov5VINO(object):
    def __init__(self, vino_path):
        # 初始化
        self.core = Core()
        '''
        # 获取设备信息
        devices = self.core.available_devices
        for device in devices:
            device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
            print(f"{device}: {device_name}")
        '''
        # 加载模型 load the openvino IR model
        self.model = self.core.read_model(model=vino_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="AUTO")  # 设备自动选择
        # self.input_layer = self.compiled_model.inputs[0]
        self.output_layer = self.compiled_model.outputs[0]
        # print(self.input_layer)
        # print(self.output_layer)

    # dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
    # thresh: 阈值
    def draw(self, image, box_data):
        # -------------------------------------------------------
        #	取整，方便画框
        # -------------------------------------------------------

        boxes = box_data[..., :4].astype(np.int32)  # x1 y1 x2 y2
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)
        for box, score, cl in zip(boxes, scores, classes):
            # top, left, right, bottom = box
            top, left, right, bottom = map(int, [box[0]*self.x_factor, box[1]*self.y_factor, box[2]*self.x_factor, box[3]*self.y_factor])
            # print('class: {}, score: {}'.format(CLASSES[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                       (top, left),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 255), 2)
        return image
    def inference(self, img_path):
        """ 1.cv2读取图像并resize
         2.图像转BGR2RGB和HWC2CHW(因为yolov5的vino模型输入为 RGB：1 × 3 × 640 × 640)
         3.图像归一化
         4.图像增加维度
         5.vino_session 推理 """
        if isinstance(img_path, np.ndarray):
            or_img = img_path
        else:
            or_img = cv.imread(img_path)

        # img = cv.resize(or_img, (640, 640))                     # resize后的原图 (640, 640, 3)【注，还有更好的图像处理方法，见博客】
        img = letterbox(or_img)[0]
        self.x_factor = or_img.shape[1] / img.shape[1]
        self.y_factor = or_img.shape[0] / img.shape[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)                # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)                      # vino模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)                       # [3, 640, 640]扩展为[1, 3, 640, 640]
        # t_run = time.time()
        pred = self.compiled_model([img])[self.output_layer]
        # print("openVINO推理时间", time.time() - t_run)

        # img尺寸(1, 3, 640, 640) (NCHW)

        return pred, or_img

def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(dets, thresh):
    # dets:x1 y1 x2 y2 score class
    # x[:,n]就是取所有集合的第n个数据
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #	置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    # print(scores)
    keep = []
    index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
    # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep

def filter_box(org_box, conf_thres, iou_thres):  # 过滤掉无用的框
    # -------------------------------------------------------
    #   删除为1的维度
    #	删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
    # (25200, 9)
    # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
    conf = org_box[..., 4] > conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
    box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框
    # print('box:符合要求的框')
    # print(box.shape)
    # box(N, 85), cls(N), all_cls(x)
    # -------------------------------------------------------
    #   通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
    all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
    # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #   1.将第6列元素替换为类别下标
    #	2.xywh2xyxy 坐标转换
    #	3.经过非极大抑制后输出的BOX下标
    #	4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []

        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5

        curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class   (Y, 6)
        curr_out_box = nms(curr_cls_box, iou_thres)  # 获得nms后，剩下的类别在curr_cls_box中的下标     (y,6)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output

# ONNX模型推理单张图片函数
def ONNX_img(onnx_path, img_path, conf_thres=0.5, iou_thres=0.5):
    model = Yolov5ONNX(onnx_path)
    t0 = time.time()
    output, or_img = model.inference(img_path)      # 1.推理
    print("ONNX inferance time:",time.time()- t0)

    # print('pred: 位置[0, 10000, :]的数组')
    # print(output.shape)
    # print(output[0, 10000, :])

    # t_post = time.time()
    outbox = filter_box(output, conf_thres, iou_thres)           # 2.滤框 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
    # print('outbox( x1 y1 x2 y2 score class):')
    # print(outbox)

    if len(outbox) == 0:
        print('没有发现物体')
        sys.exit(0)
    or_img = model.draw(or_img, outbox)                         # 3.绘制结果并显示
    # print("后处理时间", time.time() - t_post)
    # print("总时间", time.time() - t0)

    img_name = os.path.basename(img_path)
    file_name, file_extension = os.path.splitext(img_name)
    cv.imwrite('./runs/deploy/{}'.format(file_name+'_onnx'+file_extension), or_img)
    cv.imshow("{}Detecti Results".format(img_name), or_img)
    cv.waitKey(1000)
    cv.destroyAllWindows()

# VINO模型推理单张图片函数
def VINO_img(vino_path, img_path, conf_thres=0.5, iou_thres=0.5):
    model = Yolov5VINO(vino_path)
    t0 = time.time()
    output, or_img = model.inference(img_path)                      # 1.推理
    print("VINO inferrance time:", time.time() - t0)

    # print('pred: 位置[0, 10000, :]的数组')
    # print(output.shape)
    # print(output[0, 10000, :])
    # print(time.time()- t0)
    # t_post = time.time()
    outbox = filter_box(output, conf_thres, iou_thres)              # 2.滤框 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
    # print('outbox( x1 y1 x2 y2 score class):')
    # print(outbox)
    if len(outbox) == 0:
        print('没有发现物体')
        sys.exit(0)

    or_img = model.draw(or_img, outbox)                             # 3.绘制结果并显示
    # print("后处理时间", time.time() - t_post)
    # print("总时间", time.time() - t0)

    img_name = os.path.basename(img_path)
    file_name, file_extension = os.path.splitext(img_name)
    cv.imwrite('./runs/deploy/{}'.format(file_name+'_vino'+file_extension), or_img)
    cv.imshow("{}Detecti Results".format(img_name), or_img)
    cv.waitKey(1000)
    cv.destroyAllWindows()


def is_file_or_folder(path):
    if os.path.isfile(path):
        return "File"
    elif os.path.isdir(path):
        return "Folder"
    else:
        return "Not a valid path"


def is_image_file(file_path):
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    _, file_extension = os.path.splitext(file_path.lower())
    return file_extension in image_extensions


# ONNX模型推理文件夹中的图片/单张图片，自动判断路径为图片/文件夹
def ONNX_foder_img(onnx_path, path, conf_thres=0.5, iou_thres=0.5):
    # 判断 img_path 是文件还是文件夹
    path_type = is_file_or_folder(path)

    if path_type == "File":                     # 处理文件
        if is_image_file(path):                 # 处理文件图片
            print("Deal with Image File")
            ONNX_img(onnx_path, path, conf_thres, iou_thres)
        else:
            print(f"{path} is not a valid image file.")
    elif path_type == "Folder":                 # 处理文件夹
        print("Deal With Folder")
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            print(f"{path} 中没有图片文件。")
            sys.exit(0)

        for image_file in image_files:
            image_path = os.path.join(path, image_file)
            ONNX_img(onnx_path, image_path, conf_thres, iou_thres)
    else:
        print(f"{path} 不是一个有效的路径。")
        sys.exit(0)

def ONNX_foder_img_qt(onnx_path, path, conf_thres=0.5, iou_thres=0.5):
    # print("Deal With Folder")
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"{path} 中没有图片文件。")
        sys.exit(0)

    model = Yolov5ONNX(onnx_path)
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        # ONNX_img(onnx_path, image_path, conf_thres, iou_thres)
        t0 = time.time()
        output, or_img = model.inference(image_path)          # 1.推理
        infer_time = time.time() - t0
        print("ONNX inferance time:", infer_time)

        outbox = filter_box(output, conf_thres, iou_thres)   # 2.滤框 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class

        if np.size(outbox) != 0:
            or_img = model.draw(or_img, outbox)             # 3.绘制结果并显示
        or_img = cv.putText(or_img, "time= %.4f" % (infer_time), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img_name = os.path.basename(image_path)
        file_name, file_extension = os.path.splitext(img_name)
        cv.imwrite('./runs/deploy/{}'.format(file_name + '_onnx' + file_extension), or_img)

        # cv.imshow("{}Detecti Results".format(img_name), or_img)
        # cv.waitKey(1000)
        # cv.destroyAllWindows()
    # return or_img


# VINO模型推理文件夹中的图片/单张图片，自动判断路径为图片/文件夹
def VINO_foder_img(vino_path, path, conf_thres=0.5, iou_thres=0.5):
    # 判断 img_path 是文件还是文件夹
    path_type = is_file_or_folder(path)

    if path_type == "File":                     # 处理文件
        if is_image_file(path):                 # 处理文件图片
            print("Deal with Image File")
            VINO_img(vino_path, path, conf_thres, iou_thres)
        else:
            print(f"{path} is not a valid image file.")
    elif path_type == "Folder":                 # 处理文件夹
        print("Deal With Folder")
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            print(f"{path} 中没有图片文件。")
            sys.exit(0)

        for image_file in image_files:
            image_path = os.path.join(path, image_file)
            VINO_img(vino_path, image_path, conf_thres, iou_thres)
    else:
        print(f"{path} 不是一个有效的路径。")
        sys.exit(0)


# ONNX推理视频，并显示帧率
def ONNX_video(onnx_path, video_path, video_save_path, conf_thres=0.5, iou_thres=0.5):
    model = Yolov5ONNX(onnx_path)
    if video_path == "":
        video = cv.VideoCapture(0)              # 调用摄像头
    else:
        video = cv.VideoCapture(video_path)     # 读取视频文件
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
    while video.isOpened():
        t1 = time.time()
        ret, frame = video.read()
        if ret is True:
            # frame = cv.flip(frame, 1)       # 调用摄像头时使用
            output, or_img = model.inference(frame)
            outbox = filter_box(output, conf_thres, iou_thres)  # 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class

            if len(outbox) == 0:
                print('没有发现物体')
                sys.exit(0)

            or_img = model.draw(or_img, outbox)
            # img_name = os.path.basename(img_path)
            # cv.imwrite('./runs/deploy/{}'.format(img_name), or_img)

            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            or_img = cv.putText(or_img, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow("Video", or_img)
            if video_save_path != "":       # 保存结果
                out.write(or_img)
            # 按下q退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()                         # 释放并关闭窗口
    if video_save_path != "":               # 保存结果
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv.destroyAllWindows()

# VINO推理视频，并显示帧率
def VINO_video(vino_path, video_path, video_save_path, conf_thres=0.5, iou_thres=0.5):
    model = Yolov5VINO(vino_path)
    if video_path == "":
        video = cv.VideoCapture(0)              # 调用摄像头
    else:
        video = cv.VideoCapture(video_path)     # 读取视频文件
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
    while video.isOpened():
        t1 = time.time()
        ret, frame = video.read()
        if ret is True:
            # frame = cv.flip(frame, 1)       # 调用摄像头时使用
            output, or_img = model.inference(frame)
            outbox = filter_box(output, conf_thres, iou_thres)  # 最终剩下的Anchors：0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class

            if len(outbox) == 0:
                print('没有发现物体')
                sys.exit(0)

            or_img = model.draw(or_img, outbox)
            # img_name = os.path.basename(img_path)
            # cv.imwrite('./runs/deploy/{}'.format(img_name), or_img)

            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            or_img = cv.putText(or_img, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow("Video", or_img)
            if video_save_path != "":       # 保存结果
                out.write(or_img)
            # 按下q退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()                         # 释放并关闭窗口
    if video_save_path != "":               # 保存结果
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv.destroyAllWindows()


def video_paly():
    video = cv.VideoCapture('/media/zency/SSD-Data/Dataset/test.mp4')
    # 判断是否成功创建视频流
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            # frame = cv.flip(frame, 1)   # 图像翻转
            cv.imshow('Video', frame)
            # 设置视频播放速度
            # cv.waitKey(int(1000 / video.get(cv.CAP_PROP_FPS)))

            # 按下q退出
            if cv.waitKey(int(1000 / video.get(cv.CAP_PROP_FPS))) & 0xFF == ord('q'):
            # if cv.waitKey(int(1000 / video.get(cv.CAP_PROP_FPS))) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()

# YOLOv5内部函数，用于预处理输入图片
# 输入图片，新尺寸，填充颜色，是否最小矩形填充，是否允许拉伸，是否允许放大图片，步幅
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# YOLOv8内部函数，用于预处理输入图片
class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""
    def __init__(self, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, center=True, stride=32):
        # 注：在YOLOv8源程序中，auto默认为False，但是在推理时设置为True，在此改为默认True。这样，默认参数就与letterbox一样，相当于加了一个center功能
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


if __name__ == "__main__":

    onnx_path = './yolov5m.onnx'
    vino_path = 'yolov5m_openvino_model/yolov5m.xml'
    img_path = 'data/images/bus.jpg'
    video_path = '/media/zency/SSD-Data/Dataset/test.mp4'
    video_save_path1 = 'runs/deploy/test_onnx.mp4'
    video_save_path2 = 'runs/deploy/test_vino.mp4'

    ONNX_img(onnx_path, img_path)
    # ONNX_foder_img(onnx_path, img_path)
    # ONNX_video(onnx_path, video_path, video_save_path1)

    VINO_img(vino_path, img_path)
    # VINO_foder_img(vino_path, img_path)
    # VINO_video(vino_path, video_path, video_save_path2)


    # video_paly()
    print("The End")

'''
# Yolov5 ONNX/VINO部署v2.0
# 本程序和v1.0主体相同，不同的是本程序保存完整的图片和视频，而不是降采样，代价是牺牲了约0.2fps的帧率
# 本程序在ONNX部署的基础上添加了OpenVINO部署的程序
# 注：本程序在i9-12900K_CPU+RTX3090显卡机器上运行速度ONNX>Openvino，初步判定是硬件的问题
'''

