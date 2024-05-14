# Ultralytics YOLO 🚀, AGPL-3.0 license
import argparse
import os
import cv2 as cv
import sys
import time
import onnx
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core

class YOLOv8_ONNX:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        # self.classes = self.get_classes('z_class.txt')
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 检查onnx模型并初始化onnx
        onnx_model = onnx.load(self.onnx_model)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        else:
            print("Model correct")
        self.onnx_session = ort.InferenceSession(self.onnx_model)

        # Get the model inputs
        self.model_inputs = self.onnx_session.get_inputs()

    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv.FILLED
        )

        # Draw the label text on the image
        cv.putText(img, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    def preprocess(self, frame=None):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        # t_pre = time.time()
        if frame is None:
            self.img = cv.imread(self.input_image)
        else:
            self.img = frame
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Resize the image to match the input shape
        # img = cv.resize(img, (self.input_width, self.input_height))
        img = letterbox(self.img)[0]            # 修改此处为self.img，则保存的为缩略图大小；此处为img则保存的为原图大小

        # Convert the image color space from BGR to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)



        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        # print("预处理时间", time.time() - t_pre)
        # Return the preprocessed image data
        return image_data

    def postprocess_(self,input_image, output):     # 使用该方法比postprocess_方法处理速度更快，主要优化了boxes、scores、class_ids初始化部分
        t_post = time.time()
        # Transpose and squeeze the output to match the expected shape
        tx = time.time()
        outputs = np.transpose(np.squeeze(output[0]))
        ty = time.time()
        print("trans time", ty - tx)
        # Get the number of rows in the outputs array
        # rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        # boxes = []
        # scores = []
        # class_ids = []
        boxes = outputs[:, :4].tolist()
        scores = np.max(outputs[:, 4:], axis=1).tolist()
        class_ids = np.argmax(outputs[:, 4:], axis=1).tolist()

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        # Iterate over each row in the outputs array
        '''
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) )
                top = int((y - h / 2) )
                width = int(w)
                height = int(h)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        '''
        tz = time.time()
        print("for time", tz - ty)
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        ta = time.time()
        print("trans time", ta - tz)
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            box[0] = int((x - w / 2)*x_factor)
            box[1] = int((y - h / 2)*y_factor)
            box[2] = int(w*x_factor)
            box[3] = int(h*y_factor)
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        print("trans time", time.time() - ta)
        print("后处理时间", time.time() - t_post)
        # Return the modified input image
        return input_image

    def postprocess__(self,input_image, output):     # 使用该方法比postprocess方法处理速度更快，主要优化了for循环部分
        # t_post = time.time()
        # Transpose and squeeze the output to match the expected shape
        # tx = time.time()
        outputs = np.transpose(np.squeeze(output[0]))
        # ty = time.time()
        # print("trans time", ty - tx)
        # Get the number of rows in the outputs array
        # rows = outputs.shape[0]

        # ----------------------------------------------新方法，提速！
        scores = np.max(outputs[:, 4:], axis=1)
        predictions = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres].tolist()

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1).tolist()

        # Get bounding boxes for each object
        # boxes = self.extract_boxes(predictions)
        boxes = predictions[:, :4].tolist()
        # -----------------------------------------------

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        # print(x_factor,y_factor)
        # Iterate over each row in the outputs array
        # tz = time.time()
        # print("for time", tz - ty)
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        # ta = time.time()
        # print("trans time", ta - tz)
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            box[0] = int((x - w / 2)*x_factor)
            box[1] = int((y - h / 2)*y_factor)
            box[2] = int(w*x_factor)
            box[3] = int(h*y_factor)
            score = scores[i]
            class_id = class_ids[i]
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        # print("trans time", time.time() - ta)
        # print("后处理时间", time.time() - t_post)
        # Return the modified input image
        return input_image

    def main(self, frame=None):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        # session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


        # Store the shape of the input for later use
        # input_shape = model_inputs[0].shape
        # self.input_width = input_shape[2]       # 640->width
        # self.input_height = input_shape[3]      # 640->with

        # Preprocess the image data
        if frame is None:
            img_data = self.preprocess()
        else :
            img_data = self.preprocess(frame)

        self.input_height, self.input_width = img_data.shape[2:]
        # Run inference using the preprocessed image data
        # t_run = time.time()
        outputs = self.onnx_session.run(None, {self.model_inputs[0].name: img_data})
        # print("推理时间", time.time() - t_run)
        # Perform post-processing on the outputs to obtain output image.
        # return self.postprocess_(self.img, outputs)  # output image
        # return self.postprocess_(self.img, outputs)  # output image
        return self.postprocess__(self.img, outputs)  # output image

class YOLOv8_VINO:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, vino_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            vino_model: Path to the VINO model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.vino_model = vino_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        # self.classes = self.get_classes('z_class.txt')
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

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
        self.model = self.core.read_model(model=self.vino_model)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")  # 设备自动选择
        # self.input_layer = self.compiled_model.inputs[0]
        self.output_layer = self.compiled_model.outputs[0]
        # print(self.input_layer)
        # print(self.output_layer)

    # 读取类别
    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv.FILLED
        )

        # Draw the label text on the image
        cv.putText(img, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    def preprocess(self, frame=None):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        # t_pre = time.time()
        if frame is None:
            self.img = cv.imread(self.input_image)
        else:
            self.img = frame
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Resize the image to match the input shape
        # img = cv.resize(img, (self.input_width, self.input_height))
        img = letterbox(self.img)[0]            # 修改此处为self.img，则保存的为缩略图大小；此处为img则保存的为原图大小

        # Convert the image color space from BGR to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        # print("预处理时间", time.time() - t_pre)
        # Return the preprocessed image data
        return image_data

    def postprocess_(self,input_image, output):     # 使用该方法比postprocess_方法处理速度更快，主要优化了boxes、scores、class_ids初始化部分
        t_post = time.time()
        # Transpose and squeeze the output to match the expected shape
        tx = time.time()
        outputs = np.transpose(np.squeeze(output[0]))
        ty = time.time()
        print("trans time", ty - tx)
        # Get the number of rows in the outputs array
        # rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        # boxes = []
        # scores = []
        # class_ids = []
        boxes = outputs[:, :4].tolist()
        scores = np.max(outputs[:, 4:], axis=1).tolist()
        class_ids = np.argmax(outputs[:, 4:], axis=1).tolist()

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        # Iterate over each row in the outputs array
        '''
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) )
                top = int((y - h / 2) )
                width = int(w)
                height = int(h)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        '''
        tz = time.time()
        print("for time", tz - ty)
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        ta = time.time()
        print("trans time", ta - tz)
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            box[0] = int((x - w / 2)*x_factor)
            box[1] = int((y - h / 2)*y_factor)
            box[2] = int(w*x_factor)
            box[3] = int(h*y_factor)
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        print("trans time", time.time() - ta)
        print("后处理时间", time.time() - t_post)
        # Return the modified input image
        return input_image

    def postprocess__(self,input_image, output):     # 使用该方法比postprocess方法处理速度更快，主要优化了for循环部分
        # t_post = time.time()
        # Transpose and squeeze the output to match the expected shape
        # tx = time.time()
        outputs = np.transpose(np.squeeze(output[0]))
        # ty = time.time()
        # print("trans time", ty - tx)
        # Get the number of rows in the outputs array
        # rows = outputs.shape[0]

        # ----------------------------------------------新方法，提速！
        scores = np.max(outputs[:, 4:], axis=1)
        predictions = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres].tolist()

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1).tolist()

        # Get bounding boxes for each object
        # boxes = self.extract_boxes(predictions)
        boxes = predictions[:, :4].tolist()
        # -----------------------------------------------

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        # print(x_factor,y_factor)
        # Iterate over each row in the outputs array
        # tz = time.time()
        # print("for time", tz - ty)
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        # ta = time.time()
        # print("trans time", ta - tz)
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            box[0] = int((x - w / 2)*x_factor)
            box[1] = int((y - h / 2)*y_factor)
            box[2] = int(w*x_factor)
            box[3] = int(h*y_factor)
            score = scores[i]
            class_id = class_ids[i]
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        # print("trans time", time.time() - ta)
        # print("后处理时间", time.time() - t_post)
        # Return the modified input image
        return input_image

    def main(self, frame=None):
        """
        Performs inference using an VINO model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        # Preprocess the image data
        if frame is None:
            img_data = self.preprocess()
        else :
            img_data = self.preprocess(frame)
        self.input_height, self.input_width = img_data.shape[2:]


        # Run inference using the preprocessed image data
        # t_run = time.time()
        outputs = self.compiled_model([img_data])[self.output_layer]
        # print("推理时间", time.time() - t_run)

        # Perform post-processing on the outputs to obtain output image.
        # return self.postprocess_(self.img, outputs)  # output image
        # return self.postprocess_(self.img, outputs)  # output image
        return self.postprocess__(self.img, outputs)  # output image


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

def ONNX_img(args):
    # Check the requirements and select the appropriate backend (CPU or GPU)
    # check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # Create an instance of the YOLOv8 class with the specified arguments

    detection = YOLOv8_ONNX(args.onnx_path, args.img_path, args.conf_thres, args.iou_thres)
    # t0 = time.time()
    # Perform object detection and obtain the output image
    output_image = detection.main()
    # print("总时间", time.time() - t0)

    # Display  and Save the output image
    # cv.namedWindow("Output", cv.WINDOW_NORMAL)
    img_name = os.path.basename(args.img_path)
    file_name, file_extension = os.path.splitext(img_name)
    cv.imwrite('./runs/deploy/{}'.format(file_name+'_onnx'+file_extension), output_image)
    cv.imshow("Output", output_image)

    # Wait for a key press to exit
    cv.waitKey(1000)
    cv.destroyAllWindows()

def VINO_img(args):
    # Create an instance of the YOLOv8 class with the specified arguments
    detection = YOLOv8_VINO(args.vino_path, args.img_path, args.conf_thres, args.iou_thres)
    # t0 = time.time()
    # Perform object detection and obtain the output image
    output_image = detection.main()
    # print("总时间", time.time() - t0)

    # Display  and Save the output image
    # cv.namedWindow("Output", cv.WINDOW_NORMAL)
    img_name = os.path.basename(args.img_path)
    file_name, file_extension = os.path.splitext(img_name)
    cv.imwrite('./runs/deploy/{}'.format(file_name+'_vino'+file_extension), output_image)
    cv.imshow("Output", output_image)

    # Wait for a key press to exit
    cv.waitKey(1000)
    cv.destroyAllWindows()

# ONNX模型推理文件夹中的图片/单张图片，自动判断路径为图片/文件夹
def ONNX_foder_img(args):
    # 判断 img_path 是文件还是文件夹
    path_type = is_file_or_folder(args.img_path)

    if path_type == "File":                     # 处理文件
        if is_image_file(args.img_path):        # 处理文件图片
            print("Deal with Image File")
            ONNX_img(args)
        else:
            print(f"{args.img_path} is not a valid image file.")
    elif path_type == "Folder":                 # 处理文件夹
        print("Deal With Folder")
        image_files = [f for f in os.listdir(args.img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            print(f"{args.img_path} 中没有图片文件。")
            sys.exit(0)

        path = args.img_path
        for image_file in image_files:
            args.img_path = os.path.join(path, image_file)
            ONNX_img(args)
    else:
        print(f"{args.img_path} 不是一个有效的路径。")
        sys.exit(0)

# VINO模型推理文件夹中的图片/单张图片，自动判断路径为图片/文件夹
def VINO_foder_img(args):
    # 判断 img_path 是文件还是文件夹
    path_type = is_file_or_folder(args.img_path)

    if path_type == "File":                     # 处理文件
        if is_image_file(args.img_path):        # 处理文件图片
            print("Deal with Image File")
            VINO_img(args)
        else:
            print(f"{args.img_path} is not a valid image file.")
    elif path_type == "Folder":                 # 处理文件夹
        print("Deal With Folder")
        image_files = [f for f in os.listdir(args.img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            print(f"{args.img_path} 中没有图片文件。")
            sys.exit(0)

        path = args.img_path
        for image_file in image_files:
            args.img_path = os.path.join(path, image_file)
            VINO_img(args)
    else:
        print(f"{args.img_path} 不是一个有效的路径。")
        sys.exit(0)

def ONNX_video(args):
    detection = YOLOv8_ONNX(args.onnx_path, args.img_path, args.conf_thres, args.iou_thres)
    if args.img_path == "":
        video = cv.VideoCapture(0)              # 调用摄像头
        video_save_path = './runs/deploy/camera_onnx.mp4'
    else:
        video = cv.VideoCapture(args.img_path)     # 读取视频文件
        base_name = os.path.basename(args.img_path)
        file_name, file_extension = os.path.splitext(base_name)
        video_save_path = './runs/deploy/' + file_name + '_onnx' + file_extension
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
    # 判断是否成功创建视频流
    while video.isOpened():
        t1 = time.time()
        ret, frame = video.read()
        if ret is True:
            # frame = cv.flip(frame, 1)       # 调用摄像头时使用
            # --------------------------------------
            output_image = detection.main(frame)
            # print(output_image.shape)
            # -------------------------------------------
            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            output_image = cv.putText(output_image, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow("Video", output_image)
            if video_save_path != "":       # 保存结果
                out.write(output_image)
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

def VINO_video(args):
    detection = YOLOv8_VINO(args.vino_path, args.img_path, args.conf_thres, args.iou_thres)
    if args.img_path == "":
        video = cv.VideoCapture(0)              # 调用摄像头
        video_save_path = './runs/deploy/camera_vino.mp4'
    else:
        video = cv.VideoCapture(args.img_path)     # 读取视频文件
        base_name = os.path.basename(args.img_path)
        file_name, file_extension = os.path.splitext(base_name)
        video_save_path = './runs/deploy/' + file_name + '_vino' + file_extension


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
    # 判断是否成功创建视频流
    while video.isOpened():
        t1 = time.time()
        ret, frame = video.read()
        if ret is True:
            # frame = cv.flip(frame, 1)       # 调用摄像头时使用
            # --------------------------------------
            output_image = detection.main(frame)
            # print(output_image.shape)
            # -------------------------------------------
            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            output_image = cv.putText(output_image, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow("Video", output_image)
            if video_save_path != "":       # 保存结果
                out.write(output_image)
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


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    # 结果默认储存在./runs/deploy内
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="video", help="image/folder/video")      # 直接在这里修改任务
    parser.add_argument("--onnx_path", type=str, default="yolov8s.onnx", help="Input your ONNX model.")
    parser.add_argument("--vino_path", type=str, default="yolov8s_openvino_model/yolov8s.xml", help="Input your VINO model.")
    parser.add_argument("--img_path", type=str, default='', help="image path/image folder/video path")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    # Demo
    if args.content == 'image':
        args.img_path = "ultralytics/assets/bus.jpg"
        ONNX_img(args)
        VINO_img(args)
    elif args.content == 'folder':
        args.img_path = "ultralytics/assets"
        ONNX_foder_img(args)
        VINO_foder_img(args)
    elif args.content == 'video':
        args.img_path = "/media/zency/SSD-Data/Dataset/test.mp4"
        # args.img_path = ""
        ONNX_video(args)
        VINO_video(args)
    else:
        print("The content args is error!")

'''
# Yolov8 ONNX/VINO部署v2.0
# 在yolov8_example的基础上优化了后处理的部分，加快程序处理速度
# 本程序在ONNX部署的基础上添加了OpenVINO部署的程序
# 注：本程序在i9-12900K_CPU+RTX3090显卡机器上运行速度ONNX>Openvino，初步判定是硬件问题

Usage
# 注意使用前要注释部分Demo
# 1.ONNX
1.1.image
    python main.py -- content image --onnx_path yolov8m.onnx --img_path ultralytics/assets/bus.jpg --conf-thres 0.5 --iou-thres 0.5
1.2.folder
    python main.py -- content folder --onnx_path yolov8m.onnx --img_path ultralytics/assets --conf-thres 0.5 --iou-thres 0.5
1.3.video
    python main.py -- content video --onnx_path yolov8m.onnx --img_path /media/zency/SSD-Data/Dataset/test.mp4 --conf-thres 0.5 --iou-thres 0.5
    python main.py -- content video --onnx_path yolov8m.onnx --img_path "" --conf-thres 0.5 --iou-thres 0.5
# 2.ONNX
2.1.image
    python main.py -- content image --vino_path yolov8m_openvino_model/yolov8m.xml --img_path ultralytics/assets/bus.jpg --conf-thres 0.5 --iou-thres 0.5
2.2.folder
    python main.py -- content folder --vino_path yolov8m_openvino_model/yolov8m.xml --img_path ultralytics/assets --conf-thres 0.5 --iou-thres 0.5
2.3.video
    python main.py -- content video --vino_path yolov8m_openvino_model/yolov8m.xml --img_path /media/zency/SSD-Data/Dataset/test.mp4 --conf-thres 0.5 --iou-thres 0.5
    python main.py -- content video --vino_path yolov8m_openvino_model/yolov8m.xml --img_path "" --conf-thres 0.5 --iou-thres 0.5
    
参考链接https://cloud.tencent.com/developer/article/1981364?areaSource=102001.16&traceId=zs7OsuZd2sEX4KWUy-3mJ
'''
