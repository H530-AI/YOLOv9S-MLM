import argparse
import os.path
import cv2
import numpy as np
from tqdm import trange
from PIL import Image

import yaml
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM,GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

from utils.augmentations import letterbox
from utils.general import xywh2xyxy, non_max_suppression
from models.experimental import attempt_load


# 以下脚本更改自B站魔傀面具的热力图脚本


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        if len(result) == 2:
            logits_ = result[1][:, 4:]
            boxes_ = result[1][:, :4]
        else:
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])

        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov9_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        show_list = []
        for i in range(post_result.size(0)):
            if post_result[i].max() > post_result.max() * self.ratio:
                show_list.append(post_result[i])
        for i in trange(len(show_list)):
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(show_list[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


def run(
        weights='yolo.pt',
        img_path='./data/images/horses.jpg',
        data='./data/data.yaml',
        layer=11,
        ratio=0.02,
        conf_threshold=0.8,
        method='GradCAM',
        source='./runs/hot_plot/',
        test_type='all',
        renormalize=False,
):
    assert os.path.exists(data), "data.yaml文件不存在"
    assert os.path.exists(weights), "权重文件不存在"
    with open(data, 'r', encoding='utf-8') as f:
        yaml_load = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(yaml_load['names'], type(yaml_load))
    class_name = yaml_load['names']

    target = yolov9_target(test_type, conf_threshold, ratio)

    model = attempt_load(weights)
    model.requires_grad_(True)
    get_model_layers = model.model[int(layer) if isinstance(layer, str) else layer]
    method = eval(method)(model,[get_model_layers])
    method.activations_and_grads = ActivationsAndGradients(model, [get_model_layers], None)

    colors = np.random.uniform(0, 255, size=(len(class_name), 3)).astype(np.int_)

    if not os.path.exists(source):
        os.makedirs(source, exist_ok=True)

    if not os.path.isdir(img_path):
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0)
        try:
            grayscale_cam = method(tensor, [target])
        except AttributeError as e:
            return

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        pred = model(tensor)
        pred = non_max_suppression(pred[0], conf_thres=conf_threshold, iou_thres=0.65)[0]

    if renormalize:
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        image_float_np = img
        boxes = pred[:, :4].cpu().detach().numpy().astype(np.int32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)

        cam_image = eigencam_image_renormalized


    cam_image = Image.fromarray(cam_image)
    cam_image.save(source + os.path.basename(img_path))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov9-c-converted.pt', help='训练好的权重路径')
    parser.add_argument('--img_path', type=str, default='./data/images/horses.jpg', help='图片路径')
    parser.add_argument('--data', type=str, default='./data/data.yaml', help='dataset.yaml path')
    parser.add_argument('--layer', default=12, help='测试层数')
    parser.add_argument('--source', default='./runs/hot_plot/', help='输出文件夹')
    parser.add_argument('--ratio', default=0.01, help='输出热力图的阈值百分数')
    parser.add_argument('--conf_threshold', default=0.8, help='置信度阈值')
    parser.add_argument('--test_type', default='all', help='检测类型：class, box, all')
    parser.add_argument('--renormalize', default=False, help='')
    parser.add_argument('--method', default='GradCAM', help='可供选择的方法：EigenCAM, HiResCAM, '
                                                                'GradCAMPlusPlus, GradCAM, XGradCAM,LayerCAM, '
                                                                'RandomCAM, EigenGradCAM')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

