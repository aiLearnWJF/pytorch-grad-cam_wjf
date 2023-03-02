
# %%
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
import torch
import torch.functional as F
import numpy as np
import requests
import cv2
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from torch import nn
from moco.builder import MoCo


class MoCo_infer(MoCo):
    def forward(self, im_q):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        rq = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)
        return rq
# 自定义模型
model = MoCo_infer(models.resnet50,128, 65536, 0.999, 0.07, True)

# **************手动载入模型 start******************************
if 1:
    pretrain_path = "/vehicle/yckj3860/code/moco_wjf/checkpoints_0801/checkpoint_0030.pth.tar"
    try:
        state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
        print(f"Loading pretrained model from {pretrain_path}")
    except FileNotFoundError as e:
        print(f'{pretrain_path} is not found! Please check this path.')
        raise e
    except KeyError as e:
        print("State dict keys error! Please check the state dict.")
        raise e

    state_dict = state_dict['state_dict']
    state_dict_new = {}
    for k in state_dict.keys():
        if k !="":
            if(len(k.split(".")) > 2 and ".".join(k.split(".")[:2]) == "module.encoder_q"):
                state_dict_new[".".join(k.split(".")[1:])] = state_dict[k]
                # print("after:",".".join(k.split(".")[2:]))
    state_dict = state_dict_new

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(incompatible.missing_keys)
    if incompatible.unexpected_keys:
        print(incompatible.unexpected_keys)

# **************手动载入模型 end******************************

def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(url).convert('RGB'))
    img = cv2.resize(img, (512, 512))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

# %% 作为比对的原型
car_img, car_img_float, car_tensor = get_image_from_url("my_test/cam1.png")
import pdb;pdb.set_trace()
car_concept_features = model(car_tensor)[0, :]


# %% 作为比对的目标，用来画图
image, image_float, input_tensor = get_image_from_url("my_test/cam2.png")
Image.fromarray(image)


# %%
class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)
    
target_layers = [model.encoder_q.layer4[-1]]
car_targets = [SimilarityToConceptTarget(car_concept_features)]

# Where is the car in the image
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    car_grayscale_cam = cam(input_tensor=input_tensor,
                        targets=car_targets)[0, :]
car_cam_image = show_cam_on_image(image_float, car_grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(car_cam_image, cv2.COLOR_RGB2BGR)

cv2.imwrite('gradcam_cam.jpg', cam_image)
