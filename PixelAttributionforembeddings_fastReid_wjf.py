
# %%
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
import torch
# import torch.functional as F
import torch.nn.functional as F
import numpy as np
import requests
import cv2
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from torch import nn


import sys
sys.path.append('/vehicle/yckj3860/code/fast-reid-1.3.0/')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


# ┌────────────────────────────────────────────────────────────────────────┐
# │                           自定义模型 reid  start         
# └────────────────────────────────────────────────────────────────────────┘

def build_model():
    cfg = get_cfg()
    cfg.merge_from_file("/vehicle/yckj3860/code/fast-reid-1.3.0/logs/nonmotor/resnest_wjfPubExp_all_download_4700_1009_cosl_f1000/config.yaml")
    cfg.freeze()

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = DefaultTrainer.build_model(cfg)

    Checkpointer(model).load("/vehicle/yckj3860/code/fast-reid-1.3.0/logs/nonmotor/resnest_wjfPubExp_all_download_4700_1009_cosl_f1000/model_best.pth")  # load trained model
    # repvgg test need this line
    # model.backbone.deploy(True)


    return model

model = build_model().to("cpu")
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        # self.pool = torch.nn.MaxPool2d((1,1))
                
    def __call__(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return x[:,:,0,0]
model_fea = ResnetFeatureExtractor(model)
model_fea.eval()
# x_input = torch.zeros((2,3,224,224))
# y = model_fea(x_input)
# import pdb;pdb.set_trace()

# ┌────────────────────────────────────────────────────────────────────────┐
# │                           自定义模型 reid  end         
# └────────────────────────────────────────────────────────────────────────┘

def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(url).convert('RGB'))
    img = cv2.resize(img, (320, 160))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor

# %% 作为比对的原型
car_img, car_img_float, car_tensor = get_image_from_url("my_test/nonvehile1_1.jpg")

car_concept_features = model_fea(car_tensor)[0, :]



# %% 作为比对的目标，用来画图
image, image_float, input_tensor = get_image_from_url("my_test/nonvehile3_2.jpg")

Image.fromarray(image)


# %%
class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        #l2 norm
        # import pdb;pdb.set_trace()
        # model_output = F.normalize(model_output,dim=0)
        # self.features =  F.normalize(self.features,dim=0)

        # print("simlirity:",cos(model_output, self.features))
        # return cos(model_output, self.features)
        # 感觉应该平方一下，防止出现负的
        return cos(model_output, self.features)*cos(model_output, self.features)
        # return 1-cos(model_output, self.features)

    
target_layers = [model_fea.model.backbone.layer3[-1]]
car_targets = [SimilarityToConceptTarget(car_concept_features)]
# not_car_targets = [DifferenceFromConceptTarget(car_concept_features)]

# Where is the car in the image
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    car_grayscale_cam = cam(input_tensor=input_tensor,
                        targets=car_targets)[0, :]
car_cam_image = show_cam_on_image(image_float, car_grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(car_cam_image, cv2.COLOR_RGB2BGR)

cv2.imwrite('gradcam_cam.jpg', cam_image)
