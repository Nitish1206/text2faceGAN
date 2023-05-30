import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tqdm.notebook import tqdm
from dataclasses import asdict, dataclass

from preprocess import get_weighted_dataloader, extract_zip
from text_encoder.sentence_encoder import SentenceEncoder
from network import Generator, Discriminator
import cv2
from dfgan_network import NetG




@dataclass
class Config:
  epochs: int = 20
  image_size: int = 128
  initial_size: int = 64
  noise_size: int = 100
  batch_size: int = 64
  subset_size: int = 20_000
  num_channels: int = 3

  device: 'typing.Any' = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ImageGenerator:

  def __init__(self):

    self.cfg = Config()
    self.cfg_dict = asdict(self.cfg)

    self.sentence_encoder = SentenceEncoder(self.cfg.device)
    # self.newmodel = Generator(self.cfg.noise_size, self.cfg.image_size,self.cfg.num_channels, 768, 256)
    # self.netG = NetG(64, 100).to(device)
    self.newmodel = NetG(64, 100).to(self.cfg.device)
    # self.netD = NetD(64).to(device)

    # self.newmodel.load_state_dict(torch.load("models/ga.pth",map_location ='cpu'))
    self.newmodel.load_state_dict(torch.load("models/dfgan_ga_final.pth",map_location ='cpu'))
    self.newmodel.eval()
    self.test_noise = torch.randn(size=(1, self.cfg.noise_size)).cpu()

  def gen(self,txt_sent):
    # text_sentence="'The female has pretty high cheekbones and an oval face. Her hair is black. She has a slightly open mouth and a pointy nose. The female is smiling, looks attractive and has heavy makeup. She is wearing earrings and lipstick.'"


    self.test_embeddings = self.sentence_encoder.convert_text_to_embeddings([txt_sent])


    # test_noise = torch.randn(size=(1, cfg.noise_size)).cuda()
    self.test_image = self.newmodel(self.test_noise, self.test_embeddings).detach().cpu()
    # show_grid(torchvision.utils.make_grid(test_image, normalize=True, nrow=1))
    img=torchvision.utils.make_grid(self.test_image, normalize=True, nrow=1)


    # numpy_image = img.numpy()

    # # Convert the numpy array to a cv2 image
    # cv2_image = np.transpose(numpy_image, (1, 2, 0))
    # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    

    return img

    # # Display the image using cv2
    # cv2.imshow("Image", cv2_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


        