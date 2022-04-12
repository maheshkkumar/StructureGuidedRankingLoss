import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from models.StructureGuidedRankingLoss.models.DepthNet import DepthNet

torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SGR():
    def __init__(self):
        self.weights = "./models/StructureGuidedRankingLoss/weights/model.pth.tar"
        self.model = DepthNet()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0]).to(device)
        self.model.load_state_dict(torch.load(self.weights)['state_dict'])
        self.model.eval()
        self.model.to(device)
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    def evaluate(self, img):

        img = Image.open(img).convert('RGB')
        ori_width, ori_height = img.size
        int_width, int_height = 448, 448
        img = img.resize((int_width, int_height), Image.ANTIALIAS)
        tensor_img = self.transform(img)

        input_img = torch.autograd.Variable(tensor_img.to(device).unsqueeze(0), volatile=True)
        output = self.model(input_img)
        depth = output.squeeze().cpu().data.numpy()
        min_d, max_d = depth.min(), depth.max()
        depth_norm = (depth - min_d) / (max_d - min_d)

        image_pil = Image.fromarray(depth_norm.astype(np.float32))
        image_pil = np.asarray(image_pil.resize((ori_width, ori_height), Image.BILINEAR)).astype(np.float32)

        return image_pil
