import cv2
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
from PIL import Image
import time



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
frame_size = (1280,720)
IMG_PATH = './data/images'
DATA_PATH = './data'

model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
model.eval()

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

power = pow(10, 6)

def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH + '/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH + '/faceslist.pth')
    names = np.load(DATA_PATH + '/usernames.npy')
    return embeds, names


def inference(face, local_embeds, threshold=0.15):
    # local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)  # [1,512]
    # print(detect_embeds.shape)
    # [1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)  # (1,n), moi cot la tong khoang cach euclide so vs embed moi

    min_dist, embed_idx = torch.min(norm_score, dim=1)
    # print(min_dist.shape)
    if float(min_dist) * power < threshold or float(min_dist) * power > threshold*10:
        return -1, -1
    else:
        return embed_idx, min_dist.double()


def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]  # tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img, (face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

