import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import cv2

device_ids=[0]
device = f'cuda:{device_ids[0]}'
class InfDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.imgs_folder = img_folder
        self.transform = transform
        self.img_paths = []

        img_path = self.imgs_folder + '/'
        img_list = os.listdir(img_path)
        img_list.sort()

        self.img_nums = len(img_list)

        for i in range(self.img_nums):
            img_name = img_path + img_list[i]
            self.img_paths.append(img_name)
            
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img =  self.transform(img)
        name = self.img_paths[idx]
        return (img,name)

    def __len__(self):
        return self.img_nums



inf_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(224), 
         transforms.ToTensor(), 
        #  transforms.Normalize(mean=[0.4056, 0.3984, 0.4125], std=[0.1216, 0.1328, 0.1194]) ])
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataset and DataLoader
img_folder='./datasets/test'
inf_dataset = InfDataset(img_folder, transform=inf_transform)
inf_dataloader = data.DataLoader(inf_dataset, batch_size=64, shuffle=True)



def cluster_image_id(image_ids, threshold):
    clusters = []
    current_cluster = [image_ids[0]]
    for i in range(1, len(image_ids)):
        if abs(image_ids[i] - current_cluster[-1]) <= threshold:
            current_cluster.append(image_ids[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [image_ids[i]]
    clusters.append(current_cluster)
    return clusters

def softmax(x):
    e_x = np.exp(-x)+1
    return 1 / e_x

def name_cluster(datapath):
    data = pd.read_csv(datapath).sort_values(by="image_id", ascending=True)
    threshold = 100
    image_ids = data["image_id"].tolist()
    for i in range(len(image_ids)):
        image_ids[i] = int(image_ids[i].split(".")[0][6:])
    clusters = cluster_image_id(image_ids, threshold)
    name_list = []
    lable_list = []
    score_list = []


    for cluster in clusters:
        labels = []
        scores = []
        index = []
        labels_max = []
        indices_max = []
        score_max = []

        labels_min = []
        indices_min = []
        score_min = []

        for image_id in cluster:
            image_name = f"Gotcha{image_id}.png"
            row_data = data.loc[data["image_id"] == image_name]
            if not row_data.empty and row_data["softmax_score"].iloc[0] > 0.95:
                indices_max.append(image_name)
                labels_max.append(row_data["class_id"].iloc[0])
                score_max.append(row_data["softmax_score"].iloc[0])
            else:
                indices_min.append(image_name.replace('0','1'))
                labels_min.append(row_data["class_id"].iloc[0])
                score_min.append(row_data["softmax_score"].iloc[0])


        if indices_max:
            most_label_value = max(set(labels_max), key=labels_max.count)
            most_score_value = max(set(score_max), key=score_max.count)
            labels_max = [most_label_value] * len(labels_max)
            score_max = [most_score_value] * len(labels_max)

        index = indices_max + indices_min
        labels = labels_max+ labels_min
        scores = score_max + score_min

        name_list.extend(index)
        lable_list.extend(labels)
        score_list.extend(scores)
        df = pd.DataFrame({'image_id':name_list,
                    'class_id':lable_list,
                    'score':score_list})
    
        df.to_csv("./logs/results.csv",index=False)



def test():
    model_SAR_Eff = torch.load('./weights/results/SAR_feature_5000.pth')
    model_SAR_Eff.to(device)

    image_id_list=[]
    class_id_list=[]
    score_id_list=[]
    softmax_list = []
    model_SAR_Eff.eval()

    with torch.no_grad():
        for batch_idx, (img, name) in tqdm(enumerate(inf_dataloader)):
            img = img.to(device)
            output_unlabeled_SAR_Eff = model_SAR_Eff(img)
            score, pseudo_labeled = torch.max(output_unlabeled_SAR_Eff, 1)
            softmax_scores,_ = torch.max(F.softmax(output_unlabeled_SAR_Eff, dim=1),dim=1)
            for i in range(len(name)):
                image_id_one = name[i].split('/')[-1]
                image_id_list.append(image_id_one)
                class_id_list.append(pseudo_labeled[i].cpu().numpy())
                score_id_list.append(score[i].cpu().numpy())
                softmax_list.append(softmax_scores[i].cpu().numpy())

    df = pd.DataFrame({'image_id':image_id_list,
                       'class_id':class_id_list,
                       'score':score_id_list,
                       'softmax_score':softmax_list})
    df.to_csv("./logs/results_temp.csv",index=False)
    



test()
name_cluster("./logs/results_temp.csv")