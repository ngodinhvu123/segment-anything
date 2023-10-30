import torch
from torch.utils.data import Dataset
import os
from transforms import ResizeLongestSide
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from build_sam import _build_sam
from torch.utils.data import Dataset, DataLoader
#Image
path_image = 'C:\\AB\\segment-anything\\images\\'
path_annotations = 'C:\\AB\\segment-anything\\annotations\\'
const_img_size =1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MyDataSet(Dataset):
    def __init__(self, path_image, path_ann, transform=None):
        """
        Args:
            data (list): Danh sách các mẫu dữ liệu.
            targets (list): Danh sách các nhãn tương ứng.
            transform (callable, optional): Biến đổi (transform) tùy chọn cho dữ liệu.
        """
        self.image_files = [f for f in os.listdir(path_image) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        self.path_image = path_image    
        self.path_ann = path_ann
        self.transform = transform(const_img_size)
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.image_files[idx]
        image_name = image.split('.')
        image_path = os.path.join(self.path_image, image)
        annotation_path = os.path.join(self.path_ann, image_name[0]+'.json')
        img_encode = cv2.imread(image_path)
        img_encode = cv2.cvtColor(img_encode, cv2.COLOR_BGR2RGB)
        data_ann = {}
        with open(annotation_path, 'r') as json_file:
            data_ann = json.load(json_file)

        point_coords = []
        boxes = []
        seg = []
        point_labels = []
        for ann in data_ann["annotations"]:
            point_coord = ann["point_coords"]
            point_coords.append(point_coord)
            output_array = np.ones_like(point_coord, dtype=bool)
            point_labels.append(output_array)
            boxes.append(ann["bbox"])
            rle_encoded = mask_util.decode(ann["segmentation"])
            rle_encoded = torch.tensor(rle_encoded, dtype=torch.float32).unsqueeze(0)
            seg.append(rle_encoded.shape)
            #shape point
            #print(np.asarray((ann["point_coords"])).shape)

        if self.transform:
            image = self.transform.apply_image(img_encode)
            image = image.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image).to(device),
            "original_size": (data_ann["image"]["height"], data_ann["image"]["width"]),
            "point_coords": torch.tensor(point_coords),
            "boxes": torch.tensor(boxes),
            "seg": seg,
            "point_labels": point_labels,
        }


data_train = MyDataSet(path_ann=path_annotations, path_image=path_image, transform=ResizeLongestSide)


#for i in range(0,1):
#    print(data_train.__getitem__(i)["point_coords"].shape)

model = _build_sam(
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    encoder_global_attn_indexes=[2, 5, 8, 11],
    checkpoint=None,
)
model.to(device)

dataloader = DataLoader(data_train)
data = [data_train.__getitem__(1)]
print(type(data[0]["image"]))
model(data, False)


