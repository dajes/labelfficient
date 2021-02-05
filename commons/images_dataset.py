import cv2
from torch.utils.data.dataset import Dataset


class ImagesDataset(Dataset):
    def __init__(self, img_paths, resize: int = 0):
        self.img_paths = img_paths
        self.resize = resize

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize))
        return img,

    @staticmethod
    def collate_fn(batch):
        return list(zip(*batch))
