from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class ComicsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files_face = os.listdir(os.path.join(self.root_dir, 'face'))
        self.list_files_comics = os.listdir(os.path.join(self.root_dir, 'comics'))
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files_face)

    def __getitem__(self, index):
        face = self.list_files_face[index]
        comics = self.list_files_comics[index]
        face_path = os.path.join(self.root_dir, 'face', face)
        comics_path = os.path.join(self.root_dir, 'comics', comics)
    
        face = Image.open(face_path)
        comics = Image.open(comics_path)
        tr = config.transformer
        face, comics = tr(face), tr(comics)

        return face, comics

def test():
    test_dataset = ComicsDataset('face2comics/val')
    # test_dataset.__getitem__(0)
    test_dataset.__getitem__(200)

if __name__ == '__main__':
    print(test()[0])