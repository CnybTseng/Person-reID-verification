import os
import re
import torch
import random
import numpy as np
import os.path as osp
from torchvision import datasets

class TripletImageFolders(object):
    """Triplet torchvision ImageFolders. Only support Market1501 style dataset.
    Parameters:
        roots - Root directory paths.
        transform - A function/transform that takes in an PIL image and returns
            a transformed version.
        target_transform - A function/transform that takes in the target and transforms it.
        loader - A function to load an image given its path.
        is_valid_file - A function that takes path of an Image file and check if
            the file is a valid file (used to check of corrupt files).
    """
    def __init__(self,
        roots,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None):
        # Create torchvision ImageFolders.
        folders = []
        for root in roots:
            folders.append(datasets.ImageFolder(root, transform,
                target_transform, loader, is_valid_file))
        self.folders = folders
        targets = []    # target set union
        cameras = []    # camera set union
        start_target = 0    # start target for sub target set
        start_camera = 0    # start camera for sub camera set
        folderids = []  # folder indices for each data
        paths = []      # data paths
        for fid, folder in enumerate(folders):
            # Targets ensemble.
            sts = [s[1] for s in folder.samples]    # sub target set
            for t in sts:
                targets.append(t + start_target)
            start_target += max(sts) + 1    # target begins with 0
            # Camera ensemble.
            scs = [self._path2camera(s[0]) for s in folder.samples]   # sub camera set
            for c in scs:
                cameras.append(c + start_camera)
            start_camera += max(scs) + 1    # camera begins with 0
            folderids += [fid] * len(folder.samples)
            paths += [s[0] for s in folder.samples]
        self.targets = np.array(targets)
        self.cameras = np.array(cameras)
        self.folderids = folderids
        self.paths = paths

    def _path2camera(self, path):
        """Extract camera index from Market1501 style path"""
        pattern = re.compile(r'_c([\d]+)')
        match = pattern.search(path).groups()
        return int(match[0]) - 1    # original camera begin with 1
    
    def __len__(self):
        """Total number of images for all folders"""
        return self.targets.size

    def __getitem__(self, index):
        """Get a single item from dataset.
        Parameters:
            index - Sample index.
        Returns:
            Anchor sample, anchor target, positive sample, negative sample.
        """
        # Get anchor sample.
        sample = self._getsample(index)
        target = self.targets[index]
        folder = self.folders[self.folderids[index]]
        if folder.target_transform is not None:
            target = folder.target_transform(target)
        # Get positive sample.
        positive = self._getpossamp(index, target)
        # Get negative sample.
        negative = self._getnegsamp(target)
        return sample, target, positive, negative

    def _getpossamp(self, index, target):
        posids = np.argwhere(self.targets == target).flatten()
        posids = np.setdiff1d(posids, index)
        choice = np.random.choice(posids)
        return self._getsample(choice)
    
    def _getnegsamp(self, target):
        negids = np.argwhere(self.targets != target).flatten()
        choice = np.random.choice(negids)
        return self._getsample(choice)

    def _getsample(self, index):
        path = self.paths[index]
        folder = self.folders[self.folderids[index]]
        sample = folder.loader(path)
        if folder.transform is not None:
            sample = folder.transform(sample)
        return sample

if __name__ == '__main__':
    roots = ['/data/tseng/dataset/reid/market1501/Market-1501-v15.09.15/pytorch/train_all',
        '/data/tseng/dataset/substation/GUCHENG20200818/ch24_20200818131337/reid']
    dataset = TripletImageFolders(roots)
    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return batch
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=collate_fn, pin_memory=True)
    print('len: {}'.format(len(dataloader)))
    for data in dataloader:
        pass