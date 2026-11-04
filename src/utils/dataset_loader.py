from torch.utils.data import Dataset

import numpy as np
from sklearn.utils import compute_class_weight
import torch

import random
from typing import Iterator

from PIL import Image

def load_image(path, transforms):

    # Load image
    image = Image.open(path)
    image = image.convert('RGB')

    # Preprocess data
    if transforms is not None:
        image = transforms(image)

    return image

class ClassificationDataset(Dataset):


    def __init__(self, X, y, transforms=None):
        """
        Constructs a new dataset

        Args:
        """

        self.X = X
        self.y = y
        self.transforms = transforms
        self.len = len(self.X)


    def __len__(self):
        """
        Python operator to compute dataset length

        Returns: ``int``
            Dataset size
        """
        return self.len


    def __getitem__(self, idx):
        """
        Python operator to access one sample of the dataset from its position

        Args:
            idx: ``int``
                Sample position
        
        Returns: ``(PIL.Image, int)``
        """
        #print(f"{idx}_{self.len}_{len(self.X_2)}_{self.__len__()}", flush=True)
        # Get sample
        sample = self.y[idx]
        image = self.transforms(self.X[idx])

        return image, sample
    

    def compute_weights(self):
        return compute_class_weight("balanced", classes= np.unique(np.array(self.y)), y=np.array(self.y))
    
class OutlinerDataset(Dataset):


    def __init__(self, X, y, X_2 = None, y_2=None, transforms=None, transforms_2=None):
        """
        Constructs a new dataset

        Args:
        """

        self.X = X
        self.y = y
        self.X_2 = X_2
        self.y_2 = y_2
        self.transforms = transforms
        self.transforms_2 = transforms_2
        self.len = len(self.X)


    def __len__(self):
        """
        Python operator to compute dataset length

        Returns: ``int``
            Dataset size
        """
        if self.X_2 is not None:
            return self.len + len(self.X_2)
        else:
            return self.len


    def __getitem__(self, idx):
        """
        Python operator to access one sample of the dataset from its position

        Args:
            idx: ``int``
                Sample position
        
        Returns: ``(PIL.Image, int)``
        """
        #print(f"{idx}_{self.len}_{len(self.X_2)}_{self.__len__()}", flush=True)
        # Get sample
        if idx < self.len:
            sample = self.y[idx]
            image = self.transforms(self.X[idx])
            outlier = False
        else: 
            sample = self.y_2[idx - self.len]
            image = self.transforms_2(self.X_2[idx - self.len])
            outlier = True

        return image, sample, outlier
    

    def compute_weights(self):
        return compute_class_weight("balanced", classes= np.unique(np.array(self.y)), y=np.array(self.y))
    

class SurrogateDataset(Dataset):

    def __init__(self, lista, transforms=None):

        self.lista = lista
        self.X = []
        
        self.transforms = transforms

        lista_split = [name.split("_") for name in self.lista]
        for i in range(len(lista_split)):
            path_name = f"data/Imagenet1K/train/{lista_split[i][0]}/{lista_split[i][0]}_{lista_split[i][1]}"
            self.X.append(path_name)

        self.len = len(self.X)


    def __len__(self):
        """
        Python operator to compute dataset length

        Returns: ``int``
            Dataset size
        """
        return self.len
    
    def __getitem__(self, idx):
        """
        Python operator to access one sample of the dataset from its position

        Args:
            idx: ``int``
                Sample position
        
        Returns: ``(PIL.Image, int)``
        """
        #print(f"{idx}_{self.len}_{len(self.X_2)}_{self.__len__()}", flush=True)
        # Get sample
        path = self.X[idx]
        image = load_image(path, self.transforms)

        return image, -1

        
class SurrogateDatasetWeigthed(Dataset):
    def __init__(self, x, labels, logits):
        self.x = x
        self.labels = labels
        self.logits = logits
        self.len = len(self.x)

    def __len__(self):
        """
        Python operator to compute dataset length

        Returns: ``int``
            Dataset size
        """
        return self.len
    
    def __getitem__(self, idx):
        """
        Python operator to access one sample of the dataset from its position

        Args:
            idx: ``int``
                Sample position
        
        Returns: ``(PIL.Image, int, torch.Tensor)``
        """
        

        return self.x[idx], self.labels[idx], self.logits[idx]
    
# Wrapper for embeddings (not images)
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
class MixedLoader:
    """
    Iterator that mixes batches from two DataLoaders.
    Returns tuples: (inputs, labels, is_embedding: bool)
    - loader_a: DataLoader for images  -> returns (images, labels)
    - loader_b: DataLoader for embeddings -> returns (embeddings, labels)
    p_b: probability to draw a batch from loader_b (synthetic)
    If a loader exhausts, it will be re-created (wrap-around) to keep streaming.
    """
    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b
        p_b =  (100.0  * len(self.loader_b) ) / ( len(self.loader_a) + len(self.loader_b) )
        self.p_b = float(p_b)
        self._ita = None
        self._itb = None

    def __iter__(self) -> Iterator:
        self._ita = iter(self.loader_a)
        self._itb = iter(self.loader_b)
        return self

    def __next__(self):
        # choose which loader to draw from
        if random.random() < self.p_b:
            # draw from loader_b (synthetic embeddings)
            try:
                batch = next(self._itb)
            except StopIteration:
                self._itb = iter(self.loader_b)
                batch = next(self._itb)
            # expect batch = (embeddings, labels)
            inputs, labels = batch
            return inputs, labels
        else:
            # draw from loader_a (images)
            try:
                batch = next(self._ita)
            except StopIteration:
                self._ita = iter(self.loader_a)
                batch = next(self._ita)
            inputs, labels = batch
            return inputs, labels

    def __len__(self):
        # epoch size: here use sum of lengths so one epoch covers both datasets once
        try:
            la = len(self.loader_a)
        except Exception:
            la = 0
        try:
            lb = len(self.loader_b)
        except Exception:
            lb = 0
        return la + lb