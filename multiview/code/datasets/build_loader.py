import platform
import random
import torch
from functools import partial

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, Sampler
from torch.utils.data import SequentialSampler

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None, df_img_size = None, img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes
        self.df_img_size = df_img_size

    def __iter__(self):
        num_batch = 0
        batch = []
        size = self.df_img_size
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size        

def build_dataloader(dataset,
                     batch_size,
                     image_size
                     ):
    data_loader = DataLoader(
        dataset,
        batch_sampler= BatchSampler(RandomSampler(dataset),
                                 batch_size=batch_size,
                                 drop_last=True,
                                 multiscale_step=1,
                                 df_img_size=image_size,
                                 img_sizes=[int(image_size*224/288), image_size]),
                    num_workers=4)

    IMAGENET_DEFAULT_MEAN =[0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    # data_loader = PrefetchLoader(data_loader, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    return data_loader


class PrefetchLoader:
    """A data loader wrapper for prefetching data."""

    def __init__(self, loader, mean, std):
        self.loader = loader
        self._mean = mean
        self._std = std

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        self.mean = torch.tensor([x * 255 for x in self._mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in self._std]).cuda().view(1, 3, 1, 1)

        for next_input_dict in self.loader:
            with torch.cuda.stream(stream):
                if isinstance(next_input_dict[0], list):
                    next_input_dict[0] = [
                        data.cuda(non_blocking=True).float().sub_(self.mean).div_(self.std)
                        for data in next_input_dict[0]
                    ]
                else:
                    data = next_input_dict[0].cuda(non_blocking=True)
                    next_input_dict[0] = data.float().sub_(self.mean).div_(self.std)

            if not first:
                yield input
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input_dict

        next_input_dict = None
        torch.cuda.empty_cache()
        yield input

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset