import pandas as pd
from data_aug.tezro_data import TezroDataset
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator

def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
                                          transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          # transforms.ToTensor()
    ])

    return data_transforms
ds = TezroDataset('demo.csv', './datasets', transform=ContrastiveLearningViewGenerator(
                                                                  get_simclr_pipeline_transform(165),
                                                                  2))
# print(pd.read_csv('demo.csv').iloc[0]['image_id_root'])

# print(ds[0][0][1].show())
# print(next(iter(ds)))
