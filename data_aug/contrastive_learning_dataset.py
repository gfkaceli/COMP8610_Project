from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

def simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size), sigma=(0.1, 2.0)),
        transforms.ToTensor()
    ])
    return data_transforms

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': get_cifar10_dataset,
            'stl10': get_stl10_dataset
        }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn(self.root_folder, n_views)

def get_cifar10_dataset(root_folder, n_views):
    transform = ContrastiveLearningViewGenerator(
        simclr_pipeline_transform(32),
        n_views
    )
    return datasets.CIFAR10(root_folder, train=True, transform=transform, download=True)

def get_stl10_dataset(root_folder, n_views):
    transform = ContrastiveLearningViewGenerator(
        simclr_pipeline_transform(96),
        n_views
    )
    return datasets.STL10(root_folder, split='unlabeled', transform=transform, download=True)