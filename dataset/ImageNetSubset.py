import random
from typing import Tuple, Dict, List
from torchvision.datasets import ImageFolder
from PIL import Image


class SubsetImageNet(ImageFolder):
    """
    A custom ImageNet dataset that deterministically subsets a specific number
    of classes based on a fixed random seed.

    Modified to force a resize to (224, 224) immediately upon loading.
    """

    def __init__(self, split='train', num_classes: int = 100, seed: int = 42, **kwargs):
        if split == 'train':
            root = '/home/yyang/datasets/ImageNet/train'
            # root = '/home/yyang/datasets/imagenet224/train'
        else:
            root = '/home/yyang/datasets/ImageNet/val'
            # root = '/home/yyang/datasets/imagenet/val224'
        self.num_classes = num_classes
        self.seed = seed

        # We must set seed and num_classes before calling super().__init__
        # because the parent class calls self.find_classes() during initialization.
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # 1. Get all available classes in the folder (default ImageFolder behavior)
        classes, _ = super().find_classes(directory)

        if self.num_classes > len(classes):
            raise ValueError(f"Requested {self.num_classes} classes, but only found {len(classes)}")

        # 2. Use a local random generator so we don't interfere with global PyTorch/Python seeds
        rng = random.Random(self.seed)

        # 3. Randomly select the desired number of classes
        selected_classes = rng.sample(classes, self.num_classes)

        # 4. Sort the selected classes to ensure the label mapping (0 to 99)
        # is consistent across train and val splits
        selected_classes.sort()

        # 5. Create a new class-to-index mapping
        class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}

        return selected_classes, class_to_idx

    def __getitem__(self, index: int):
        """
        Overrides __getitem__ to resize the image to 224x224 immediately after loading.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        # --- Force Resize Here ---
        # ImageNet images vary in size. We force them to 224x224.
        # This ignores aspect ratio (squash), similar to 'transforms.Resize((224, 224))'
        sample = sample.resize((224, 224), resample=Image.BILINEAR)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class FirstClassesImageNet(ImageFolder):
    """
    A custom ImageNet dataset that deterministically subsets the first specific number
    of classes (alphabetically sorted).

    Modified to force a resize to (224, 224) immediately upon loading.
    """

    def __init__(self, split='train', num_classes: int = 100, **kwargs):
        if split == 'train':
            root = '/home/yyang/datasets/imagenet-100/train'
            # root = '/media/yyang/DATA/datasets/ILSVRC2012/train'
        else:
            root = '/home/yyang/datasets/imagenet-100/val'
            # root = '/media/yyang/DATA/datasets/ILSVRC2012/val'
        self.num_classes = num_classes

        # We must set num_classes before calling super().__init__
        # because the parent class calls self.find_classes() during initialization.
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # 1. Get all available classes in the folder (default ImageFolder behavior automatically sorts them)
        classes, _ = super().find_classes(directory)

        if self.num_classes > len(classes):
            raise ValueError(f"Requested {self.num_classes} classes, but only found {len(classes)}")

        # 2. Select the first `num_classes` from the alphabetically sorted list
        selected_classes = classes[:self.num_classes]

        # 3. Create a new class-to-index mapping (0 to num_classes - 1)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}

        return selected_classes, class_to_idx

    def __getitem__(self, index: int):
        """
        Overrides __getitem__ to resize the image to 224x224 immediately after loading.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        # --- Force Resize Here ---
        # ImageNet images vary in size. We force them to 224x224.
        # This ignores aspect ratio (squash), similar to 'transforms.Resize((224, 224))'
        # sample = sample.resize((224, 224), resample=Image.BILINEAR)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target