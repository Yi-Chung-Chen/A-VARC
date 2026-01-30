import glob
import json
import os
import os.path as osp
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset


def normalize_01_into_pm1(x):
    """Normalize x from [0, 1] to [-1, 1] by (x*2) - 1"""
    return x.add(x).add_(-1)


def pil_loader(path):
    """PIL image loader"""
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img


# Class mappings for dataset variants
IMAGENET_A_CLASSES = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108,
    110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386,
    397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483,
    486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606,
    607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758,
    763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845,
    847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954,
    956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988
]

OBJECTNET_CLASSES = [
    409, 412, 414, 418, 419, 423, 434, 440, 444, 446, 455, 457, 462, 463, 470, 473, 479, 487, 499, 504, 507, 508, 518,
    530, 531, 533, 539, 543, 545, 549, 550, 559, 560, 563, 567, 578, 587, 588, 589, 601, 606, 608, 610, 618, 619, 620,
    623, 626, 629, 630, 632, 644, 647, 651, 655, 658, 659, 664, 671, 673, 677, 679, 689, 694, 695, 696, 700, 703, 720,
    721, 725, 728, 729, 731, 732, 737, 740, 742, 749, 752, 759, 761, 765, 769, 770, 772, 773, 774, 778, 783, 790, 792,
    797, 804, 806, 809, 811, 813, 828, 834, 837, 841, 842, 846, 849, 850, 851, 859, 868, 879, 882, 883, 893, 898, 902,
    906, 907, 909, 923, 930, 950, 951, 954, 968, 999
]

IMAGENET_R_SYNSETS = [
    "n01443537", "n01484850", "n01494475", "n01498041", "n01514859", "n01518878", 
    "n01531178", "n01534433", "n01614925", "n01616318", "n01630670", "n01632777",
    "n01644373", "n01677366", "n01694178", "n01748264", "n01770393", "n01774750",
    "n01784675", "n01806143", "n01820546", "n01833805", "n01843383", "n01847000",
    "n01855672", "n01860187", "n01882714", "n01910747", "n01944390", "n01983481",
    "n01986214", "n02007558", "n02009912", "n02051845", "n02056570", "n02066245",
    "n02071294", "n02077923", "n02085620", "n02086240", "n02088094", "n02088238",
    "n02088364", "n02088466", "n02091032", "n02091134", "n02092339", "n02094433",
    "n02096585", "n02097298", "n02098286", "n02099601", "n02099712", "n02102318",
    "n02106030", "n02106166", "n02106550", "n02106662", "n02108089", "n02108915",
    "n02109525", "n02110185", "n02110341", "n02110958", "n02112018", "n02112137",
    "n02113023", "n02113624", "n02113799", "n02114367", "n02117135", "n02119022",
    "n02123045", "n02128385", "n02128757", "n02129165", "n02129604", "n02130308",
    "n02134084", "n02138441", "n02165456", "n02190166", "n02206856", "n02219486",
    "n02226429", "n02233338", "n02236044", "n02268443", "n02279972", "n02317335",
    "n02325366", "n02346627", "n02356798", "n02363005", "n02364673", "n02391049",
    "n02395406", "n02398521", "n02410509", "n02423022", "n02437616", "n02445715",
    "n02447366", "n02480495", "n02480855", "n02481823", "n02483362", "n02486410",
    "n02510455", "n02526121", "n02607072", "n02655020", "n02672831", "n02701002",
    "n02749479", "n02769748", "n02793495", "n02797295", "n02802426", "n02808440",
    "n02814860", "n02823750", "n02841315", "n02843684", "n02883205", "n02906734",
    "n02909870", "n02939185", "n02948072", "n02950826", "n02951358", "n02966193",
    "n02980441", "n02992529", "n03124170", "n03272010", "n03345487", "n03372029",
    "n03424325", "n03452741", "n03467068", "n03481172", "n03494278", "n03495258",
    "n03498962", "n03594945", "n03602883", "n03630383", "n03649909", "n03676483",
    "n03710193", "n03773504", "n03775071", "n03888257", "n03930630", "n03947888",
    "n04086273", "n04118538", "n04133789", "n04141076", "n04146614", "n04147183",
    "n04192698", "n04254680", "n04266014", "n04275548", "n04310018", "n04325704",
    "n04347754", "n04389033", "n04409515", "n04465501", "n04487394", "n04522168",
    "n04536866", "n04552348", "n04591713", "n07614500", "n07693725", "n07695742",
    "n07697313", "n07697537", "n07714571", "n07714990", "n07718472", "n07720875",
    "n07734744", "n07742313", "n07745940", "n07749582", "n07753275", "n07753592",
    "n07768694", "n07873807", "n07880968", "n07920052", "n09472597", "n09835506",
    "n10565667", "n12267677"
]

# Load ImageNet class index mapping
def _load_imagenet_class_mapping():
    """Load ImageNet class index mapping"""
    class_mapping_path = osp.join(osp.dirname(__file__), 'imagenet_class_index.json')
    with open(class_mapping_path, 'r') as f:
        imagenet_classes = json.load(f)
    
    # Create mapping: synset -> index
    synset_to_idx = {}
    for idx, (synset, name) in imagenet_classes.items():
        synset_to_idx[synset] = int(idx)
    
    return synset_to_idx

SYNSET_TO_IDX = _load_imagenet_class_mapping()


def get_imagenet_r_classes():
    """Get ImageNet-R class indices"""
    imagenetr_classes = []
    for synset in IMAGENET_R_SYNSETS:
        if synset in SYNSET_TO_IDX:
            imagenetr_classes.append(SYNSET_TO_IDX[synset])
    return sorted(imagenetr_classes)


def apply_synset_subset(dataset, synset_subset_path):
    """Apply synset subset filtering to dataset"""
    # Load synset subset
    with open(synset_subset_path, 'r') as f:
        subset_synsets = [line.strip() for line in f.readlines()]
    
    subset_synsets = sorted(subset_synsets)
    
    # Map synsets to ImageNet indices
    subset_indices = []
    for synset in subset_synsets:
        if synset not in SYNSET_TO_IDX:
            raise ValueError(f"Synset {synset} not found in ImageNet class mapping")
        subset_indices.append(SYNSET_TO_IDX[synset])
    
    # Apply filtering based on dataset type
    if hasattr(dataset, 'samples'):  # ImageFolder-like
        filtered_samples = []
        for sample_path, _ in dataset.samples:
            synset = sample_path.split('/')[-2]
            if synset in subset_synsets:
                original_idx = SYNSET_TO_IDX[synset]
                filtered_samples.append((sample_path, original_idx))
        
        dataset.samples = filtered_samples
        dataset.targets = [target for _, target in filtered_samples]
        if hasattr(dataset, 'imgs'):
            dataset.imgs = filtered_samples
            
    elif hasattr(dataset, '_images'):  # ImageNet variants
        filtered_images = []
        for image_path in dataset._images:
            synset = image_path.split('/')[-2]
            if synset in subset_synsets:
                filtered_images.append(image_path)
        dataset._images = filtered_images
        
    elif hasattr(dataset, 'fnames'):  # ImageNetV2
        filtered_fnames = []
        for fname in dataset.fnames:
            class_idx = int(str(fname).split('/')[-2])
            if class_idx in subset_indices:
                filtered_fnames.append(fname)
        dataset.fnames = filtered_fnames
    
    # Add subset information
    dataset.subset_indices = subset_indices
    
    print(f"Applied synset subset: {len(subset_synsets)} classes, subset indices: {subset_indices[:10]}...")
    return dataset


def reorder_imagenetr_by_style(images):
    """
    Reorder ImageNet-R samples by style diversity using round-robin across styles.
    
    Args:
        images: List of image paths
        
    Returns:
        List of reordered image paths
        
    Raises:
        ValueError: If any filename doesn't match {style}_{index}.jpg pattern
    """
    import re
    from collections import defaultdict
    
    # Parse filenames and group by style
    style_to_samples = defaultdict(list)
    pattern = re.compile(r'^(.+)_(\d+)\.jpg$')
    
    for image_path in images:
        filename = image_path.split('/')[-1]
        match = pattern.match(filename)
        
        if not match:
            raise ValueError(f"ImageNet-R filename '{filename}' doesn't match expected pattern {{style}}_{{index}}.jpg")
        
        style, index = match.groups()
        style_to_samples[style].append((int(index), image_path))
    
    # Sort samples within each style by index
    for style in style_to_samples:
        style_to_samples[style].sort(key=lambda x: x[0])
    
    # Round-robin reordering across styles
    reordered = []
    styles = sorted(style_to_samples.keys())  # Consistent ordering
    max_samples = max(len(samples) for samples in style_to_samples.values())
    
    for idx in range(max_samples):
        for style in styles:
            if idx < len(style_to_samples[style]):
                _, image_path = style_to_samples[style][idx]
                reordered.append(image_path)
    
    return reordered


def apply_per_class_sampling(dataset, sample_per_class):
    """Apply per-class sampling to dataset"""
    random.seed(42)  # For reproducibility
    
    if hasattr(dataset, 'samples'):  # ImageFolder-like
        synset_to_samples = defaultdict(list)
        for sample_path, target in dataset.samples:
            synset = sample_path.split('/')[-2]
            synset_to_samples[synset].append((sample_path, target))
        
        filtered_samples = []
        for synset, samples in synset_to_samples.items():
            if len(samples) < sample_per_class:
                raise ValueError(f"Class {synset} has only {len(samples)} samples, need {sample_per_class}")
            selected = samples[:sample_per_class]
            filtered_samples.extend(selected)
        
        dataset.samples = sorted(filtered_samples)
        dataset.targets = [target for _, target in dataset.samples]
        if hasattr(dataset, 'imgs'):
            dataset.imgs = dataset.samples
            
    elif hasattr(dataset, '_images'):  # ImageNet variants
        synset_to_images = defaultdict(list)
        for image_path in dataset._images:
            synset = image_path.split('/')[-2]
            synset_to_images[synset].append(image_path)
        
        filtered_images = []
        for synset, images in synset_to_images.items():
            if len(images) < sample_per_class:
                raise ValueError(f"Class {synset} has only {len(images)} samples, need {sample_per_class}")

            # Apply style-aware reordering for ImageNet-R
            if isinstance(dataset, ImageNetR):
                images = reorder_imagenetr_by_style(images)

            selected = images[:sample_per_class]
            filtered_images.extend(selected)
        
        dataset._images = sorted(filtered_images)
        
    elif hasattr(dataset, 'fnames'):  # ImageNetV2
        class_to_fnames = defaultdict(list)
        for fname in dataset.fnames:
            class_idx = int(str(fname).split('/')[-2])
            class_to_fnames[class_idx].append(fname)
        
        filtered_fnames = []
        for class_idx, fnames in class_to_fnames.items():
            if len(fnames) < sample_per_class:
                raise ValueError(f"Class {class_idx} has only {len(fnames)} samples, need {sample_per_class}")
            selected = fnames[:sample_per_class]
            filtered_fnames.extend(selected)
        
        dataset.fnames = sorted(filtered_fnames, key=lambda x: str(x))
    
    print(f"Applied per-class sampling: {sample_per_class} samples per class")
    return dataset


class ImageNet(datasets.ImageFolder):
    """Standard ImageNet validation dataset"""
    
    def __init__(self, root, transform=None):
        super().__init__(root=osp.join(root, 'imagenet', 'val'), 
                        transform=transform)
        self.num_classes = 1000
        self.class_indices = list(range(1000))  # 0-999
        self.dataset_type = "imagenet"


class ImageNetV2(ImageNetV2Dataset):
    """ImageNet-V2 matched-frequency variant"""
    
    def __init__(self, root, transform=None):
        super().__init__(variant="matched-frequency", location=root, transform=transform)
        self.fnames.sort()
        self.num_classes = 1000
        self.class_indices = list(range(1000))  # 0-999
        self.dataset_type = "imagenetv2"


class ImageNetA(Dataset):
    """ImageNet-A adversarial examples (200 classes)"""
    
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = osp.join(root, 'imagenet-a')
        self._images = glob.glob(self.root + '/*/*')
        self._images.sort()
        self.transform = transform
        self.num_classes = 200
        self.class_indices = IMAGENET_A_CLASSES
        self.dataset_type = "imagenet-a"
        
        print(f"ImageNet-A: Found {len(self._images)} images")
    
    def __len__(self):
        return len(self._images)
    
    def path_to_cls(self, path):
        synset = path.split('/')[-2]
        return SYNSET_TO_IDX[synset]
    
    def __getitem__(self, index):
        filepath = self._images[index]
        img = pil_loader(filepath)
        label = self.path_to_cls(filepath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class ImageNetR(Dataset):
    """ImageNet-R rendition variants (200 classes)"""
    
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = osp.join(root, 'imagenet-r')
        self._images = glob.glob(self.root + '/*/*')
        self._images.sort()
        self.transform = transform
        self.num_classes = 200
        self.class_indices = get_imagenet_r_classes()
        self.dataset_type = "imagenet-r"
        
        print(f"ImageNet-R: Found {len(self._images)} images across {self.num_classes} classes")
    
    def __len__(self):
        return len(self._images)
    
    def path_to_cls(self, path):
        synset = path.split('/')[-2]
        return SYNSET_TO_IDX[synset]
    
    def __getitem__(self, index):
        filepath = self._images[index]
        img = pil_loader(filepath)
        label = self.path_to_cls(filepath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class ImageNetSketch(Dataset):
    """ImageNet-Sketch hand-drawn sketches (1000 classes)"""
    
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = osp.join(root, 'imagenet-sketch')
        self._images = glob.glob(self.root + '/*/*')
        self._images.sort()
        self.transform = transform
        self.num_classes = 1000
        self.class_indices = list(range(1000))  # 0-999
        self.dataset_type = "imagenet-sketch"
        
        print(f"ImageNet-Sketch: Found {len(self._images)} images")
    
    def __len__(self):
        return len(self._images)
    
    def path_to_cls(self, path):
        synset = path.split('/')[-2]
        return SYNSET_TO_IDX[synset]
    
    def __getitem__(self, index):
        filepath = self._images[index]
        img = pil_loader(filepath)
        label = self.path_to_cls(filepath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def build_dataset(data_path, final_reso, dataset_type,
                  synset_subset_path=None, sample_per_class=None):
    """
    Build dataset with unified interface.
    
    Args:
        data_path: Root path to datasets
        final_reso: Final image resolution
        dataset_type: One of "imagenet", "imagenetv2", "imagenet-a", "imagenet-r", 
                     "imagenet-sketch", "objectnet"
        synset_subset_path: Path to synset subset file (optional)
        sample_per_class: Number of samples per class (optional)
    
    Returns:
        dataset: Dataset with standardized attributes:
            - num_classes: Number of classes
            - class_indices: Original ImageNet indices
            - subset_indices: Filtered indices (if synset_subset_path used)
            - dataset_type: Dataset type string
    """
    # Build transform
    mid_reso = round(1.125 * final_reso)
    
    # Single crop transform
    transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    
    # Create dataset based on type
    if dataset_type == "imagenet":
        dataset = ImageNet(root=data_path, transform=transform)
    elif dataset_type == "imagenetv2":
        dataset = ImageNetV2(root=data_path, transform=transform)
    elif dataset_type == "imagenet-a":
        dataset = ImageNetA(root=data_path, transform=transform)
    elif dataset_type == "imagenet-r":
        dataset = ImageNetR(root=data_path, transform=transform)
    elif dataset_type == "imagenet-sketch":
        dataset = ImageNetSketch(root=data_path, transform=transform)
    elif dataset_type == "objectnet":
        from objectnet import ObjectNetDataset
        dataset = ObjectNetDataset(root=data_path, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Apply subset filtering if specified
    if synset_subset_path is not None:
        dataset = apply_synset_subset(dataset, synset_subset_path)
    
    # Apply per-class sampling if specified
    if sample_per_class is not None:
        dataset = apply_per_class_sampling(dataset, sample_per_class)
    
    # Set subset_indices if not already set (for logit filtering)
    if not hasattr(dataset, 'subset_indices'):
        dataset.subset_indices = dataset.class_indices
    
    print(f"Dataset {dataset_type}: {len(dataset)} samples, {dataset.num_classes} classes")
    
    return dataset