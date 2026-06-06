import os
from torch.utils.data import Dataset
try:
    import orjson
except ImportError:
    import json

    class _OrjsonFallback:
        @staticmethod
        def loads(value):
            return json.loads(value)

    orjson = _OrjsonFallback()
from PIL import Image
import random
from torch.utils.data import default_collate
import numpy as np
from typing import List
from collections import defaultdict
from .transforms import squarepad_transform, targetpad_transform, DataAugmentation

def ensure_rgb(img):
    """
    Ensure image is in RGB mode, converting from grayscale if necessary
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

# FashionIQ Dataset Tree:
# fashion-iq/
# ├── images/
# │   ├── [image_name].png
# │   └── ...
# ├── captions/
# │   ├── cap.dress.train.json
# │   ├── cap.dress.val.json
# │   ├── cap.dress.test.json
# │   ├── cap.shirt.train.json
# │   ├── cap.shirt.val.json
# │   ├── cap.shirt.test.json
# │   ├── cap.toptee.train.json
# │   ├── cap.toptee.val.json
# │   └── cap.toptee.test.json
# └── image_splits/
#     ├── split.dress.train.json
#     ├── split.dress.val.json
#     ├── split.dress.test.json
#     ├── split.shirt.train.json
#     ├── split.shirt.val.json
#     ├── split.shirt.test.json
#     ├── split.toptee.train.json
#     ├── split.toptee.val.json
#     └── split.toptee.test.json
# 
class FashionIQDataset(Dataset):
    def __init__(
            self, 
            mode: str, 
            clothtype: str,
            preprocess: str,
            split: str = 'val', 
            path: str='.',
            augmenter: DataAugmentation = None
        ):
        
        super().__init__()

        # validate & store parameters
        assert mode in ['query', 'target']
        assert clothtype in ['dress', 'shirt', 'toptee']
        assert split in ['train', 'val', 'test']
        self.mode = mode
        self.split = split
        self.image_path = os.path.join(path, 'images')
        self.augmenter = augmenter
        self.preprocess = preprocess

        # read metadata & split files
        metadata_file = os.path.join(path, f'captions/cap.{clothtype}.{split}.json')
        split_file = os.path.join(path, f'image_splits/split.{clothtype}.{split}.json')

        if mode == 'query':
            self.metadata = orjson.loads(open(metadata_file).read())
            
        if mode == 'target':
            self.names = orjson.loads(open(split_file).read())


    def __getitem__(self, index):
        if self.mode == 'target':
            img_name = self.names[index]
            img_file = os.path.join(self.image_path, f'{img_name}.png')
            img = ensure_rgb(Image.open(img_file))
            img = self.preprocess(img)
            return {
                'img_name': img_name,
                'img': img
            }

        elif self.mode == 'query': 
            ref_img_name = self.metadata[index]['candidate']
            tgt_img_name = self.metadata[index]['target']
            text_instruction = self.metadata[index]['captions']
            ref_img_file = os.path.join(self.image_path, f'{ref_img_name}.png')

            if self.split == 'train':
                ref_img = ensure_rgb(Image.open(ref_img_file))
                if self.augmenter:
                    ref_img = self.augmenter.apply(ref_img)
                ref_img = self.preprocess(ref_img)

                tgt_img_file = os.path.join(self.image_path, f'{tgt_img_name}.png')
                tgt_img = ensure_rgb(Image.open(tgt_img_file))
                if self.augmenter:
                    tgt_img = self.augmenter.apply(tgt_img)
                tgt_img = self.preprocess(tgt_img)
                return {
                    'ref_img_name': ref_img_name,
                    'ref_img': ref_img,
                    'text_instruction': text_instruction,
                    'tgt_img_name': tgt_img_name,
                    'tgt_img': tgt_img
                }

            elif self.split == 'val':
                ref_img = self.preprocess(ensure_rgb(Image.open(ref_img_file)))
                return {
                    'ref_img_name': ref_img_name,
                    'ref_img': ref_img,
                    'text_instruction': text_instruction,
                    'tgt_img_name': tgt_img_name,
                }


    def __len__(self):
        if self.mode == 'target':
            return len(self.names)
        if self.mode == 'query':
            return len(self.metadata)


# reference: CLIP4CIR (CVPRW 2022)
def combine_captions(flattened_captions: List[str]):
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param caption_pair: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def fiq_collate_fn_train(batch):
    # default_collate with dict input returns a dict with batched values
    collated_batch = default_collate(batch)
    
    # process captions
    text_instruction = combine_captions(np.array(collated_batch['text_instruction']).T.flatten().tolist())
    
    return {
        'ref_img_name': collated_batch['ref_img_name'],
        'ref_img': collated_batch['ref_img'],
        'text_instruction': text_instruction,
        'tgt_img_name': collated_batch['tgt_img_name'],
        'tgt_img': collated_batch['tgt_img']
    }


def fiq_collate_fn_val(batch):
    # default_collate with dict input returns a dict with batched values
    collated_batch = default_collate(batch)
    
    # process captions
    flattened_captions = np.array(collated_batch['text_instruction']).T.flatten().tolist()
    text_instruction = [
        f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
        i in range(0, len(flattened_captions), 2)]
    
    return {
        'ref_img_name': collated_batch['ref_img_name'],
        'ref_img': collated_batch['ref_img'],
        'text_instruction': text_instruction,
        'tgt_img_name': collated_batch['tgt_img_name']
    }


# CIRR Dataset Tree:
# CIRR/
# ├── captions/
# │   ├── cap.rc2.train.json
# │   ├── cap.rc2.val.json
# │   └── cap.rc2.test1.json
# ├── image_splits/
# │   ├── split.rc2.train.json
# │   ├── split.rc2.val.json
# │   └── split.rc2.test1.json
# ├── test1/
# ├── dev/
# └── train/
#     ├── [numbered_directories_0-99]/
#     └── ...
# 
class CIRRDataset(Dataset):
    def __init__(
            self, 
            mode, 
            preprocess, 
            split: str='val', 
            path: str='.',
            augmenter: DataAugmentation = None
        ):
        super().__init__()
        assert split in ['train', 'test1', 'val']
        assert mode in ['query', 'target']
        self.path = path
        self.split = split
        self.mode = mode
        self.preprocess = preprocess
        self.augmenter = augmenter
        self.triplets = orjson.loads(open(os.path.join(path, f'captions/cap.rc2.{self.split}.json')).read())
        self.namepath = orjson.loads(open(os.path.join(path, f'image_splits/split.rc2.{split}.json')).read())


    def __getitem__(self, index):
        # index should not be batched
        if self.mode == 'target':
            img_name = list(self.namepath.keys())[index]
            img_rel_path = self.namepath[img_name]
            img_full_path = os.path.join(self.path, img_rel_path)
            img = self.preprocess(ensure_rgb(Image.open(img_full_path)))
            return {
                'img_name': img_name,
                'img': img
            }

        if self.mode == 'query':
            ref_img_name = self.triplets[index]['reference']
            ref_img_file = os.path.join(self.path, self.namepath[ref_img_name])
            ref_img = ensure_rgb(Image.open(ref_img_file))
            if self.augmenter:
                ref_img = self.augmenter.apply(ref_img)
            ref_img = self.preprocess(ref_img)
            caption = self.triplets[index]['caption']
             
            if self.split == 'train':
                # load and process target image for training
                tgt_img_name = self.triplets[index]['target_hard']
                tgt_img_file = os.path.join(self.path, self.namepath[tgt_img_name])
                tgt_img = ensure_rgb(Image.open(tgt_img_file))
                if self.augmenter:
                    tgt_img = self.augmenter.apply(tgt_img)
                tgt_img = self.preprocess(tgt_img)
                return {
                    'ref_img_name': ref_img_name,
                    'ref_img': ref_img,
                    'text_instruction': caption,
                    'tgt_img_name': tgt_img_name,
                    'tgt_img': tgt_img
                }
            elif self.split == 'val':
                # only need target image name during validation
                tgt_img_name = self.triplets[index]['target_hard']
                return {
                    'ref_img_name': ref_img_name,
                    'ref_img': ref_img,
                    'text_instruction': caption,
                    'tgt_img_name': tgt_img_name
                }
            elif self.split == 'test1':
                # we need pairid, subset, and subset names for test submission
                pairid = self.triplets[index]['pairid']
                subset_names = self.triplets[index]['img_set']['members']
                return {
                    'pairid': pairid,
                    'ref_img_name': ref_img_name,
                    'ref_img': ref_img,
                    'text_instruction': caption,
                    'subset_names': subset_names
                }

    def __len__(self):
        if self.mode == 'target':
            return len(self.namepath)
        if self.mode == 'query':
            return len(self.triplets)


class ITCPRCrossDomainDataset(Dataset):
    """Source-filtered ITCPR triplets for cross-domain training and evaluation."""

    SOURCES = ('Celeb-reID', 'LAST', 'PRCC')

    def __init__(
            self,
            path,
            mode,
            source,
            preprocess,
            split='val',
            augmenter=None,
        ):
        super().__init__()
        assert mode in ('query', 'target')
        assert split in ('train', 'val')
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}, got {source}")

        self.path = path
        self.mode = mode
        self.source = source
        self.preprocess = preprocess
        self.split = split
        self.augmenter = augmenter

        with open(os.path.join(path, 'query.json')) as f:
            queries = orjson.loads(f.read())
        with open(os.path.join(path, 'gallery.json')) as f:
            gallery = orjson.loads(f.read())

        self.queries = [row for row in queries if row['datasets'] == source]
        self.gallery = [row for row in gallery if row['datasets'] == source]

        positives_by_iid = defaultdict(list)
        for row in self.gallery:
            iid = int(row['instance_id'])
            if iid != -1:
                positives_by_iid[iid].append(row)
        self.positives_by_iid = positives_by_iid

        missing = [
            int(row['instance_id']) for row in self.queries
            if int(row['instance_id']) not in positives_by_iid
        ]
        if missing:
            raise ValueError(
                f"{source} contains {len(missing)} queries without a positive gallery target"
            )

    def _load_image(self, relative_path, augment=False):
        image = ensure_rgb(Image.open(os.path.join(self.path, relative_path)))
        if augment and self.augmenter:
            image = self.augmenter.apply(image)
        return self.preprocess(image)

    def __getitem__(self, index):
        if self.mode == 'target':
            row = self.gallery[index]
            return {
                'img_name': row['file_path'],
                'img': self._load_image(row['file_path']),
            }

        row = self.queries[index]
        positive = self.positives_by_iid[int(row['instance_id'])][0]
        result = {
            'ref_img_name': row['file_path'],
            'ref_img': self._load_image(
                row['file_path'], augment=self.split == 'train'
            ),
            'text_instruction': row['caption'],
            'tgt_img_name': positive['file_path'],
        }
        if self.split == 'train':
            result['tgt_img'] = self._load_image(
                positive['file_path'], augment=True
            )
        return result

    def __len__(self):
        return len(self.gallery) if self.mode == 'target' else len(self.queries)


def build_data(config, preprocess):
    """
    Build datasets and dataloaders based on configuration
    
    Args:
        config: Configuration object containing data and training parameters
        
    Returns:
        dict: Dictionary containing train_loader, val_loader, and target_loader
    """
    import torch
    
    # Setup data augmentation if specified
    augmenter = None
    if hasattr(config.data, 'augmentation') and config.data.augmentation.enabled:
        augmenter = DataAugmentation(methods=config.data.augmentation.methods)
    
    # Build datasets based on dataset type
    if config.data.dataset == 'fashioniq':
        train_dataset = FashionIQDataset(
            path=config.data.data_path,
            mode='query',
            split='train',
            clothtype=config.data.category,
            preprocess=preprocess,
            augmenter=augmenter
        )
        val_dataset = FashionIQDataset(
            path=config.data.data_path,
            mode='query',
            split='val',
            clothtype=config.data.category,
            preprocess=preprocess,
        )
        target_dataset = FashionIQDataset(
            path=config.data.data_path,
            mode='target',
            split='val',
            clothtype=config.data.category,
            preprocess=preprocess,
        )
        
        # Use appropriate collate functions for FashionIQ
        train_collate_fn = fiq_collate_fn_train
        val_collate_fn = fiq_collate_fn_val
        
    elif config.data.dataset == 'cirr':
        train_dataset = CIRRDataset(
            path=config.data.data_path,
            mode='query',
            split='train',
            preprocess=preprocess,
            augmenter=augmenter
        )
        val_dataset = CIRRDataset(
            path=config.data.data_path,
            mode='query',
            split='val',
            preprocess=preprocess,
        )
        target_dataset = CIRRDataset(
            path=config.data.data_path,
            mode='target',
            split='val',
            preprocess=preprocess,
        )
        
        # CIRR uses default collate function
        train_collate_fn = None
        val_collate_fn = None

    elif config.data.dataset == 'itcpr_cross_domain':
        train_source = getattr(config.data, 'train_source', 'Celeb-reID')
        eval_source = getattr(config.data, 'eval_source', 'LAST')
        train_dataset = ITCPRCrossDomainDataset(
            path=config.data.data_path,
            mode='query',
            source=train_source,
            split='train',
            preprocess=preprocess,
            augmenter=augmenter,
        )
        val_dataset = ITCPRCrossDomainDataset(
            path=config.data.data_path,
            mode='query',
            source=eval_source,
            split='val',
            preprocess=preprocess,
        )
        target_dataset = ITCPRCrossDomainDataset(
            path=config.data.data_path,
            mode='target',
            source=eval_source,
            split='val',
            preprocess=preprocess,
        )
        train_collate_fn = None
        val_collate_fn = None
        
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset}")
    
    # Build dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=train_collate_fn,
        drop_last=config.training.drop_last,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.validation.query_batch_size,
        shuffle=config.validation.shuffle,
        num_workers=config.validation.num_workers,
        pin_memory=config.validation.pin_memory,
        collate_fn=val_collate_fn,
        drop_last=False,
    )
    
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=config.validation.target_batch_size,
        shuffle=False,
        num_workers=config.validation.num_workers,
        pin_memory=config.validation.pin_memory,
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'target_loader': target_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'target_dataset': target_dataset,
    }
