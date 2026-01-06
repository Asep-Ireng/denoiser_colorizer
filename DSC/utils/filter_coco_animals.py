"""
COCO Dataset Animal Image Filter Script
Filters animal images from COCO 2017 training set for denoiser-colorizer model.

NOTE: You only need to download COCO 2017 Train images (118K images, 18GB).
      This script splits those images into train/val/test sets automatically.

Downloads required:
  - Images: http://images.cocodataset.org/zips/train2017.zip (18GB)
  - Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip (241MB)

Usage:
    # Step 1: Generate split JSON files
    python filter_coco_animals.py --annotations path/to/instances_train2017.json --output_dir ./data
    
    # Step 2: Copy images to train/val/test folders
    python filter_coco_animals.py --annotations path/to/instances_train2017.json \\
        --images_dir path/to/train2017 --output_dir ./data --copy_images
"""

import json
import os
import random
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# COCO Animal Category IDs (supercategory: 'animal')
# Reference: https://cocodataset.org/#explore
ANIMAL_CATEGORY_IDS = {
    # Birds
    14: 'bird',
    # Mammals
    15: 'cat',
    16: 'dog', 
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
}

# Default split configuration (total 15,000 images)
DEFAULT_TRAIN = 12000
DEFAULT_VAL = 0      # Optional validation set
DEFAULT_TEST = 3000


def load_annotations(annotation_file: str) -> dict:
    """Load COCO annotation JSON file."""
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def get_animal_image_ids(annotations: dict) -> set:
    """
    Extract image IDs that contain at least one animal annotation.
    
    Args:
        annotations: COCO annotations dictionary
        
    Returns:
        Set of image IDs containing animals
    """
    animal_image_ids = set()
    category_counts = defaultdict(int)
    
    print("Filtering animal annotations...")
    for ann in tqdm(annotations['annotations']):
        cat_id = ann['category_id']
        if cat_id in ANIMAL_CATEGORY_IDS:
            animal_image_ids.add(ann['image_id'])
            category_counts[ANIMAL_CATEGORY_IDS[cat_id]] += 1
    
    print("\nAnimal category distribution:")
    for cat_name, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat_name}: {count} annotations")
    
    return animal_image_ids


def create_image_id_to_filename_map(annotations: dict) -> dict:
    """Create mapping from image ID to filename."""
    return {img['id']: img['file_name'] for img in annotations['images']}


def split_dataset(image_ids: list, train_count: int, val_count: int, test_count: int, seed: int = 42) -> tuple:
    """
    Randomly split image IDs into train, val, and test sets.
    
    Args:
        image_ids: List of image IDs
        train_count: Number of training images
        val_count: Number of validation images (can be 0)
        test_count: Number of test images
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    random.seed(seed)
    shuffled = image_ids.copy()
    random.shuffle(shuffled)
    
    total_needed = train_count + val_count + test_count
    if len(shuffled) < total_needed:
        print(f"Warning: Only {len(shuffled)} images available, need {total_needed}")
        # Proportionally adjust
        ratio = len(shuffled) / total_needed
        train_count = int(train_count * ratio)
        val_count = int(val_count * ratio)
        test_count = len(shuffled) - train_count - val_count
        
    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count:train_count + val_count]
    test_ids = shuffled[train_count + val_count:train_count + val_count + test_count]
    
    return train_ids, val_ids, test_ids


def save_split_info(train_ids: list, val_ids: list, test_ids: list, output_dir: str, id_to_filename: dict):
    """Save train/val/test split information to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, ids in splits.items():
        if len(ids) == 0:
            continue
        split_info = {
            'count': len(ids),
            'image_ids': ids, 
            'filenames': [id_to_filename[img_id] for img_id in ids]
        }
        with open(output_path / f'{split_name}_animals.json', 'w') as f:
            json.dump(split_info, f, indent=2)
    
    print(f"\nSaved split info to {output_path}")
    print(f"  Train: {len(train_ids)} images -> train_animals.json")
    if len(val_ids) > 0:
        print(f"  Val:   {len(val_ids)} images -> val_animals.json")
    print(f"  Test:  {len(test_ids)} images -> test_animals.json")


def copy_images(image_ids: list, id_to_filename: dict, 
                source_dir: str, dest_dir: str, split_name: str):
    """
    Copy images from source to destination directory.
    
    Args:
        image_ids: List of image IDs to copy
        id_to_filename: Mapping from ID to filename
        source_dir: Source directory containing COCO images
        dest_dir: Destination directory
        split_name: 'train', 'val', or 'test'
    """
    if len(image_ids) == 0:
        return
        
    dest_path = Path(dest_dir) / split_name
    dest_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_dir)
    
    print(f"\nCopying {split_name} images...")
    copied = 0
    missing = 0
    
    for img_id in tqdm(image_ids):
        filename = id_to_filename[img_id]
        src_file = source_path / filename
        
        if src_file.exists():
            shutil.copy2(src_file, dest_path / filename)
            copied += 1
        else:
            missing += 1
    
    print(f"  Copied: {copied}, Missing: {missing}")


def main():
    parser = argparse.ArgumentParser(description='Filter COCO animal images from train2017 and split into train/val/test')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to COCO annotations JSON (e.g., instances_train2017.json)')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Path to COCO images directory (e.g., train2017/)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for split info and images')
    parser.add_argument('--train_count', type=int, default=DEFAULT_TRAIN,
                        help=f'Number of training images (default: {DEFAULT_TRAIN})')
    parser.add_argument('--val_count', type=int, default=DEFAULT_VAL,
                        help=f'Number of validation images (default: {DEFAULT_VAL})')
    parser.add_argument('--test_count', type=int, default=DEFAULT_TEST,
                        help=f'Number of test images (default: {DEFAULT_TEST})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to output directory')
    
    args = parser.parse_args()
    
    # Load COCO annotations
    annotations = load_annotations(args.annotations)
    
    # Get animal image IDs
    animal_image_ids = get_animal_image_ids(annotations)
    print(f"\nTotal images with animals: {len(animal_image_ids)}")
    
    # Create ID to filename mapping
    id_to_filename = create_image_id_to_filename_map(annotations)
    
    # Split into train/val/test
    animal_list = list(animal_image_ids)
    train_ids, val_ids, test_ids = split_dataset(
        animal_list, 
        args.train_count,
        args.val_count,
        args.test_count,
        args.seed
    )
    
    # Save split information
    save_split_info(train_ids, val_ids, test_ids, args.output_dir, id_to_filename)
    
    # Optionally copy images
    if args.copy_images and args.images_dir:
        copy_images(train_ids, id_to_filename, args.images_dir, args.output_dir, 'train')
        copy_images(val_ids, id_to_filename, args.images_dir, args.output_dir, 'val')
        copy_images(test_ids, id_to_filename, args.images_dir, args.output_dir, 'test')
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)
    
    if not args.copy_images:
        print(f"\nNext steps:")
        print(f"  1. Download COCO 2017 Train images: http://images.cocodataset.org/zips/train2017.zip")
        print(f"  2. Run again with --copy_images --images_dir <path_to_train2017> to copy filtered images")


if __name__ == '__main__':
    main()

