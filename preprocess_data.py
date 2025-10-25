import os
import cv2
import random
import logging
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse
import json
import yaml
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration class for YOLO dataset preprocessing."""
    source_dir: Path = Path("data/reorganized_dataset")
    processed_dir: Path = Path("data/processed")
    img_size: Tuple[int, int] = (640, 640)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    quality: int = 95  # JPEG quality
    verify_images: bool = True
    copy_data: bool = True  # Copy instead of process if already in YOLO format

class YOLOPreprocessor:
    """
    Robust preprocessor for YOLO-formatted datasets.
    Handles datasets that are already organized in train/val/test splits.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._validate_config()
        self._setup_environment()
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.config.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.config.source_dir}")
            
        if any(x <= 0 for x in self.config.img_size):
            raise ValueError("Image dimensions must be positive integers")
            
        if not (0 <= self.config.quality <= 100):
            raise ValueError("JPEG quality must be between 0 and 100")
    
    def _setup_environment(self):
        """Setup processing environment and directories."""
        try:
            # Create output directories
            for split in ['train', 'val', 'test']:
                (self.config.processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
                (self.config.processed_dir / split / "labels").mkdir(parents=True, exist_ok=True)
                
            logger.info(f" Created output directories in: {self.config.processed_dir}")
            
        except Exception as e:
            logger.error(f" Failed to create output directories: {e}")
            raise

    def analyze_dataset_structure(self) -> Dict:
        """
        Analyze the existing YOLO dataset structure.
        
        Returns:
            Dict: Dataset structure information
        """
        structure_info = {
            'splits_found': [],
            'total_images': 0,
            'class_distribution': {},
            'split_details': {}
        }
        
        expected_splits = ['train', 'val', 'test']
        
        for split in expected_splits:
            split_path = self.config.source_dir / split
            images_path = split_path / "images"
            labels_path = split_path / "labels"
            
            if images_path.exists() and labels_path.exists():
                structure_info['splits_found'].append(split)
                
                # Count images
                image_files = list(images_path.glob("*.*"))
                image_count = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                
                # Count labels
                label_files = list(labels_path.glob("*.txt"))
                
                structure_info['split_details'][split] = {
                    'images': image_count,
                    'labels': len(label_files),
                    'image_files': [f.name for f in image_files[:5]],  # Sample filenames
                    'label_files': [f.name for f in label_files[:5]]   # Sample filenames
                }
                structure_info['total_images'] += image_count
                
                logger.info(f" Found {split} split: {image_count} images, {len(label_files)} labels")
            else:
                logger.warning(f" {split} split not found or incomplete")
        
        if not structure_info['splits_found']:
            raise ValueError("No valid YOLO dataset splits found! Expected structure: data/split/images/ and data/split/labels/")
        
        return structure_info

    def validate_image_label_pair(self, image_path: Path, label_path: Path) -> bool:
        """
        Validate that image and label files are properly paired.
        
        Args:
            image_path: Path to image file
            label_path: Path to label file
            
        Returns:
            bool: True if valid pair
        """
        try:
            # Check if image exists and is readable
            if not image_path.exists():
                return False
                
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Check if label exists
            if not label_path.exists():
                logger.warning(f" Label file missing for image: {image_path.name}")
                return False
            
            # Validate label format
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.warning(f" Invalid label format in {label_path.name}: {line}")
                        return False
                    # Validate class ID and coordinates
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if not (0 <= class_id <= 1000):  # Reasonable class ID range
                        return False
                    for coord in coords:
                        if not (0 <= coord <= 1):  # YOLO coordinates should be normalized
                            logger.warning(f" Invalid coordinates in {label_path.name}: {coords}")
                            return False
            
            return True
            
        except Exception as e:
            logger.warning(f" Error validating pair {image_path.name}: {e}")
            return False

    def process_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Process a single image (resize and save with quality).
        
        Args:
            image_path: Source image path
            output_path: Output image path
            
        Returns:
            bool: True if successful
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Resize image
            img_resized = cv2.resize(img, self.config.img_size)
            
            # Save with specified quality
            success = cv2.imwrite(
                str(output_path), 
                img_resized, 
                [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
            )
            
            return success
            
        except Exception as e:
            logger.error(f" Error processing image {image_path.name}: {e}")
            return False

    def copy_label(self, label_path: Path, output_path: Path) -> bool:
        """
        Copy label file to output directory.
        
        Args:
            label_path: Source label path
            output_path: Output label path
            
        Returns:
            bool: True if successful
        """
        try:
            shutil.copy2(label_path, output_path)
            return True
        except Exception as e:
            logger.error(f" Error copying label {label_path.name}: {e}")
            return False

    def process_split(self, split_name: str) -> Dict:
        """
        Process a single split (train/val/test).
        
        Args:
            split_name: Name of the split to process
            
        Returns:
            Dict: Processing statistics
        """
        stats = {
            'images_processed': 0,
            'images_failed': 0,
            'labels_processed': 0,
            'labels_failed': 0,
            'invalid_pairs': 0
        }
        
        source_split_path = self.config.source_dir / split_name
        source_images_path = source_split_path / "images"
        source_labels_path = source_split_path / "labels"
        
        output_images_path = self.config.processed_dir / split_name / "images"
        output_labels_path = self.config.processed_dir / split_name / "labels"
        
        if not source_images_path.exists():
            logger.warning(f" Source images path not found: {source_images_path}")
            return stats
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(source_images_path.glob(f"*{ext}"))
            image_files.extend(source_images_path.glob(f"*{ext.upper()}"))
        
        logger.info(f" Processing {split_name} split: {len(image_files)} images")
        
        for image_path in tqdm(image_files, desc=f"Processing {split_name}"):
            try:
                # Corresponding label path
                label_name = image_path.with_suffix('.txt').name
                label_path = source_labels_path / label_name
                
                # Output paths
                output_image_path = output_images_path / image_path.name
                output_label_path = output_labels_path / label_name
                
                # Validate pair
                if self.config.verify_images and not self.validate_image_label_pair(image_path, label_path):
                    stats['invalid_pairs'] += 1
                    continue
                
                # Process image
                if self.process_image(image_path, output_image_path):
                    stats['images_processed'] += 1
                else:
                    stats['images_failed'] += 1
                    continue
                
                # Copy label
                if label_path.exists():
                    if self.copy_label(label_path, output_label_path):
                        stats['labels_processed'] += 1
                    else:
                        stats['labels_failed'] += 1
                else:
                    stats['labels_failed'] += 1
                    logger.warning(f" Label file not found: {label_path}")
                
            except Exception as e:
                logger.error(f" Error processing {image_path.name}: {e}")
                stats['images_failed'] += 1
        
        return stats

    def save_processing_metadata(self, structure_info: Dict, processing_stats: Dict):
        """
        Save processing metadata and statistics.
        
        Args:
            structure_info: Dataset structure information
            processing_stats: Processing statistics
        """
        metadata = {
            'config': {
                'source_dir': str(self.config.source_dir),
                'processed_dir': str(self.config.processed_dir),
                'img_size': self.config.img_size,
                'train_split': self.config.train_split,
                'val_split': self.config.val_split,
                'test_split': self.config.test_split,
                'random_seed': self.config.random_seed,
                'quality': self.config.quality
            },
            'dataset_structure': structure_info,
            'processing_stats': processing_stats,
            'environment': {
                'opencv_version': cv2.__version__,
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Save as JSON
        metadata_path = self.config.processed_dir / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f" Processing metadata saved to: {metadata_path}")

    def process_dataset(self) -> bool:
        """
        Main dataset processing pipeline for YOLO-formatted data.
        
        Returns:
            bool: True if processing successful
        """
        try:
            logger.info(" Starting YOLO dataset preprocessing...")
            
            # Analyze dataset structure
            structure_info = self.analyze_dataset_structure()
            
            # Process each split
            processing_stats = {}
            
            for split_name in structure_info['splits_found']:
                logger.info(f"\n Processing {split_name} split...")
                stats = self.process_split(split_name)
                processing_stats[split_name] = stats
                
                logger.info(f" {split_name} completed:")
                logger.info(f"  - Images: {stats['images_processed']} processed, {stats['images_failed']} failed")
                logger.info(f"  - Labels: {stats['labels_processed']} processed, {stats['labels_failed']} failed")
                logger.info(f"  - Invalid pairs: {stats['invalid_pairs']}")
            
            # Save processing metadata
            self.save_processing_metadata(structure_info, processing_stats)
            
            # Summary
            total_images = sum(stats['images_processed'] for stats in processing_stats.values())
            total_failed = sum(stats['images_failed'] for stats in processing_stats.values())
            
            logger.info(f"\n Preprocessing completed!")
            logger.info(f" Summary:")
            logger.info(f"  - Total images processed: {total_images}")
            logger.info(f"  - Total images failed: {total_failed}")
            logger.info(f"  - Output directory: {self.config.processed_dir}")
            logger.info(f"  - Splits processed: {', '.join(structure_info['splits_found'])}")
            
            return True
            
        except Exception as e:
            logger.error(f" Preprocessing failed: {e}")
            return False

def main():
    """Main function for YOLO dataset preprocessing."""
    parser = argparse.ArgumentParser(description="Robust YOLO Dataset Preprocessing")
    parser.add_argument('--source-dir', default='data/reorganized_dataset', 
                       help='Source directory with YOLO format (default: data/reorganized_dataset)')
    parser.add_argument('--output-dir', default='data/processed', 
                       help='Output directory (default: data/processed)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640], 
                       help='Image size (default: 640 640)')
    parser.add_argument('--quality', type=int, default=95, 
                       help='JPEG quality (0-100, default: 95)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    parser.add_argument('--no-verify', action='store_true', 
                       help='Disable image-label pair verification')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*60)
    print("Robust YOLO Dataset Preprocessing")
    print("="*60)
    
    try:
        # Create configuration
        config = PreprocessingConfig(
            source_dir=Path(args.source_dir),
            processed_dir=Path(args.output_dir),
            img_size=tuple(args.img_size),
            quality=args.quality,
            random_seed=args.seed,
            verify_images=not args.no_verify
        )
        
        # Display configuration
        print(f" Source directory: {config.source_dir}")
        print(f" Output directory: {config.processed_dir}")
        print(f"  Image size: {config.img_size}")
        print(f"  Quality: {config.quality}, Seed: {config.random_seed}")
        print(f" Verification: {config.verify_images}")
        print("="*60)
        
        # Initialize and run preprocessor
        preprocessor = YOLOPreprocessor(config)
        success = preprocessor.process_dataset()
        
        if success:
            print("\n YOLO dataset preprocessing completed successfully!")
            print(f" Processed data stored in: {config.processed_dir}")
            return 0
        else:
            print("\n Preprocessing failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n Preprocessing cancelled by user")
        return 1
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        logger.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    exit(main())