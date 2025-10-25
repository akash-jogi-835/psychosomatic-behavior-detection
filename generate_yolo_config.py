import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple  # Added Tuple import
from dataclasses import dataclass
import sys
import json

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class YOLOConfig:
    """Configuration class for YOLO dataset configuration generation."""
    processed_dir: Path = Path("data/processed")
    output_path: Path = Path("data/processed/data.yaml")
    class_names: List[str] = None
    auto_discover_classes: bool = True
    nc: Optional[int] = None  # Force number of classes
    verify_directories: bool = True
    create_backup: bool = True
    default_classes: List[str] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['person', 'face', 'phone']  # Default classes
        if self.default_classes is None:
            self.default_classes = ['person', 'face', 'phone']

class YOLOConfigGenerator:
    """
    Robust YOLO configuration generator with comprehensive validation.
    """
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.config.processed_dir.exists():
            raise ValueError(f"Processed directory does not exist: {self.config.processed_dir}")
            
        if len(self.config.class_names) == 0:
            raise ValueError("Class names list cannot be empty")
            
        if self.config.nc is not None and self.config.nc <= 0:
            raise ValueError("Number of classes (nc) must be positive")
    
    def discover_classes_from_labels(self, labels_dir: Path) -> List[str]:
        """
        Discover class names from label files automatically.
        
        Args:
            labels_dir: Path to labels directory
            
        Returns:
            List of discovered class names
        """
        class_ids = set()
        
        try:
            if not labels_dir.exists():
                logger.warning(f"⚠️ Labels directory not found: {labels_dir}")
                return self.config.default_classes
            
            label_files = list(labels_dir.glob("*.txt"))
            if not label_files:
                logger.warning(f"⚠️ No label files found in: {labels_dir}")
                return self.config.default_classes
            
            logger.info(f"🔍 Discovering classes from {len(label_files)} label files...")
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 1:
                                    class_id = int(parts[0])
                                    class_ids.add(class_id)
                except Exception as e:
                    logger.warning(f"⚠️ Error reading label file {label_file.name}: {e}")
                    continue
            
            # Convert class IDs to names
            if class_ids:
                max_class_id = max(class_ids)
                discovered_classes = [f"class_{i}" for i in range(max_class_id + 1)]
                logger.info(f"✅ Discovered {len(discovered_classes)} classes: IDs {sorted(class_ids)}")
                return discovered_classes
            else:
                logger.warning("⚠️ No class IDs found in label files")
                return self.config.default_classes
                
        except Exception as e:
            logger.error(f"❌ Error discovering classes: {e}")
            return self.config.default_classes
    
    def verify_dataset_structure(self) -> Dict[str, Any]:
        """
        Verify the dataset structure and collect statistics.
        
        Returns:
            Dictionary with dataset structure information
        """
        structure_info = {
            'splits_exist': {},
            'image_counts': {},
            'label_counts': {},
            'missing_directories': [],
            'warnings': []
        }
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.config.processed_dir / split / "images"
            labels_dir = self.config.processed_dir / split / "labels"
            
            split_info = {
                'images_exists': images_dir.exists(),
                'labels_exists': labels_dir.exists(),
                'image_count': 0,
                'label_count': 0
            }
            
            structure_info['splits_exist'][split] = split_info
            
            if images_dir.exists():
                # Count image files
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend(images_dir.glob(f"*{ext}"))
                    image_files.extend(images_dir.glob(f"*{ext.upper()}"))
                split_info['image_count'] = len(image_files)
                structure_info['image_counts'][split] = len(image_files)
            else:
                structure_info['missing_directories'].append(f"{split}/images")
                structure_info['warnings'].append(f"Missing directory: {split}/images")
            
            if labels_dir.exists():
                # Count label files
                label_files = list(labels_dir.glob("*.txt"))
                split_info['label_count'] = len(label_files)
                structure_info['label_counts'][split] = len(label_files)
            else:
                structure_info['missing_directories'].append(f"{split}/labels")
                structure_info['warnings'].append(f"Missing directory: {split}/labels")
            
            # Check for imbalance
            if split_info['image_count'] != split_info['label_count']:
                structure_info['warnings'].append(
                    f"Imbalance in {split}: {split_info['image_count']} images vs {split_info['label_count']} labels"
                )
        
        return structure_info
    
    def auto_discover_classes(self) -> List[str]:
        """
        Automatically discover class names from the dataset.
        
        Returns:
            List of class names
        """
        if not self.config.auto_discover_classes:
            logger.info("🔧 Using manually specified class names")
            return self.config.class_names
        
        # Try to discover from train labels first, then val, then test
        for split in ['train', 'val', 'test']:
            labels_dir = self.config.processed_dir / split / "labels"
            if labels_dir.exists():
                discovered_classes = self.discover_classes_from_labels(labels_dir)
                if discovered_classes and discovered_classes != self.config.default_classes:
                    logger.info(f"✅ Using discovered classes from {split} split")
                    return discovered_classes
        
        logger.warning("⚠️ Could not auto-discover classes, using defaults")
        return self.config.default_classes
    
    def create_backup(self, file_path: Path) -> bool:
        """
        Create a backup of existing configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            bool: True if backup created successfully
        """
        if not file_path.exists():
            return True
        
        try:
            backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
            counter = 1
            while backup_path.exists():
                backup_path = file_path.with_suffix(f".backup{counter}{file_path.suffix}")
                counter += 1
            
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"💾 Backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False
    
    def validate_class_names(self, class_names: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate class names for YOLO compatibility.
        
        Args:
            class_names: List of class names to validate
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        
        # Check for empty names
        if any(not name.strip() for name in class_names):
            warnings.append("Empty class names found")
        
        # Check for duplicates
        if len(class_names) != len(set(class_names)):
            warnings.append("Duplicate class names found")
        
        # Check for special characters (basic check)
        special_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for name in class_names:
            if any(char in name for char in special_chars):
                warnings.append(f"Class name '{name}' contains special characters")
        
        # Check length
        for name in class_names:
            if len(name) > 50:
                warnings.append(f"Class name '{name}' is too long (>50 chars)")
        
        return len(warnings) == 0, warnings
    
    def generate_yolo_config(self) -> Dict[str, Any]:
        """
        Generate a YOLOv8-compatible dataset configuration file.
        
        Returns:
            Dictionary with generation results
        """
        result = {
            'success': False,
            'output_path': None,
            'warnings': [],
            'errors': [],
            'config_data': None
        }
        
        try:
            logger.info("🚀 Generating YOLO dataset configuration...")
            
            # Verify dataset structure
            structure_info = self.verify_dataset_structure()
            result['structure_info'] = structure_info
            
            # Add warnings from structure verification
            result['warnings'].extend(structure_info['warnings'])
            
            # Auto-discover classes if enabled
            if self.config.auto_discover_classes:
                self.config.class_names = self.auto_discover_classes()
            
            # Validate class names
            classes_valid, class_warnings = self.validate_class_names(self.config.class_names)
            result['warnings'].extend(class_warnings)
            
            if not classes_valid:
                result['errors'].append("Class name validation failed")
            
            # Determine number of classes
            if self.config.nc is not None:
                nc = self.config.nc
                if nc != len(self.config.class_names):
                    result['warnings'].append(
                        f"Specified nc ({nc}) doesn't match class names count ({len(self.config.class_names)})"
                    )
            else:
                nc = len(self.config.class_names)
            
            # Prepare configuration data
            config_data = {
                "path": str(self.config.processed_dir.parent),  # Dataset root dir
                "train": "train/images",
                "val": "val/images", 
                "test": "test/images",
                "nc": nc,
                "names": {i: name for i, name in enumerate(self.config.class_names)}
            }
            
            # Clean up empty entries for missing splits
            for split in ['train', 'val', 'test']:
                split_dir = self.config.processed_dir / split / "images"
                if not split_dir.exists():
                    config_data[split] = ""
                    result['warnings'].append(f"Missing split: {split}")
            
            config_data = {k: v for k, v in config_data.items() if v not in ["", None]}
            
            result['config_data'] = config_data
            
            # Create backup if requested
            if self.config.create_backup and self.config.output_path.exists():
                self.create_backup(self.config.output_path)
            
            # Ensure output directory exists
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration file
            with open(self.config.output_path, "w", encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            result['output_path'] = str(self.config.output_path)
            result['success'] = True
            
            logger.info(f"✅ YOLO configuration generated successfully: {self.config.output_path}")
            
        except Exception as e:
            result['errors'].append(f"Configuration generation failed: {e}")
            logger.error(f"❌ Error generating YOLO configuration: {e}")
        
        return result
    
    def save_generation_report(self, result: Dict[str, Any]):
        """
        Save generation report for debugging and documentation.
        
        Args:
            result: Generation result dictionary
        """
        try:
            report_path = self.config.output_path.with_suffix('.report.json')
            report = {
                'generation_result': result,
                'config_used': {
                    'processed_dir': str(self.config.processed_dir),
                    'output_path': str(self.config.output_path),
                    'class_names': self.config.class_names,
                    'auto_discover_classes': self.config.auto_discover_classes,
                    'nc': self.config.nc
                },
                'environment': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📝 Generation report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save generation report: {e}")
    
    def print_summary(self, result: Dict[str, Any]):
        """
        Print a human-readable summary of the configuration generation.
        
        Args:
            result: Generation result dictionary
        """
        print("\n" + "="*60)
        print("🎯 YOLO DATASET CONFIGURATION SUMMARY")
        print("="*60)
        
        if result['success']:
            print(f"✅ Configuration generated successfully!")
            print(f"📁 Output: {result['output_path']}")
            
            config_data = result['config_data']
            print(f"📊 Configuration:")
            print(f"   • Number of classes: {config_data['nc']}")
            print(f"   • Classes: {list(config_data['names'].values())}")
            print(f"   • Train path: {config_data.get('train', 'Not found')}")
            print(f"   • Val path: {config_data.get('val', 'Not found')}")
            print(f"   • Test path: {config_data.get('test', 'Not found')}")
            
            # Show structure info
            structure = result.get('structure_info', {})
            if structure.get('image_counts'):
                print(f"\n📈 Dataset Statistics:")
                for split in ['train', 'val', 'test']:
                    if split in structure['image_counts']:
                        print(f"   • {split.upper()}: {structure['image_counts'][split]} images, "
                              f"{structure['label_counts'][split]} labels")
            
        else:
            print(f"❌ Configuration generation failed!")
            for error in result.get('errors', []):
                print(f"   • Error: {error}")
        
        # Show warnings
        if result.get('warnings'):
            print(f"\n⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"   • {warning}")
        
        # Show next steps
        if result['success']:
            print(f"\n🔜 Next Steps:")
            print(f"   1. Verify the configuration at: {result['output_path']}")
            print(f"   2. Use this config in your YOLO training command:")
            print(f"      'yolo train data={result['output_path']} model=yolov8n.pt'")
        
        print("="*60)

def main():
    """Main function for YOLO configuration generation."""
    parser = argparse.ArgumentParser(description="Robust YOLOv8 Dataset Configuration Generator")
    parser.add_argument("--processed-dir", default="data/processed", 
                       help="Path to processed dataset (default: data/processed)")
    parser.add_argument("--output", default="data/processed/data.yaml", 
                       help="Path to save YAML config (default: data/processed/data.yaml)")
    parser.add_argument("--classes", nargs="+", 
                       help="List of class names (overrides auto-discovery)")
    parser.add_argument("--nc", type=int, 
                       help="Force number of classes (overrides auto-calculation)")
    parser.add_argument("--no-auto-discover", action="store_true", 
                       help="Disable auto class discovery")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Disable backup of existing config")
    parser.add_argument("--no-verify", action="store_true", 
                       help="Disable directory verification")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create configuration
        config = YOLOConfig(
            processed_dir=Path(args.processed_dir),
            output_path=Path(args.output),
            class_names=args.classes if args.classes else None,
            auto_discover_classes=not args.no_auto_discover,
            nc=args.nc,
            verify_directories=not args.no_verify,
            create_backup=not args.no_backup
        )
        
        print("\n" + "="*60)
        print("🔧 YOLO Dataset Configuration Generator")
        print("="*60)
        print(f"📁 Processed dir: {config.processed_dir}")
        print(f"📂 Output: {config.output_path}")
        print(f"🔍 Auto-discover: {config.auto_discover_classes}")
        print(f"💾 Backup: {config.create_backup}")
        print("="*60)
        
        # Initialize and run generator
        generator = YOLOConfigGenerator(config)
        result = generator.generate_yolo_config()
        
        # Save report
        generator.save_generation_report(result)
        
        # Print summary
        generator.print_summary(result)
        
        return 0 if result['success'] else 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Configuration generation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    exit(main())