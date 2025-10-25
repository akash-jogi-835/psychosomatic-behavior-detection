import os
import cv2
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
import yaml

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class VerificationConfig:
    """Configuration class for dataset verification."""
    processed_dir: Path = Path("data/processed")
    supported_image_exts: Tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    check_image_integrity: bool = True
    check_label_format: bool = True
    check_dimensions: bool = True
    expected_img_size: Optional[Tuple[int, int]] = None
    max_invalid_files: int = 1000  # Limit invalid files logging
    save_detailed_report: bool = True

class YOLODatasetVerifier:
    """
    Robust YOLO dataset verification with comprehensive checks.
    """
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.config.processed_dir.exists():
            raise ValueError(f"Processed directory does not exist: {self.config.processed_dir}")
            
        if self.config.max_invalid_files <= 0:
            raise ValueError("max_invalid_files must be positive")
    
    def verify_yolo_label_file(self, label_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Verify the format and values in a YOLO label file with comprehensive checks.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not label_path.exists():
                return False, "Label file does not exist"
            
            # Check file size (empty files are invalid)
            if label_path.stat().st_size == 0:
                return False, "Label file is empty"
            
            with open(label_path, "r", encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Check for maximum reasonable number of objects
            if len(lines) > 1000:
                return False, f"Excessive objects ({len(lines)}) in label file"

            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                
                # Check number of components (should be 5 for YOLO format)
                if len(parts) != 5:
                    return False, f"Line {line_num}: Invalid format, expected 5 values, got {len(parts)}"

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError as e:
                    return False, f"Line {line_num}: Invalid number format - {e}"

                # Check class ID validity
                if class_id < 0:
                    return False, f"Line {line_num}: Invalid class ID ({class_id}), must be >= 0"

                # Check coordinate validity
                if len(coords) != 4:
                    return False, f"Line {line_num}: Expected 4 coordinates, got {len(coords)}"

                # Check normalized coordinate ranges
                for i, coord in enumerate(coords):
                    if not (0 <= coord <= 1):
                        coord_name = ['x_center', 'y_center', 'width', 'height'][i]
                        return False, f"Line {line_num}: {coord_name} ({coord}) out of range [0,1]"

                # Check bounding box dimensions
                if coords[2] <= 0 or coords[3] <= 0:  # width and height
                    return False, f"Line {line_num}: Invalid bounding box dimensions (w:{coords[2]}, h:{coords[3]})"

                # Check if bounding box is too small (potential annotation error)
                if coords[2] < 0.01 or coords[3] < 0.01:
                    logger.debug(f" Very small bounding box in {label_path.name}: w={coords[2]:.4f}, h={coords[3]:.4f}")

            return True, None

        except UnicodeDecodeError:
            return False, "Label file encoding error (not UTF-8)"
        except Exception as e:
            return False, f"Unexpected error reading label file: {e}"
    
    def verify_image_file(self, image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
        """
        Verify image file integrity and properties.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message, image_dimensions)
        """
        try:
            if not image_path.exists():
                return False, "Image file does not exist", None
            
            # Check file size
            file_size = image_path.stat().st_size
            if file_size == 0:
                return False, "Image file is empty", None
            if file_size < 100:  # Unreasonably small for an image
                return False, f"Image file too small ({file_size} bytes)", None

            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "OpenCV cannot read image file", None
            
            # Check image dimensions
            height, width = img.shape[:2]
            if width <= 0 or height <= 0:
                return False, f"Invalid image dimensions ({width}x{height})", (width, height)
            
            # Check for expected image size
            if self.config.expected_img_size and self.config.check_dimensions:
                expected_width, expected_height = self.config.expected_img_size
                if (width, height) != (expected_width, expected_height):
                    return False, f"Unexpected image size ({width}x{height}), expected ({expected_width}x{expected_height})", (width, height)
            
            # Check image data integrity
            if img.size == 0:
                return False, "Image data is empty", (width, height)
            
            # Check for corrupted image (all zeros or all same value)
            if img.std() == 0:
                return False, "Image appears corrupted (no variance in pixel values)", (width, height)

            return True, None, (width, height)

        except Exception as e:
            return False, f"Unexpected error reading image: {e}", None
    
    def verify_image_label_pair(self, image_path: Path, label_path: Path) -> Dict[str, Any]:
        """
        Verify an image-label pair with comprehensive checks.
        
        Args:
            image_path: Path to image file
            label_path: Path to label file
            
        Returns:
            Dictionary with verification results
        """
        result = {
            'valid': False,
            'image_valid': False,
            'label_valid': False,
            'errors': [],
            'image_dimensions': None,
            'object_count': 0
        }
        
        # Verify image
        if self.config.check_image_integrity:
            img_valid, img_error, img_dims = self.verify_image_file(image_path)
            result['image_valid'] = img_valid
            result['image_dimensions'] = img_dims
            if not img_valid:
                result['errors'].append(f"Image: {img_error}")
        else:
            result['image_valid'] = image_path.exists()
            if not result['image_valid']:
                result['errors'].append("Image file missing")
        
        # Verify label
        if self.config.check_label_format:
            label_valid, label_error = self.verify_yolo_label_file(label_path)
            result['label_valid'] = label_valid
            if not label_valid:
                result['errors'].append(f"Label: {label_error}")
            else:
                # Count objects in valid label
                try:
                    with open(label_path, 'r') as f:
                        result['object_count'] = len([line for line in f if line.strip()])
                except:
                    result['object_count'] = 0
        else:
            result['label_valid'] = label_path.exists()
            if not result['label_valid']:
                result['errors'].append("Label file missing")
        
        # Overall validity
        result['valid'] = result['image_valid'] and result['label_valid']
        
        return result
    
    def verify_split(self, split_path: Path) -> Dict[str, Any]:
        """
        Verify image-label pairs within a split (train/val/test).
        
        Args:
            split_path: Path to the split directory
            
        Returns:
            Dictionary with split verification statistics
        """
        split_stats = {
            "split_name": split_path.name,
            "total_images": 0,
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "missing_labels": 0,
            "invalid_images": 0,
            "invalid_labels": 0,
            "image_dimensions": {},
            "object_counts": [],
            "invalid_files": [],
            "split_exists": False
        }
        
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        # Check if split directories exist
        if not images_path.exists() or not labels_path.exists():
            logger.warning(f" Missing folders in split: {split_path}")
            split_stats['errors'] = ["Missing images or labels directory"]
            return split_stats
        
        split_stats['split_exists'] = True
        
        # Find image files
        image_files = []
        for ext in self.config.supported_image_exts:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        split_stats["total_images"] = len(image_files)
        
        if split_stats["total_images"] == 0:
            logger.warning(f" No images found in split: {split_path}")
            return split_stats
        
        logger.info(f" Verifying {split_path.name} split: {split_stats['total_images']} images")
        
        for image_path in tqdm(image_files, desc=f"Checking {split_path.name}"):
            label_path = labels_path / f"{image_path.stem}.txt"
            
            # Basic file existence check
            if not label_path.exists():
                split_stats["missing_labels"] += 1
                split_stats["invalid_pairs"] += 1
                if len(split_stats["invalid_files"]) < self.config.max_invalid_files:
                    split_stats["invalid_files"].append({
                        "file": str(image_path), 
                        "reason": "Label file missing",
                        "type": "missing_label"
                    })
                continue
            
            # Comprehensive pair verification
            pair_result = self.verify_image_label_pair(image_path, label_path)
            
            if pair_result['valid']:
                split_stats["valid_pairs"] += 1
                # Track image dimensions
                if pair_result['image_dimensions']:
                    dims_str = f"{pair_result['image_dimensions'][0]}x{pair_result['image_dimensions'][1]}"
                    split_stats["image_dimensions"][dims_str] = split_stats["image_dimensions"].get(dims_str, 0) + 1
                # Track object counts
                split_stats["object_counts"].append(pair_result['object_count'])
            else:
                split_stats["invalid_pairs"] += 1
                
                if not pair_result['image_valid']:
                    split_stats["invalid_images"] += 1
                if not pair_result['label_valid']:
                    split_stats["invalid_labels"] += 1
                
                if len(split_stats["invalid_files"]) < self.config.max_invalid_files:
                    split_stats["invalid_files"].append({
                        "file": str(image_path),
                        "reason": "; ".join(pair_result['errors']),
                        "type": "invalid_pair"
                    })
        
        # Calculate additional statistics
        if split_stats["object_counts"]:
            split_stats["object_stats"] = {
                "min": min(split_stats["object_counts"]),
                "max": max(split_stats["object_counts"]),
                "mean": sum(split_stats["object_counts"]) / len(split_stats["object_counts"]),
                "total": sum(split_stats["object_counts"])
            }
        
        return split_stats
    
    def generate_summary_report(self, overall_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Args:
            overall_stats: Statistics from all splits
            
        Returns:
            Dictionary with summary information
        """
        splits_data = overall_stats["splits"]
        
        summary = {
            "total_images": 0,
            "total_valid_pairs": 0,
            "total_invalid_pairs": 0,
            "total_objects": 0,
            "split_health": {},
            "overall_health": "UNKNOWN"
        }
        
        for split_name, split_stats in splits_data.items():
            summary["total_images"] += split_stats["total_images"]
            summary["total_valid_pairs"] += split_stats["valid_pairs"]
            summary["total_invalid_pairs"] += split_stats["invalid_pairs"]
            
            # Calculate split health
            if split_stats["total_images"] > 0:
                health_ratio = split_stats["valid_pairs"] / split_stats["total_images"]
                if health_ratio >= 0.95:
                    health_status = "EXCELLENT"
                elif health_ratio >= 0.85:
                    health_status = "GOOD"
                elif health_ratio >= 0.70:
                    health_status = "FAIR"
                else:
                    health_status = "POOR"
                
                summary["split_health"][split_name] = {
                    "health": health_status,
                    "valid_ratio": health_ratio,
                    "valid_pairs": split_stats["valid_pairs"],
                    "total_images": split_stats["total_images"]
                }
            
            # Total objects
            if "object_stats" in split_stats:
                summary["total_objects"] += split_stats["object_stats"]["total"]
        
        # Overall health
        if summary["total_images"] > 0:
            overall_ratio = summary["total_valid_pairs"] / summary["total_images"]
            if overall_ratio >= 0.95:
                summary["overall_health"] = "EXCELLENT"
            elif overall_ratio >= 0.85:
                summary["overall_health"] = "GOOD"
            elif overall_ratio >= 0.70:
                summary["overall_health"] = "FAIR"
            else:
                summary["overall_health"] = "POOR"
            
            summary["overall_valid_ratio"] = overall_ratio
        
        return summary
    
    def save_verification_report(self, overall_stats: Dict[str, Any], report_path: Path):
        """
        Save comprehensive verification report.
        
        Args:
            overall_stats: Verification statistics
            report_path: Path to save the report
        """
        report = {
            "verification_config": {
                "processed_dir": str(self.config.processed_dir),
                "check_image_integrity": self.config.check_image_integrity,
                "check_label_format": self.config.check_label_format,
                "check_dimensions": self.config.check_dimensions,
                "expected_img_size": self.config.expected_img_size,
                "max_invalid_files": self.config.max_invalid_files
            },
            "verification_stats": overall_stats,
            "environment": {
                "opencv_version": cv2.__version__,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # Save as JSON
        with open(report_path, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save as YAML if requested
        if self.config.save_detailed_report:
            yaml_path = report_path.with_suffix('.yaml')
            with open(yaml_path, "w", encoding='utf-8') as f:
                yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f" Verification report saved to: {report_path}")
    
    def print_verification_summary(self, overall_stats: Dict[str, Any]):
        """
        Print a human-readable verification summary.
        
        Args:
            overall_stats: Verification statistics
        """
        summary = overall_stats["summary"]
        splits = overall_stats["splits"]
        
        print("\n" + "="*60)
        print(" YOLO DATASET VERIFICATION SUMMARY")
        print("="*60)
        
        print(f"\n OVERALL HEALTH: {summary['overall_health']}")
        print(f" Valid pairs: {summary['total_valid_pairs']} / {summary['total_images']} "
              f"({summary.get('overall_valid_ratio', 0)*100:.1f}%)")
        
        if summary['total_objects'] > 0:
            print(f" Total objects: {summary['total_objects']}")
        
        print(f"\n SPLIT DETAILS:")
        for split_name, split_stats in splits.items():
            if split_stats['split_exists']:
                valid_ratio = split_stats['valid_pairs'] / split_stats['total_images'] if split_stats['total_images'] > 0 else 0
                health = summary['split_health'].get(split_name, {}).get('health', 'UNKNOWN')
                
                print(f"   {split_name.upper():<6} - {health:<8} - {split_stats['valid_pairs']:>4} / {split_stats['total_images']:>4} "
                      f"({valid_ratio*100:5.1f}%) valid")
                
                # Show issues if any
                issues = []
                if split_stats['missing_labels'] > 0:
                    issues.append(f"missing labels: {split_stats['missing_labels']}")
                if split_stats['invalid_images'] > 0:
                    issues.append(f"bad images: {split_stats['invalid_images']}")
                if split_stats['invalid_labels'] > 0:
                    issues.append(f"bad labels: {split_stats['invalid_labels']}")
                
                if issues:
                    print(f"             Issues: {', '.join(issues)}")
        
        print(f"\n📏 IMAGE DIMENSIONS:")
        for split_name, split_stats in splits.items():
            if split_stats.get('image_dimensions'):
                print(f"   {split_name.upper():<6}: {', '.join([f'{k}({v})' for k, v in split_stats['image_dimensions'].items()])}")
        
        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        if summary['overall_health'] == 'EXCELLENT':
            print("    Dataset is ready for training!")
        elif summary['overall_health'] == 'GOOD':
            print("    Dataset is good for training (minor issues)")
        elif summary['overall_health'] == 'FAIR':
            print("     Consider fixing issues before training")
        else:
            print("    Significant issues found - review and fix before training")
        
        print("="*60)
    
    def verify_dataset(self) -> Dict[str, Any]:
        """
        Main dataset verification routine.
        
        Returns:
            Dictionary with comprehensive verification results
        """
        splits = ["train", "val", "test"]
        overall_stats = {"splits": {}, "summary": {}}
        
        logger.info(f" Starting robust YOLO dataset verification in: {self.config.processed_dir}")
        
        # Verify each split
        for split in splits:
            split_path = self.config.processed_dir / split
            if not split_path.exists():
                logger.warning(f" Split directory not found: {split_path}")
                overall_stats["splits"][split] = {
                    "split_name": split,
                    "split_exists": False,
                    "error": "Split directory not found"
                }
                continue
            
            stats = self.verify_split(split_path)
            overall_stats["splits"][split] = stats
        
        # Generate summary
        overall_stats["summary"] = self.generate_summary_report(overall_stats)
        
        # Save report
        if self.config.save_detailed_report:
            report_path = self.config.processed_dir / "verification_report.json"
            self.save_verification_report(overall_stats, report_path)
        
        # Print summary
        self.print_verification_summary(overall_stats)
        
        return overall_stats

def main():
    """Main function for YOLO dataset verification."""
    parser = argparse.ArgumentParser(description="Robust YOLO Dataset Verification")
    parser.add_argument('--processed-dir', default='data/processed', 
                       help='Processed dataset directory (default: data/processed)')
    parser.add_argument('--expected-size', type=int, nargs=2, 
                       help='Expected image size (e.g., 640 640)')
    parser.add_argument('--no-image-check', action='store_true', 
                       help='Disable image integrity checks')
    parser.add_argument('--no-label-check', action='store_true', 
                       help='Disable label format checks')
    parser.add_argument('--max-invalid-files', type=int, default=1000,
                       help='Maximum invalid files to log (default: 1000)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create configuration
        config = VerificationConfig(
            processed_dir=Path(args.processed_dir),
            check_image_integrity=not args.no_image_check,
            check_label_format=not args.no_label_check,
            expected_img_size=tuple(args.expected_size) if args.expected_size else None,
            max_invalid_files=args.max_invalid_files
        )
        
        # Initialize and run verifier
        verifier = YOLODatasetVerifier(config)
        results = verifier.verify_dataset()
        
        # Return appropriate exit code
        overall_health = results["summary"].get("overall_health", "UNKNOWN")
        if overall_health in ["POOR", "UNKNOWN"]:
            return 1
        else:
            return 0
            
    except KeyboardInterrupt:
        print("\n Verification cancelled by user")
        return 1
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        logger.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    exit(main())