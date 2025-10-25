# ============================================================
# YOLOv8 Fine-Tuning Section - Robust Version
# ============================================================
import yaml
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field
from pathlib import Path
import logging
import argparse
import json
import sys
import os
from typing import Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class YOLOTrainConfig:
    data_yaml: Path = Path("data/processed/data.yaml")
    model_name: str = "yolov8n.pt"
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    device: str = "auto"
    project: Path = Path("runs/train")
    name: str = "fpidet_yolov8n"
    save_json: bool = True
    seed: int = 42
    patience: int = 50
    workers: int = 8
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    save_period: int = -1
    pretrained: bool = True
    verbose: bool = True
    exist_ok: bool = True  # Allow overwriting existing experiment
    resume: bool = False  # Resume from last checkpoint
    
    def __post_init__(self):
        """Validate and convert paths after initialization"""
        self.data_yaml = Path(self.data_yaml)
        self.project = Path(self.project)

class YOLOTrainer:
    """Handles YOLOv8 model fine-tuning and result tracking with robust error handling."""
    
    def __init__(self, config: YOLOTrainConfig):
        self.config = config
        self.model = None
        self.results = None
        self._validate_config()
        self._setup_environment()

    def _setup_environment(self):
        """Setup training environment including CUDA and directories"""
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            
        # Create project directory if it doesn't exist
        self.config.project.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" Environment setup complete")
        logger.info(f" PyTorch version: {torch.__version__}")
        logger.info(f" CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f" GPU: {torch.cuda.get_device_name(0)}")

    def _validate_config(self):
        """Comprehensive configuration validation"""
        errors = []
        
        # Check data configuration
        if not self.config.data_yaml.exists():
            errors.append(f"Data config not found: {self.config.data_yaml}")
        else:
            # Validate YAML structure
            try:
                with open(self.config.data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)
                required_keys = ['train', 'val', 'nc', 'names']
                for key in required_keys:
                    if key not in data_config:
                        errors.append(f"Missing required key in data.yaml: {key}")
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML in data config: {e}")
        
        # Check model availability
        if not self.config.model_name.endswith(('.pt', '.yaml')):
            # It's a model name, check if it's available
            available_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
            if self.config.model_name.replace('.pt', '') not in available_models:
                logger.warning(f"Model {self.config.model_name} might not be a standard YOLOv8 model")
        
        # Validate numerical parameters
        if self.config.epochs <= 0:
            errors.append("Epochs must be positive")
        if self.config.batch <= 0:
            errors.append("Batch size must be positive")
        if self.config.imgsz % 32 != 0:
            errors.append("Image size should be divisible by 32")
        if self.config.patience <= 0:
            errors.append("Patience must be positive")
        
        # Validate device
        if self.config.device != "auto":
            if self.config.device == "cpu":
                pass  # CPU is always valid
            else:
                try:
                    devices = [int(d.strip()) for d in self.config.device.split(',')]
                    if torch.cuda.is_available():
                        for device in devices:
                            if device >= torch.cuda.device_count():
                                errors.append(f"GPU device {device} not available")
                    else:
                        errors.append("CUDA not available but GPU devices specified")
                except ValueError:
                    errors.append(f"Invalid device specification: {self.config.device}")
        
        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f" Configuration validated successfully")
        logger.info(f" Using data config: {self.config.data_yaml}")

    def _get_train_kwargs(self) -> Dict[str, Any]:
        """Prepare training arguments with comprehensive settings"""
        train_kwargs = {
            "data": str(self.config.data_yaml),
            "epochs": self.config.epochs,
            "batch": self.config.batch,
            "imgsz": self.config.imgsz,
            "device": self.config.device,
            "project": str(self.config.project),
            "name": self.config.name,
            "seed": self.config.seed,
            "patience": self.config.patience,
            "workers": self.config.workers,
            "optimizer": self.config.optimizer,
            "lr0": self.config.lr0,
            "lrf": self.config.lrf,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_momentum": self.config.warmup_momentum,
            "warmup_bias_lr": self.config.warmup_bias_lr,
            "box": self.config.box,
            "cls": self.config.cls,
            "dfl": self.config.dfl,
            "save_period": self.config.save_period,
            "pretrained": self.config.pretrained,
            "verbose": self.config.verbose,
            "exist_ok": self.config.exist_ok,
            "resume": self.config.resume,
        }
        
        # Remove None values to use YOLO defaults
        return {k: v for k, v in train_kwargs.items() if v is not None}

    def train(self):
        """Run YOLOv8 training with comprehensive error handling and logging."""
        logger.info(" Starting YOLOv8 fine-tuning...")
        logger.info(f" Training configuration:")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch}")
        logger.info(f"  Image size: {self.config.imgsz}")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Project: {self.config.project}/{self.config.name}")

        try:
            # Load model with error handling
            logger.info(f" Loading model: {self.config.model_name}")
            try:
                self.model = YOLO(self.config.model_name)
                logger.info(f" Model loaded successfully")
            except Exception as e:
                logger.error(f" Failed to load model {self.config.model_name}: {e}")
                # Try to load from local path if it exists
                model_path = Path(self.config.model_name)
                if model_path.exists():
                    logger.info(f" Trying to load from local path: {model_path}")
                    self.model = YOLO(str(model_path))
                else:
                    raise

            # Prepare training arguments
            train_kwargs = self._get_train_kwargs()
            
            # Start training
            logger.info(" Starting training process...")
            self.results = self.model.train(**train_kwargs)

            logger.info(f" Training complete! Results saved in {self.config.project}/{self.config.name}")
            
            # Save summary
            if self.config.save_json:
                self.save_training_summary()

            # Validate training results
            self._validate_training_results()

        except KeyboardInterrupt:
            logger.warning(" Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f" YOLO training failed: {e}")
            # Provide helpful suggestions for common errors
            self._handle_training_error(e)
            raise

    def _validate_training_results(self):
        """Validate that training completed successfully and produced expected outputs"""
        try:
            results_dir = Path(self.config.project) / self.config.name
            if not results_dir.exists():
                logger.warning(" Results directory not found")
                return
            
            expected_files = ['weights/best.pt', 'weights/last.pt', 'results.csv']
            for file in expected_files:
                file_path = results_dir / file
                if file_path.exists():
                    logger.info(f" Found: {file}")
                else:
                    logger.warning(f" Missing expected file: {file}")
                
        except Exception as e:
            logger.warning(f" Results validation failed: {e}")

    def _handle_training_error(self, error: Exception):
        """Provide helpful suggestions for common training errors"""
        error_msg = str(error).lower()
        
        if "cuda" in error_msg or "gpu" in error_msg:
            logger.info(" Try setting --device cpu if GPU issues persist")
        elif "memory" in error_msg:
            logger.info(" Try reducing batch size or image size to save memory")
        elif "data" in error_msg or "dataset" in error_msg:
            logger.info(" Check your data.yaml file and dataset paths")
        elif "model" in error_msg:
            logger.info(" Verify the model file exists and is not corrupted")

    def save_training_summary(self):
        """Save comprehensive training metrics and configuration to JSON."""
        summary_path = Path(self.config.project) / self.config.name / f"{self.config.name}_summary.json"
        try:
            # Get the best model metrics if available
            metrics = {}
            if self.results and hasattr(self.results, 'results_dict'):
                metrics = self.results.results_dict
            elif hasattr(self.model, 'metrics'):
                metrics = getattr(self.model.metrics, 'results_dict', {})
            
            summary_data = {
                "training_config": {
                    "model": self.config.model_name,
                    "epochs": self.config.epochs,
                    "batch": self.config.batch,
                    "imgsz": self.config.imgsz,
                    "device": self.config.device,
                    "project": str(self.config.project),
                    "name": self.config.name,
                    "seed": self.config.seed,
                    "optimizer": self.config.optimizer,
                    "learning_rate": self.config.lr0,
                },
                "paths": {
                    "data_yaml": str(self.config.data_yaml),
                    "results_dir": str(Path(self.config.project) / self.config.name),
                    "best_model": str(Path(self.config.project) / self.config.name / "weights" / "best.pt"),
                    "last_model": str(Path(self.config.project) / self.config.name / "weights" / "last.pt"),
                },
                "metrics": metrics,
                "environment": {
                    "pytorch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            }

            # Ensure directory exists
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, default=str)

            logger.info(f" Training summary saved to: {summary_path}")

        except Exception as e:
            logger.error(f" Failed to save training summary: {e}")

def main():
    """Enhanced CLI for YOLOv8 training with better argument handling"""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on custom dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and model arguments
    parser.add_argument("--data", default="data/processed/data.yaml", 
                       help="Path to dataset YAML file")
    parser.add_argument("--model", default="yolov8n.pt", 
                       help="YOLOv8 model variant or path to custom weights")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, 
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Image size (should be divisible by 32)")
    parser.add_argument("--device", default="auto", 
                       help="Device (cpu, 0, 0,1, etc.)")
    
    # Project settings
    parser.add_argument("--project", default="runs/train", 
                       help="Project save directory")
    parser.add_argument("--name", default="fpidet_yolov8n", 
                       help="Experiment name")
    parser.add_argument("--no-save-json", action="store_true", 
                       help="Disable saving JSON summary")
    
    # Advanced training parameters
    parser.add_argument("--patience", type=int, default=50, 
                       help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=8, 
                       help="Number of data loading workers")
    parser.add_argument("--lr0", type=float, default=0.01, 
                       help="Initial learning rate")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from last checkpoint")
    
    args = parser.parse_args()

    try:
        config = YOLOTrainConfig(
            data_yaml=Path(args.data),
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=Path(args.project),
            name=args.name,
            save_json=not args.no_save_json,
            patience=args.patience,
            workers=args.workers,
            lr0=args.lr0,
            resume=args.resume
        )

        trainer = YOLOTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info(" Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())