import zipfile
import os
import logging
from pathlib import Path
from typing import Optional, Union
import hashlib
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Robust data loader for extracting and managing datasets.
    """
    
    def __init__(self, default_zip_path: str = "dataset/FPI-Det.zip", default_extract_path: str = "data/"):
        self.default_zip_path = Path(default_zip_path)
        self.default_extract_path = Path(default_extract_path)
        
    def validate_zip_file(self, zip_path: Union[str, Path]) -> bool:
        """
        Validate that the zip file exists and is a valid zip file.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            bool: True if valid, False otherwise
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            logger.error(f" Zip file not found: {zip_path}")
            return False
            
        if not zip_path.is_file():
            logger.error(f" Path is not a file: {zip_path}")
            return False
            
        # Check file extension
        if zip_path.suffix.lower() != '.zip':
            logger.warning(f"  File extension is not .zip: {zip_path}")
            
        # Verify it's actually a zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test zip file integrity
                bad_file = zip_ref.testzip()
                if bad_file is not None:
                    logger.error(f" Corrupted file in zip: {bad_file}")
                    return False
                return True
        except zipfile.BadZipFile:
            logger.error(f" File is not a valid zip file: {zip_path}")
            return False
        except Exception as e:
            logger.error(f" Error validating zip file {zip_path}: {str(e)}")
            return False
    
    def calculate_file_hash(self, file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        Calculate file hash for verification.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
            
        Returns:
            str: File hash
        """
        file_path = Path(file_path)
        hash_func = getattr(hashlib, algorithm)()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f" Error calculating hash for {file_path}: {str(e)}")
            return ""
    
    def is_already_extracted(self, extract_path: Union[str, Path], min_files: int = 1) -> bool:
        """
        Check if dataset is already extracted and appears complete.
        
        Args:
            extract_path: Path where dataset should be extracted
            min_files: Minimum number of files expected in the extracted directory
            
        Returns:
            bool: True if already extracted and appears complete
        """
        extract_path = Path(extract_path)
        
        if not extract_path.exists():
            return False
            
        # Count files in directory (excluding hidden files)
        file_count = 0
        try:
            for item in extract_path.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    file_count += 1
                    
            if file_count >= min_files:
                logger.info(f" Dataset already extracted with {file_count} files")
                return True
            else:
                logger.warning(f"  Extraction directory exists but has only {file_count} files (expected at least {min_files})")
                return False
                
        except Exception as e:
            logger.error(f" Error checking extraction directory: {str(e)}")
            return False
    
    def get_zip_contents_info(self, zip_path: Union[str, Path]) -> dict:
        """
        Get information about zip file contents.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            dict: Information about zip contents
        """
        zip_path = Path(zip_path)
        info = {
            'file_count': 0,
            'total_size': 0,
            'file_types': {},
            'root_dirs': set()
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    info['file_count'] += 1
                    info['total_size'] += file_info.file_size
                    
                    # Count file types
                    ext = Path(file_info.filename).suffix.lower()
                    info['file_types'][ext] = info['file_types'].get(ext, 0) + 1
                    
                    # Track root directories
                    first_dir = file_info.filename.split('/')[0]
                    if first_dir and not first_dir.startswith('__') and not first_dir.startswith('.'):
                        info['root_dirs'].add(first_dir)
                        
        except Exception as e:
            logger.error(f" Error reading zip contents: {str(e)}")
            
        return info
    
    def create_backup(self, extract_path: Union[str, Path]) -> bool:
        """
        Create a backup of existing extracted data.
        
        Args:
            extract_path: Path to the extraction directory
            
        Returns:
            bool: True if backup created successfully
        """
        extract_path = Path(extract_path)
        backup_path = extract_path.with_name(extract_path.name + '_backup')
        
        try:
            if extract_path.exists():
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(extract_path, backup_path)
                logger.info(f" Backup created: {backup_path}")
                return True
        except Exception as e:
            logger.error(f" Error creating backup: {str(e)}")
            
        return False
    
    def extract_dataset(self, 
                       zip_path: Optional[Union[str, Path]] = None, 
                       extract_to: Optional[Union[str, Path]] = None,
                       force_extract: bool = False,
                       create_backup: bool = True,
                       min_expected_files: int = 10) -> bool:
        """
        Extracts dataset with robust error handling and validation.
        
        Args:
            zip_path: Path to the zip file (uses default if None)
            extract_to: Extraction directory (uses default if None)
            force_extract: Force re-extraction even if already extracted
            create_backup: Create backup of existing data before extraction
            min_expected_files: Minimum number of files expected after extraction
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        # Use defaults if not provided
        zip_path = Path(zip_path) if zip_path else self.default_zip_path
        extract_to = Path(extract_to) if extract_to else self.default_extract_path
        
        logger.info(f" Starting dataset extraction...")
        logger.info(f"   Source: {zip_path}")
        logger.info(f"   Target: {extract_to}")
        
        # Validate zip file
        if not self.validate_zip_file(zip_path):
            return False
        
        # Check if already extracted
        if not force_extract and self.is_already_extracted(extract_to, min_expected_files):
            logger.info(" Using existing extracted dataset")
            return True
        
        # Get zip file info
        zip_info = self.get_zip_contents_info(zip_path)
        logger.info(f"Zip file contains: {zip_info['file_count']} files, "
                   f"{zip_info['total_size'] / (1024*1024):.2f} MB")
        
        if zip_info['root_dirs']:
            logger.info(f" Root directories: {', '.join(zip_info['root_dirs'])}")
        
        # Create backup if requested and directory exists
        if extract_to.exists() and create_backup:
            self.create_backup(extract_to)
        
        # Create extraction directory
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created extraction directory: {extract_to}")
        except Exception as e:
            logger.error(f" Error creating extraction directory: {str(e)}")
            return False
        
        # Extract the dataset
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Show progress for large files
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                logger.info(f" Extracting {total_files} files...")
                zip_ref.extractall(extract_to)
                
            logger.info(f" Successfully extracted {total_files} files to: {extract_to}")
            
            # Verify extraction
            if self.is_already_extracted(extract_to, min_expected_files):
                logger.info(" Extraction verified successfully")
                return True
            else:
                logger.warning("  Extraction completed but verification failed")
                return False
                
        except zipfile.BadZipFile:
            logger.error(f"Failed to extract: {zip_path} is not a valid zip file")
            return False
        except PermissionError:
            logger.error(f" Permission denied when extracting to: {extract_to}")
            return False
        except Exception as e:
            logger.error(f" Error during extraction: {str(e)}")
            # Clean up partially extracted files
            if extract_to.exists():
                try:
                    shutil.rmtree(extract_to)
                    logger.info(" Cleaned up partially extracted files")
                except:
                    logger.error(" Failed to clean up partially extracted files")
            return False

def extract_dataset(zip_path: str = "dataset/FPI-Det.zip", 
                   extract_to: str = "data/",
                   force: bool = False) -> bool:
    """
    Convenience function for simple extraction.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Extraction directory
        force: Force re-extraction
        
    Returns:
        bool: True if successful
    """
    loader = DataLoader()
    return loader.extract_dataset(zip_path, extract_to, force_extract=force)

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Basic extraction
    success = loader.extract_dataset()
    
    # Force re-extraction with backup
    # success = loader.extract_dataset(force_extract=True, create_backup=True)
    
    if success:
        print(" Dataset extraction completed successfully!")
    else:
        print(" Dataset extraction failed!")
        exit(1)