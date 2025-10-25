import logging
from src.data_loader import DataLoader, extract_dataset

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='w')
        ]
    )

def main():
    """Main function to set up the psychosomatic behavior detection system."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print(" Psychosomatic Behavior Detection - Phase 1 Setup")
    print("=" * 50)
    
    # Option 1: Simple usage with the convenience function
    logger.info("Starting dataset extraction using convenience function...")
    success = extract_dataset()
    
    if success:
        print(" Dataset extraction completed successfully!")
        logger.info("Dataset extraction completed successfully")
    else:
        print(" Dataset extraction failed. Check logs for details.")
        logger.error("Dataset extraction failed")
        return
    
    # Option 2: Advanced usage with the DataLoader class (commented out)
    # Uncomment below for more control and features
    
    # logger.info("Starting dataset extraction using DataLoader class...")
    # loader = DataLoader()
    
    # # Extract with additional options
    # success = loader.extract_dataset(
    #     zip_path="dataset/FPI-Det.zip",
    #     extract_to="data/",
    #     force_extract=False,  # Set to True to force re-extraction
    #     create_backup=True,   # Create backup if directory exists
    #     min_expected_files=10 # Minimum files expected after extraction
    # )
    
    # if success:
    #     print(" Dataset extraction completed successfully!")
    #     logger.info("Dataset extraction completed successfully with advanced options")
    # else:
    #     print(" Dataset extraction failed. Check logs for details.")
    #     logger.error("Dataset extraction failed with advanced options")
    #     return
    
    print("\n Next steps:")
    print("  1. Explore the extracted data in the 'data/' directory")
    print("  2. Check app.log for detailed extraction information")
    print("  3. Proceed with data preprocessing and analysis")
    print("=" * 50)

if __name__ == "__main__":
    main()