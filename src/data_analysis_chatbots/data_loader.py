"""Data loader for various datasets used in the project."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from .config_loader import ConfigLoader
from .utils import get_project_root, ensure_dir


class DataLoader:
    """Load and manage datasets for analysis."""

    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize the DataLoader.

        Args:
            config: Configuration loader instance. If None, creates a new one.
        """
        self.config = config or ConfigLoader()
        self.project_root = get_project_root()
        self._setup_paths()

    def _setup_paths(self):
        """Setup data paths from configuration."""
        paths = self.config.get_paths()
        self.data_root = self.project_root / paths.get('data_root', 'data')
        self.raw_data_path = self.project_root / paths.get('raw_data', 'data/raw')
        self.processed_data_path = self.project_root / paths.get('processed_data', 'data/processed')
        self.outputs_path = self.project_root / paths.get('outputs', 'data/outputs')

        # Ensure directories exist
        for path in [self.raw_data_path, self.processed_data_path, self.outputs_path]:
            ensure_dir(path)

    def load_dataset(self, dataset_name: str, data_type: str = 'raw') -> pd.DataFrame:
        """
        Load a dataset by name.

        Args:
            dataset_name: Name of the dataset (e.g., 'disaster_tweets', 'ecommerce')
            data_type: Type of data to load ('raw' or 'processed')

        Returns:
            DataFrame containing the dataset

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If the dataset name is not recognized
        """
        try:
            dataset_config = self.config.get_dataset_config(dataset_name)
            filename = dataset_config.get('filename')

            if data_type == 'raw':
                file_path = self.raw_data_path / filename
            elif data_type == 'processed':
                file_path = self.processed_data_path / filename
            else:
                raise ValueError(f"Invalid data_type: {data_type}. Must be 'raw' or 'processed'")

            if not file_path.exists():
                logger.warning(f"Dataset file not found: {file_path}")
                logger.info(f"To download datasets, run: python -m data_analysis_chatbots.data_downloader")
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            # Load the data
            logger.info(f"Loading dataset: {dataset_name} from {file_path}")
            df = pd.read_csv(file_path)
            logger.success(f"Successfully loaded {len(df)} rows from {dataset_name}")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise

    def load_disaster_tweets(self, data_type: str = 'raw') -> pd.DataFrame:
        """Load the Disaster Tweets dataset."""
        return self.load_dataset('disaster_tweets', data_type)

    def load_ecommerce(self, data_type: str = 'raw') -> pd.DataFrame:
        """Load the E-Commerce dataset."""
        return self.load_dataset('ecommerce', data_type)

    def load_mall_customers(self, data_type: str = 'raw') -> pd.DataFrame:
        """Load the Mall Customers dataset."""
        return self.load_dataset('mall_customers', data_type)

    def load_personality(self, data_type: str = 'raw') -> pd.DataFrame:
        """Load the Customer Personality Analysis dataset."""
        return self.load_dataset('personality', data_type)

    def load_marketing_segmentation(self, data_type: str = 'raw') -> pd.DataFrame:
        """Load the Marketing Segmentation dataset."""
        return self.load_dataset('marketing_segmentation', data_type)

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save processed data to the processed data directory.

        Args:
            df: DataFrame to save
            filename: Name of the file to save

        Returns:
            Path to the saved file
        """
        file_path = self.processed_data_path / filename
        logger.info(f"Saving processed data to {file_path}")
        df.to_csv(file_path, index=False)
        logger.success(f"Successfully saved {len(df)} rows to {filename}")
        return file_path

    def save_output(self, df: pd.DataFrame, filename: str, file_format: str = 'csv') -> Path:
        """
        Save analysis output to the outputs directory.

        Args:
            df: DataFrame to save
            filename: Name of the file to save
            file_format: Format to save ('csv', 'excel', 'json')

        Returns:
            Path to the saved file
        """
        if not filename.endswith(f'.{file_format}'):
            filename = f"{filename}.{file_format}"

        file_path = self.outputs_path / filename
        logger.info(f"Saving output to {file_path}")

        if file_format == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format == 'excel':
            df.to_excel(file_path, index=False)
        elif file_format == 'json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.success(f"Successfully saved output to {filename}")
        return file_path

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing dataset information
        """
        return self.config.get_dataset_config(dataset_name)

    def list_available_datasets(self) -> list:
        """
        List all available datasets in the configuration.

        Returns:
            List of dataset names
        """
        datasets = self.config.get('datasets', {})
        return list(datasets.keys())
