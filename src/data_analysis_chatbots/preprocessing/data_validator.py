"""Data validation utilities."""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger


class DataValidator:
    """Validate and check data quality."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataValidator.

        Args:
            df: DataFrame to validate
        """
        self.df = df
        self.validation_results = {}

    def check_missing_values(self) -> Dict[str, Any]:
        """
        Check for missing values in the DataFrame.

        Returns:
            Dictionary with missing value statistics
        """
        logger.info("Checking for missing values...")

        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df) * 100).round(2)

        results = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_by_column': {
                col: {
                    'count': int(count),
                    'percentage': float(missing_percent[col])
                }
                for col, count in missing_count.items()
                if count > 0
            },
            'columns_with_missing': list(missing_count[missing_count > 0].index),
            'total_missing_cells': int(missing_count.sum())
        }

        self.validation_results['missing_values'] = results
        logger.info(f"Found {results['total_missing_cells']} missing values")

        return results

    def check_duplicates(self, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check for duplicate rows.

        Args:
            subset: List of columns to check for duplicates. If None, checks all columns.

        Returns:
            Dictionary with duplicate statistics
        """
        logger.info("Checking for duplicates...")

        if subset:
            duplicates = self.df.duplicated(subset=subset, keep='first')
        else:
            duplicates = self.df.duplicated(keep='first')

        duplicate_count = duplicates.sum()
        duplicate_rows = self.df[duplicates]

        results = {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float((duplicate_count / len(self.df) * 100).round(2)),
            'checked_columns': subset if subset else 'all',
            'has_duplicates': bool(duplicate_count > 0)
        }

        self.validation_results['duplicates'] = results
        logger.info(f"Found {duplicate_count} duplicate rows")

        return results

    def check_data_types(self) -> Dict[str, Any]:
        """
        Check data types of columns.

        Returns:
            Dictionary with data type information
        """
        logger.info("Checking data types...")

        dtypes = self.df.dtypes.astype(str).to_dict()

        results = {
            'data_types': dtypes,
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(self.df.select_dtypes(include=['datetime64']).columns)
        }

        self.validation_results['data_types'] = results
        logger.info(f"Found {len(results['numeric_columns'])} numeric, "
                   f"{len(results['categorical_columns'])} categorical columns")

        return results

    def check_value_ranges(self, column: str) -> Dict[str, Any]:
        """
        Check value ranges for a numeric column.

        Args:
            column: Column name to check

        Returns:
            Dictionary with range statistics
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")

        logger.info(f"Checking value ranges for: {column}")

        results = {
            'column': column,
            'min': float(self.df[column].min()),
            'max': float(self.df[column].max()),
            'mean': float(self.df[column].mean()),
            'median': float(self.df[column].median()),
            'std': float(self.df[column].std()),
            'q25': float(self.df[column].quantile(0.25)),
            'q75': float(self.df[column].quantile(0.75))
        }

        return results

    def check_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Check for outliers in a numeric column.

        Args:
            column: Column name to check
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)

        Returns:
            Dictionary with outlier information
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")

        logger.info(f"Checking outliers for: {column} using {method} method")

        if method == 'iqr':
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)

        elif method == 'zscore':
            mean = self.df[column].mean()
            std = self.df[column].std()
            z_scores = np.abs((self.df[column] - mean) / std)
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_count = outliers.sum()

        results = {
            'column': column,
            'method': method,
            'outlier_count': int(outlier_count),
            'outlier_percentage': float((outlier_count / len(self.df) * 100).round(2)),
            'outlier_indices': list(self.df[outliers].index) if outlier_count < 100 else [],
            'has_outliers': bool(outlier_count > 0)
        }

        return results

    def check_unique_values(self, column: str) -> Dict[str, Any]:
        """
        Check unique values in a column.

        Args:
            column: Column name to check

        Returns:
            Dictionary with unique value information
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        logger.info(f"Checking unique values for: {column}")

        unique_count = self.df[column].nunique()
        value_counts = self.df[column].value_counts()

        results = {
            'column': column,
            'unique_count': int(unique_count),
            'total_count': len(self.df),
            'unique_percentage': float((unique_count / len(self.df) * 100).round(2)),
            'most_common': value_counts.head(10).to_dict() if unique_count < 1000 else {},
            'is_unique': unique_count == len(self.df)
        }

        return results

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.

        Returns:
            Dictionary with complete validation results
        """
        logger.info("Generating comprehensive data quality report...")

        # Run all checks
        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()

        # Check outliers for numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        for col in numeric_columns:
            try:
                outlier_results[col] = self.check_outliers(col)
            except Exception as e:
                logger.warning(f"Could not check outliers for {col}: {e}")

        self.validation_results['outliers'] = outlier_results

        # Add summary
        self.validation_results['summary'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024**2),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }

        logger.success("Data quality report generated successfully")

        return self.validation_results

    def print_report(self):
        """Print a formatted validation report."""
        if not self.validation_results:
            self.generate_report()

        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY REPORT")
        logger.info("="*60)

        summary = self.validation_results.get('summary', {})
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Rows: {summary.get('total_rows', 0):,}")
        logger.info(f"  Columns: {summary.get('total_columns', 0)}")
        logger.info(f"  Memory Usage: {summary.get('memory_usage_mb', 0):.2f} MB")

        missing = self.validation_results.get('missing_values', {})
        logger.info(f"\nMissing Values:")
        logger.info(f"  Total Missing Cells: {missing.get('total_missing_cells', 0):,}")
        if missing.get('columns_with_missing'):
            logger.info(f"  Columns with Missing Values: {len(missing['columns_with_missing'])}")
            for col in missing['columns_with_missing'][:5]:
                info = missing['missing_by_column'][col]
                logger.info(f"    - {col}: {info['count']} ({info['percentage']}%)")

        duplicates = self.validation_results.get('duplicates', {})
        logger.info(f"\nDuplicates:")
        logger.info(f"  Duplicate Rows: {duplicates.get('duplicate_count', 0):,} "
              f"({duplicates.get('duplicate_percentage', 0)}%)")

        outliers = self.validation_results.get('outliers', {})
        if outliers:
            logger.info(f"\nOutliers (IQR method):")
            for col, data in outliers.items():
                if data.get('has_outliers'):
                    logger.info(f"  - {col}: {data['outlier_count']} ({data['outlier_percentage']}%)")

        logger.info("\n" + "="*60 + "\n")
