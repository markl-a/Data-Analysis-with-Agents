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

    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in the DataFrame.

        Returns:
            Dictionary mapping column names to missing value counts
        """
        logger.info("Checking for missing values...")

        missing_count = self.df.isnull().sum()
        results = {col: int(count) for col, count in missing_count.items()}

        self.validation_results['missing_values'] = results
        logger.info(f"Found {sum(results.values())} missing values")

        return results

    def check_duplicates(self, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.

        Args:
            subset: List of columns to check for duplicates. If None, checks all columns.

        Returns:
            Number of duplicate rows
        """
        logger.info("Checking for duplicates...")

        if subset:
            duplicates = self.df.duplicated(subset=subset, keep='first')
        else:
            duplicates = self.df.duplicated(keep='first')

        duplicate_count = int(duplicates.sum())

        self.validation_results['duplicates'] = duplicate_count
        logger.info(f"Found {duplicate_count} duplicate rows")

        return duplicate_count

    def check_data_types(self) -> Dict[str, str]:
        """
        Check data types of columns.

        Returns:
            Dictionary mapping column names to their data types
        """
        logger.info("Checking data types...")

        dtypes = self.df.dtypes.astype(str).to_dict()

        self.validation_results['data_types'] = dtypes
        logger.info(f"Checked data types for {len(dtypes)} columns")

        return dtypes

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
            Dictionary with total_rows, total_columns, missing_values, duplicate_rows
        """
        logger.info("Generating comprehensive data quality report...")

        missing_values = self.check_missing_values()
        duplicate_rows = self.check_duplicates()
        self.check_data_types()

        results = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': missing_values,
            'duplicate_rows': duplicate_rows
        }

        self.validation_results = results
        logger.success("Data quality report generated successfully")

        return results

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.

        Returns:
            DataFrame with summary statistics
        """
        logger.info("Getting summary statistics...")
        return self.df.describe()

    def validate_column_exists(self, column: str) -> bool:
        """
        Check if a column exists in the DataFrame.

        Args:
            column: Column name to check

        Returns:
            True if column exists, False otherwise
        """
        return column in self.df.columns

    def validate_no_nulls(self, column: str) -> bool:
        """
        Check if a column has no null values.

        Args:
            column: Column name to check

        Returns:
            True if no nulls, False otherwise
        """
        if column not in self.df.columns:
            return False
        return self.df[column].isnull().sum() == 0

    def validate_numeric_range(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """
        Check if all values in a column are within a specified range.

        Args:
            column: Column name to check
            min_value: Minimum allowed value (optional)
            max_value: Maximum allowed value (optional)

        Returns:
            True if all values are within range, False otherwise
        """
        if column not in self.df.columns:
            return False

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return False

        values = self.df[column].dropna()

        if min_value is not None and values.min() < min_value:
            return False

        if max_value is not None and values.max() > max_value:
            return False

        return True

    def fix_missing_values(
        self,
        strategy: str = 'drop',
        fill_value: Any = None
    ) -> pd.DataFrame:
        """
        Fix missing values in the DataFrame.

        Args:
            strategy: Strategy to use ('drop' or 'fill')
            fill_value: Value to fill with when strategy is 'fill'

        Returns:
            DataFrame with missing values fixed
        """
        logger.info(f"Fixing missing values with strategy: {strategy}")

        if strategy == 'drop':
            result = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                # Fill numeric columns with fill_value, leave others
                result = self.df.copy()
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                result[numeric_cols] = result[numeric_cols].fillna(fill_value)
            else:
                result = self.df.fillna(method='ffill').fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Fixed missing values. Rows: {len(self.df)} -> {len(result)}")
        return result

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.

        Args:
            subset: List of columns to consider for duplicates

        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicates...")

        if subset:
            result = self.df.drop_duplicates(subset=subset, keep='first')
        else:
            result = self.df.drop_duplicates(keep='first')

        logger.info(f"Removed duplicates. Rows: {len(self.df)} -> {len(result)}")
        return result

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
