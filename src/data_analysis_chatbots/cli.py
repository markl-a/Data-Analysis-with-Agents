"""命令行接口(CLI)工具"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from .data_downloader import DataDownloader
from .data_loader import DataLoader
from .preprocessing import DataValidator
from .clustering import KMeansClusterer, RFMAnalyzer
from .marketing import CLVPredictor
from .utils import setup_logging


def download_data(args):
    """下載數據集"""
    setup_logging(level="INFO")

    downloader = DataDownloader()

    if args.sample:
        downloader.download_sample_data()
    elif args.all:
        downloader.download_all_datasets(force=args.force)
    elif args.dataset:
        downloader.download_dataset(args.dataset, force=args.force)
    else:
        print("請指定 --all, --dataset 或 --sample")
        sys.exit(1)


def analyze_data(args):
    """執行數據分析"""
    setup_logging(level="INFO")

    loader = DataLoader()

    # 載入數據
    try:
        if args.dataset == 'mall_customers':
            df = loader.load_mall_customers()
        elif args.dataset == 'ecommerce':
            df = loader.load_ecommerce()
        elif args.dataset == 'personality':
            df = loader.load_personality()
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)

        logger.info(f"Loaded {len(df)} rows from {args.dataset}")

        # 執行分析
        if args.analysis == 'validate':
            validator = DataValidator(df)
            validator.print_report()

        elif args.analysis == 'cluster':
            if args.dataset == 'mall_customers':
                clusterer = KMeansClusterer(n_clusters=args.n_clusters)
                features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
                labels = clusterer.fit_predict(df, features)

                df['Cluster'] = labels
                logger.success(f"Clustering completed with {args.n_clusters} clusters")

                # 保存結果
                output_file = args.output or 'data/outputs/cluster_results.csv'
                df.to_csv(output_file, index=False)
                logger.success(f"Results saved to {output_file}")

        elif args.analysis == 'rfm':
            if args.dataset != 'ecommerce':
                logger.error("RFM analysis requires ecommerce dataset")
                sys.exit(1)

            rfm_analyzer = RFMAnalyzer(
                df=df,
                customer_id_col='CustomerID',
                date_col='InvoiceDate',
                amount_col='TotalAmount'
            )

            segments = rfm_analyzer.segment_customers()
            summary = rfm_analyzer.get_segment_summary()

            logger.success("RFM analysis completed")
            print("\nSegment Summary:")
            print(summary)

            # 保存結果
            output_file = args.output or 'data/outputs/rfm_segments.csv'
            segments.to_csv(output_file, index=False)
            logger.success(f"Results saved to {output_file}")

        else:
            logger.error(f"Unknown analysis type: {args.analysis}")
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.info("Run 'dac-download' to download datasets first")
        sys.exit(1)


def main():
    """主CLI入口"""
    parser = argparse.ArgumentParser(
        description="Data Analysis with Chatbots CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  dac-download --all

  # Download specific dataset
  dac-download --dataset mall_customers

  # Generate sample data
  dac-download --sample

  # Validate data
  dac-analyze --dataset mall_customers --analysis validate

  # Run clustering
  dac-analyze --dataset mall_customers --analysis cluster --n-clusters 5

  # Run RFM analysis
  dac-analyze --dataset ecommerce --analysis rfm
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument('--all', action='store_true', help='Download all datasets')
    download_parser.add_argument('--dataset', type=str, help='Download specific dataset')
    download_parser.add_argument('--sample', action='store_true', help='Generate sample data')
    download_parser.add_argument('--force', action='store_true', help='Force re-download')
    download_parser.set_defaults(func=download_data)

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data')
    analyze_parser.add_argument('--dataset', type=str, required=True,
                               choices=['mall_customers', 'ecommerce', 'personality'],
                               help='Dataset to analyze')
    analyze_parser.add_argument('--analysis', type=str, required=True,
                               choices=['validate', 'cluster', 'rfm'],
                               help='Type of analysis')
    analyze_parser.add_argument('--n-clusters', type=int, default=5,
                               help='Number of clusters (for clustering)')
    analyze_parser.add_argument('--output', type=str, help='Output file path')
    analyze_parser.set_defaults(func=analyze_data)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
