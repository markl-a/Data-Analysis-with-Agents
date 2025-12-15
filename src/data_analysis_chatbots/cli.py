"""命令行接口(CLI)工具"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from .data_downloader import DataDownloader
from .data_loader import DataLoader
from .preprocessing import DataValidator
from .clustering import (
    KMeansClusterer,
    DBSCANClusterer,
    GMMClusterer,
    HierarchicalClusterer,
    RFMAnalyzer
)
from .marketing import CLVPredictor
from .utils import setup_logging


def download_data(args) -> None:
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
        logger.error("請指定 --all, --dataset 或 --sample")
        sys.exit(1)


def analyze_data(args) -> None:
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
                features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

                # 根據算法類型創建聚類器
                algorithm = getattr(args, 'algorithm', 'kmeans')

                if algorithm == 'kmeans':
                    clusterer = KMeansClusterer(n_clusters=args.n_clusters)
                    labels = clusterer.fit_predict(df, features)
                    logger.success(f"K-Means clustering completed with {args.n_clusters} clusters")

                elif algorithm == 'dbscan':
                    eps = getattr(args, 'eps', 0.5)
                    min_samples = getattr(args, 'min_samples', 5)
                    clusterer = DBSCANClusterer(eps=eps, min_samples=min_samples)
                    labels = clusterer.fit_predict(df, features)
                    logger.success(
                        f"DBSCAN clustering completed: {clusterer.n_clusters_} clusters, "
                        f"{clusterer.n_noise_} noise points"
                    )

                elif algorithm == 'gmm':
                    clusterer = GMMClusterer(n_components=args.n_clusters)
                    labels = clusterer.fit_predict(df, features)
                    probabilities = clusterer.predict_proba(df, features)
                    df['Max_Probability'] = probabilities.max(axis=1)
                    logger.success(f"GMM clustering completed with {args.n_clusters} components")

                elif algorithm == 'hierarchical':
                    linkage_method = getattr(args, 'linkage', 'ward')
                    clusterer = HierarchicalClusterer(
                        n_clusters=args.n_clusters,
                        linkage=linkage_method
                    )
                    labels = clusterer.fit_predict(df, features)
                    logger.success(
                        f"Hierarchical clustering completed with {args.n_clusters} clusters "
                        f"(linkage={linkage_method})"
                    )
                else:
                    logger.error(f"Unknown algorithm: {algorithm}")
                    sys.exit(1)

                df['Cluster'] = labels

                # 獲取並顯示聚類摘要
                summary = clusterer.get_cluster_summary(df, features)
                logger.info("\nCluster Summary:")
                logger.info(summary.to_string())

                # 評估聚類質量
                metrics = clusterer.evaluate_clustering(df, features)
                logger.info(f"\nClustering Metrics: {metrics}")

                # 保存結果
                output_file = args.output or f'data/outputs/{algorithm}_cluster_results.csv'
                df.to_csv(output_file, index=False)
                logger.success(f"Results saved to {output_file}")

                # 保存摘要
                summary_file = output_file.replace('.csv', '_summary.csv')
                summary.to_csv(summary_file, index=False)
                logger.success(f"Summary saved to {summary_file}")

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
            logger.info("\nSegment Summary:")
            logger.info(summary)

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


def main() -> None:
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

  # Run K-Means clustering (default)
  dac-analyze --dataset mall_customers --analysis cluster --n-clusters 5

  # Run DBSCAN clustering
  dac-analyze --dataset mall_customers --analysis cluster --algorithm dbscan --eps 0.5 --min-samples 10

  # Run GMM clustering
  dac-analyze --dataset mall_customers --analysis cluster --algorithm gmm --n-clusters 3

  # Run Hierarchical clustering
  dac-analyze --dataset mall_customers --analysis cluster --algorithm hierarchical --n-clusters 4 --linkage ward

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

    # Clustering algorithm options
    analyze_parser.add_argument('--algorithm', type=str, default='kmeans',
                               choices=['kmeans', 'dbscan', 'gmm', 'hierarchical'],
                               help='Clustering algorithm (default: kmeans)')
    analyze_parser.add_argument('--n-clusters', type=int, default=5,
                               help='Number of clusters (for kmeans/gmm/hierarchical)')

    # DBSCAN specific parameters
    analyze_parser.add_argument('--eps', type=float, default=0.5,
                               help='DBSCAN: Maximum distance between samples (default: 0.5)')
    analyze_parser.add_argument('--min-samples', type=int, default=5,
                               help='DBSCAN: Minimum samples in neighborhood (default: 5)')

    # Hierarchical specific parameters
    analyze_parser.add_argument('--linkage', type=str, default='ward',
                               choices=['ward', 'complete', 'average', 'single'],
                               help='Hierarchical: Linkage method (default: ward)')

    analyze_parser.add_argument('--output', type=str, help='Output file path')
    analyze_parser.set_defaults(func=analyze_data)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
