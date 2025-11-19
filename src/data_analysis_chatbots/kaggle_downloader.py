"""
Kaggle 數據集下載器

自動從 Kaggle 下載數據集並整合到專案中

使用前需要：
1. 安裝 kaggle: pip install kaggle
2. 設置 Kaggle API 憑證: https://www.kaggle.com/docs/api
3. 下載 kaggle.json 並放到 ~/.kaggle/kaggle.json

作者: Data Analysis with Chatbots Team
日期: 2025-01-19
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
import subprocess
import shutil


class KaggleDatasetDownloader:
    """Kaggle 數據集下載器"""

    # 常用的 Kaggle 數據集映射
    POPULAR_DATASETS = {
        # 結構化數據
        'titanic': 'titanic',
        'house-prices': 'house-prices-advanced-regression-techniques',
        'credit-fraud': 'mlg-ulb/creditcardfraud',
        'customer-churn': 'blastchar/telco-customer-churn',
        'bank-marketing': 'henriqueyamahata/bank-marketing',
        'wine-quality': 'uciml/red-wine-quality-cortez-et-al-2009',
        'adult-income': 'uciml/adult-census-income',

        # 時間序列
        'bitcoin': 'mczielinski/bitcoin-historical-data',
        'stock-market': 'borismarjanovic/price-volume-data-for-all-us-stocks-etfs',
        'sales': 'c3rp/store-sales-forecasting',
        'energy': 'robikscube/hourly-energy-consumption',
        'covid19': 'sudalairajkumar/novel-corona-virus-2019-dataset',

        # NLP
        'sentiment': 'kazanova/sentiment140',
        'spam': 'uciml/sms-spam-collection-dataset',
        'news': 'rmisra/news-category-dataset',
        'toxic-comments': 'julian3833/jigsaw-toxic-comment-classification-challenge',

        # 推薦系統
        'movies': 'grouplens/movielens-20m-dataset',
        'books': 'zygmunt/goodbooks-10k',
        'music': 'pieca111/music-artists-popularity',

        # 計算機視覺
        'mnist': 'oddrationale/mnist-in-csv',
        'fashion-mnist': 'zalando-research/fashionmnist',
        'cifar10': 'swaroopkml/cifar10-pngs-in-folders',
        'cats-dogs': 'tongpython/cat-and-dog',
        'plant-disease': 'vipoooool/new-plant-diseases-dataset',

        # 其他
        'customer-segmentation': 'vjchoudhary7/customer-segmentation-tutorial-in-python',
        'fraud-detection': 'kartik2112/fraud-detection',
        'predictive-maintenance': 'shivamb/machine-predictive-maintenance-classification',
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """初始化下載器

        Args:
            data_dir: 數據存儲目錄，默認為專案根目錄的 data/
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / 'data'

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 檢查 Kaggle API 是否可用
        self.kaggle_available = self._check_kaggle_api()

    def _check_kaggle_api(self) -> bool:
        """檢查 Kaggle API 是否可用

        Returns:
            bool: 是否可用
        """
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Kaggle API 已安裝: {result.stdout.strip()}")

                # 檢查憑證
                kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
                if kaggle_json.exists():
                    print(f"✓ Kaggle API 憑證已配置")
                    return True
                else:
                    print("⚠️  Kaggle API 憑證未配置")
                    print("請參考: https://www.kaggle.com/docs/api")
                    return False
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Kaggle API 未安裝")
            print("請運行: pip install kaggle")
            return False

    def download_dataset(
        self,
        dataset_name: str,
        destination: Optional[Path] = None,
        unzip: bool = True,
        force: bool = False
    ) -> Path:
        """下載 Kaggle 數據集

        Args:
            dataset_name: 數據集名稱（可以是簡稱或完整路徑）
            destination: 目標目錄，默認為 data/raw/{dataset_name}
            unzip: 是否解壓
            force: 是否強制重新下載

        Returns:
            Path: 下載的數據集路徑
        """
        if not self.kaggle_available:
            raise RuntimeError("Kaggle API 不可用，請先配置")

        # 解析數據集名稱
        if dataset_name in self.POPULAR_DATASETS:
            full_name = self.POPULAR_DATASETS[dataset_name]
            print(f"📦 使用預設數據集: {dataset_name} -> {full_name}")
        else:
            full_name = dataset_name

        # 設置目標目錄
        if destination is None:
            safe_name = full_name.replace('/', '_')
            destination = self.data_dir / 'raw' / safe_name

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        # 檢查是否已存在
        if not force and any(destination.iterdir()):
            print(f"⏭️  數據集已存在: {destination}")
            return destination

        print(f"⬇️  開始下載數據集: {full_name}")
        print(f"📁 目標目錄: {destination}")

        try:
            # 使用 Kaggle CLI 下載
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', full_name,
                '-p', str(destination)
            ]

            if unzip:
                cmd.append('--unzip')

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分鐘超時
            )

            if result.returncode == 0:
                print(f"✅ 下載完成: {destination}")

                # 列出下載的文件
                files = list(destination.glob('*'))
                if files:
                    print(f"\n📄 下載的文件 ({len(files)} 個):")
                    for f in files[:10]:  # 只顯示前10個
                        size = f.stat().st_size if f.is_file() else 0
                        size_mb = size / 1024 / 1024
                        print(f"  - {f.name} ({size_mb:.2f} MB)")
                    if len(files) > 10:
                        print(f"  ... 還有 {len(files) - 10} 個文件")

                return destination
            else:
                error_msg = result.stderr or result.stdout
                print(f"❌ 下載失敗: {error_msg}")
                raise RuntimeError(f"下載失敗: {error_msg}")

        except subprocess.TimeoutExpired:
            print("❌ 下載超時（10分鐘）")
            raise RuntimeError("下載超時")
        except Exception as e:
            print(f"❌ 下載出錯: {e}")
            raise

    def download_competition_data(
        self,
        competition_name: str,
        destination: Optional[Path] = None,
        force: bool = False
    ) -> Path:
        """下載 Kaggle 競賽數據

        Args:
            competition_name: 競賽名稱
            destination: 目標目錄
            force: 是否強制重新下載

        Returns:
            Path: 下載的數據集路徑
        """
        if not self.kaggle_available:
            raise RuntimeError("Kaggle API 不可用，請先配置")

        if destination is None:
            destination = self.data_dir / 'competitions' / competition_name

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        if not force and any(destination.iterdir()):
            print(f"⏭️  競賽數據已存在: {destination}")
            return destination

        print(f"⬇️  開始下載競賽數據: {competition_name}")

        try:
            cmd = [
                'kaggle', 'competitions', 'download',
                '-c', competition_name,
                '-p', str(destination)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                print(f"✅ 下載完成: {destination}")

                # 解壓所有 zip 文件
                import zipfile
                for zip_file in destination.glob('*.zip'):
                    print(f"📦 解壓: {zip_file.name}")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(destination)
                    zip_file.unlink()  # 刪除 zip 文件

                return destination
            else:
                error_msg = result.stderr or result.stdout
                print(f"❌ 下載失敗: {error_msg}")
                raise RuntimeError(f"下載失敗: {error_msg}")

        except Exception as e:
            print(f"❌ 下載出錯: {e}")
            raise

    def list_popular_datasets(self):
        """列出常用數據集"""
        print("\n" + "=" * 80)
        print("常用 Kaggle 數據集")
        print("=" * 80)

        categories = {
            '結構化數據': ['titanic', 'house-prices', 'credit-fraud', 'customer-churn',
                      'bank-marketing', 'wine-quality', 'adult-income'],
            '時間序列': ['bitcoin', 'stock-market', 'sales', 'energy', 'covid19'],
            'NLP': ['sentiment', 'spam', 'news', 'toxic-comments'],
            '推薦系統': ['movies', 'books', 'music'],
            '計算機視覺': ['mnist', 'fashion-mnist', 'cifar10', 'cats-dogs', 'plant-disease'],
            '其他': ['customer-segmentation', 'fraud-detection', 'predictive-maintenance'],
        }

        for category, datasets in categories.items():
            print(f"\n【{category}】")
            for ds in datasets:
                full_name = self.POPULAR_DATASETS[ds]
                print(f"  {ds:30s} -> {full_name}")

        print("\n" + "=" * 80)
        print("使用方法:")
        print("  downloader.download_dataset('titanic')  # 使用簡稱")
        print("  downloader.download_dataset('username/dataset-name')  # 使用完整路徑")
        print("=" * 80 + "\n")

    def search_datasets(self, keyword: str, max_results: int = 10) -> List[Dict]:
        """搜索 Kaggle 數據集

        Args:
            keyword: 搜索關鍵詞
            max_results: 最大結果數

        Returns:
            List[Dict]: 搜索結果列表
        """
        if not self.kaggle_available:
            print("❌ Kaggle API 不可用")
            return []

        print(f"🔍 搜索關鍵詞: {keyword}")

        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '-s', keyword, '--max-size', str(max_results)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(result.stdout)
                return []
            else:
                print(f"❌ 搜索失敗: {result.stderr}")
                return []

        except Exception as e:
            print(f"❌ 搜索出錯: {e}")
            return []


def setup_kaggle_credentials():
    """設置 Kaggle API 憑證的互動式指南"""
    print("\n" + "=" * 80)
    print("Kaggle API 憑證設置指南")
    print("=" * 80)

    print("\n步驟 1: 獲取 API Token")
    print("  1. 登錄 Kaggle: https://www.kaggle.com")
    print("  2. 進入 Account 設置: https://www.kaggle.com/settings")
    print("  3. 滾動到 'API' 部分")
    print("  4. 點擊 'Create New API Token'")
    print("  5. 下載 kaggle.json 文件")

    print("\n步驟 2: 配置憑證")
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    print(f"\n  目標位置: {kaggle_json}")

    if kaggle_json.exists():
        print(f"  ✅ 憑證文件已存在")
    else:
        kaggle_dir.mkdir(exist_ok=True)
        print(f"  📁 創建目錄: {kaggle_dir}")
        print(f"  ⚠️  請手動將 kaggle.json 複製到: {kaggle_json}")

    print("\n步驟 3: 設置權限 (Linux/Mac)")
    print(f"  chmod 600 {kaggle_json}")

    print("\n步驟 4: 測試")
    print("  kaggle datasets list")

    print("\n" + "=" * 80 + "\n")


# 便捷函數
def quick_download(dataset_name: str, force: bool = False) -> Path:
    """快速下載數據集

    Args:
        dataset_name: 數據集名稱
        force: 是否強制重新下載

    Returns:
        Path: 數據集路徑
    """
    downloader = KaggleDatasetDownloader()
    return downloader.download_dataset(dataset_name, force=force)


def main():
    """主函數 - 演示用法"""
    print("=" * 80)
    print("Kaggle 數據集下載器")
    print("=" * 80)

    downloader = KaggleDatasetDownloader()

    # 列出常用數據集
    downloader.list_popular_datasets()

    # 示例：下載 Titanic 數據集
    print("\n示例：下載 Titanic 數據集")
    print("-" * 80)

    try:
        data_path = downloader.download_dataset('titanic')
        print(f"\n✅ 數據集路徑: {data_path}")

        # 列出文件
        print("\n文件列表:")
        for file in data_path.glob('*'):
            print(f"  - {file.name}")

    except Exception as e:
        print(f"\n❌ 下載失敗: {e}")
        print("\n如果還沒有配置 Kaggle API，請運行:")
        print("  from data_analysis_chatbots.kaggle_downloader import setup_kaggle_credentials")
        print("  setup_kaggle_credentials()")


if __name__ == '__main__':
    main()
