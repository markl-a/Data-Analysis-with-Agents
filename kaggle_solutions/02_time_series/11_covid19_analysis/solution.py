"""
COVID-19數據分析
疫情數據趨勢分析和預測
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class COVID19Analyzer:
    def create_data(self, n=365):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        cases = np.cumsum(np.random.poisson(100, n))
        df = pd.DataFrame({
            'Date': dates,
            'Cases': cases,
            'Deaths': (cases * 0.02 * np.random.uniform(0.8, 1.2, n)).astype(int)
        })
        return df

    def analyze(self, df):
        df['NewCases'] = df['Cases'].diff().fillna(0)
        df['7DayAvg'] = df['NewCases'].rolling(7).mean()

        print(f"總確診: {df['Cases'].iloc[-1]:,}")
        print(f"總死亡: {df['Deaths'].iloc[-1]:,}")
        print(f"死亡率: {df['Deaths'].iloc[-1] / df['Cases'].iloc[-1]:.2%}")

        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['NewCases'], alpha=0.3, label='每日新增')
        plt.plot(df['Date'], df['7DayAvg'], label='7日平均')
        plt.legend()
        plt.title('COVID-19 新增確診趨勢')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('covid19_trend.png', dpi=300, bbox_inches='tight')
        print("圖表已保存")

if __name__ == "__main__":
    print("COVID-19 數據分析")
    analyzer = COVID19Analyzer()
    df = analyzer.create_data()
    analyzer.analyze(df)
