# 126. 強化學習入門 - OpenAI Gym

## 項目概述

這是一個強化學習入門項目，使用經典的 CartPole 環境來學習 Q-Learning 和 Deep Q-Network (DQN) 的基本概念，為更複雜的強化學習任務奠定基礎。

**Kaggle 數據集**: [OpenAI Gym CartPole](https://www.kaggle.com/datasets/balajibaskar/openai-gym-cartpole)

**難度**: ⭐⭐⭐ 進階

## 目標

訓練智能體學會平衡倒立擺（最大化累積獎勵）

## 環境描述

### 狀態空間

| 狀態 | 描述 | 範圍 |
|------|------|------|
| Cart Position | 車的位置 | [-4.8, 4.8] |
| Cart Velocity | 車的速度 | [-Inf, Inf] |
| Pole Angle | 桿的角度 | [-24°, 24°] |
| Pole Angular Velocity | 桿的角速度 | [-Inf, Inf] |

### 動作空間

| 動作 | 描述 |
|------|------|
| 0 | 向左推車 |
| 1 | 向右推車 |

### 獎勵機制

- 每個時間步 +1 獎勵
- 桿倒下或車出界則終止

## 關鍵概念

1. **Q-Learning**: 通過 Q 表學習最優動作價值
2. **探索與利用**: ε-greedy 策略平衡探索與利用
3. **經驗回放**: 打破經驗相關性
4. **目標網絡**: 穩定訓練過程

## 技術方法

### 算法
- Q-Learning (表格方法)
- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN

### 超參數
- 學習率 (α): 0.001
- 折扣因子 (γ): 0.99
- 探索率 (ε): 1.0 → 0.01
- 批次大小: 64
- 記憶容量: 10000

### 評估指標
- 平均回合獎勵
- 成功率（達到 200 步）
- 學習曲線

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐⭐ 進階
**預計完成時間**: 4-5 小時
**推薦給**: 對強化學習感興趣的學習者
