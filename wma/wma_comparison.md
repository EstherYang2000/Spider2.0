# 比較原始 WMA 與 Cross-Consistency 增強版 WMA

## 1. 實作概述

### 原始 WMA (Weighted Majority Algorithm)

原始 WMA 是一個簡單但強大的集成學習演算法，主要特點：

- 每位專家都有一個權重，初始值通常為 1.0
- 當專家預測錯誤時，其權重會乘以一個衰減因子 (1-epsilon)
- 當專家預測正確時，權重保持不變
- 最終預測是由所有專家根據其權重進行加權投票決定

```python
# 核心權重更新邏輯
def update_weights(self, expert_name: str, is_correct: bool):
    if not is_correct:
        old_weight = self.experts[expert_name]
        self.experts[expert_name] = old_weight * (1 - self.epsilon)
```

### Cross-Consistency 增強版 WMA

Cross-Consistency 增強版在原始 WMA 的基礎上，加入了專家間一致性的考量：

- 保留原始 WMA 的所有功能
- 增加計算專家間一致性的機制 (使用 Jaccard 相似度)
- 根據一致性分數給予專家額外的權重獎勵
- 記錄專家一致性的歷史趨勢
- 提供可調整的一致性獎勵參數

```python
# 一致性獎勵邏輯
def apply_consistency_bonus(self, consistency_scores):
    for expert, score in consistency_scores.items():
        if expert in self.experts:
            # 根據一致性分數給予獎勵
            self.experts[expert] *= (1 + self.consistency_bonus * score)
```

## 2. 關鍵差異

| 特性 | 原始 WMA | Cross-Consistency WMA |
|------|----------|----------------------|
| **權重更新機制** | 僅基於正確性 | 基於正確性 + 一致性 |
| **專家獎勵** | 無獎勵機制，只有懲罰 | 對一致性高的專家有獎勵機制 |
| **歷史記錄** | 不記錄歷史 | 記錄專家一致性歷史 |
| **可調參數** | epsilon (衰減因子) | epsilon + consistency_bonus (一致性獎勵) |
| **記憶體使用** | 較低 | 較高 (需存儲一致性歷史) |
| **計算複雜度** | O(n) | O(n²) (需計算專家間兩兩一致性) |

## 3. 測試結果分析

我們使用 5 個測試案例比較了兩種實作：

```
Test Cases: 5
Original WMA Accuracy: 0.60
Cross-Consistency WMA Accuracy: 0.40
```

### 案例分析

1. **案例 1 (兩者都正確)**: 大多數專家意見一致且正確
   - 兩種方法都選擇了正確的 SQL

2. **案例 2 (兩者都錯誤)**: 專家意見分歧
   - 兩種方法都選擇了錯誤的 SQL

3. **案例 3 (兩者都錯誤)**: 多數專家錯誤，少數專家正確
   - 兩種方法都選擇了錯誤的 SQL (多數決的局限性)

4. **案例 4 (原始正確，增強版錯誤)**: 一位專家始終正確，其他專家始終錯誤
   - 原始 WMA 選擇了正確的 SQL
   - Cross-Consistency WMA 因為過度獎勵一致性而選擇了錯誤的 SQL

5. **案例 5 (兩者都正確)**: 專家一致性程度不同
   - 兩種方法都選擇了正確的 SQL

### 權重分析

最終專家權重比較：

```
Final Expert Weights:
--------------------------------------------------------------------------------
  Expert   |  Original WMA   |   Cross-Consistency WMA   |  Consistency Score
--------------------------------------------------------------------------------
 expert1   | 1.0000         | 1.0681                   | 0.2667
 expert2   | 0.9703         | 1.0709                   | 0.4000
 expert3   | 0.9606         | 1.0092                   | 0.2000
 expert4   | 0.9606         | 1.0428                   | 0.3333
```

- 原始 WMA: 權重只會減少，不會增加，所以最高權重為 1.0
- Cross-Consistency WMA: 權重可以超過 1.0，因為一致性獎勵機制

## 4. 適用場景

### 原始 WMA 適用場景

- 專家正確性是唯一重要的因素
- 資料集中存在明確的正確答案
- 計算資源有限，需要更高效的演算法
- 專家間的一致性不是重要的信號

### Cross-Consistency WMA 適用場景

- 專家間的一致性是重要的信號
- 沒有明確的正確答案，需要依賴專家共識
- 需要平衡專家的正確性和一致性
- 有足夠的計算資源處理更複雜的演算法
- 需要追蹤專家一致性的歷史趨勢

## 5. 參數調整建議

### 原始 WMA

- **epsilon**: 控制錯誤預測的懲罰程度
  - 較小的值 (如 0.001): 緩慢調整，適合穩定的專家
  - 較大的值 (如 0.1): 快速調整，適合變化大的環境

### Cross-Consistency WMA

- **epsilon**: 同原始 WMA
- **consistency_bonus**: 控制一致性獎勵的程度
  - 較小的值 (如 0.01): 輕微獎勵一致性，適合正確性更重要的場景
  - 較大的值 (如 0.1): 大幅獎勵一致性，適合共識更重要的場景

## 6. 結論與建議

- **原始 WMA** 是一個簡單、高效且專注於正確性的演算法，適合有明確正確答案的場景。
- **Cross-Consistency WMA** 引入了專家間一致性的考量，適合需要平衡正確性和共識的場景。
- 在實際應用中，建議根據具體需求選擇合適的演算法，並通過實驗調整參數。
- 對於 SQL 生成這類任務，如果有明確的正確答案，原始 WMA 可能更適合；如果需要依賴專家共識，Cross-Consistency WMA 可能更有優勢。
- 一致性獎勵參數 (consistency_bonus) 需要謹慎調整，過高可能導致錯誤的共識被過度獎勵。

## 7. 未來改進方向

1. **動態調整一致性獎勵**: 根據歷史表現自動調整一致性獎勵參數
2. **分層一致性計算**: 考慮不同層次的一致性 (如語法一致性、語義一致性)
3. **專家分組**: 將專家分組，先在組內達成共識，再在組間進行投票
4. **時間衰減**: 引入時間因素，使較新的一致性記錄有更高的權重
5. **混合模型**: 結合其他集成學習方法，如 Boosting 或 Stacking
