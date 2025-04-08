import logging
from collections import defaultdict

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class WeightedMajorityAlgorithm:
    """
    實作 WMA (Weighted Majority Algorithm) 的簡易版本：
      - 每位「專家」(expert) 都有一個權重
      - 若該專家產生的最終預測是錯誤的，就降低 (衰減) 其權重
      - 若正確，則保持或輕微增幅 (本範例預設不增，只維持)
    """

    def __init__(self, experts=None, epsilon=0.005, consistency_bonus=0.02):
        """
        初始化 WMA 演算法。
        
        :param experts: dict 或 None
            - 若為 dict, 形如 {"expert_name1": 1.0, "expert_name2": 1.0, ...}
            - 若為 None, 則預設為空，在運行時再動態加入專家
        :param epsilon: float, 衰減比例(0<epsilon<1)，專家預測錯誤時要乘的因子 (1 - epsilon)
        :param consistency_bonus: float, 一致性獎勵比例，當專家與其他專家預測一致時的獎勵
        """
        if experts is None:
            experts = {}
        self.experts = experts      # {expert_name: weight}
        self.epsilon = epsilon      # 衰減比例
        self.consistency_bonus = consistency_bonus  # 一致性獎勵比例
        self.expert_agreement_history = defaultdict(list)  # 記錄專家間的一致性歷史

    def add_expert(self, expert_name: str, init_weight: float = 1.0):
        """
        Add a new expert to the algorithm with an initial weight.
        
        Args:
            expert_name (str): Name/identifier of the expert
            init_weight (float): Initial weight for the expert (default: 1.0)
        """
        if expert_name not in self.experts:
            self.experts[expert_name] = init_weight

    def update_weights(self, expert_name: str, is_correct: bool):
        """
        Update the weight of a single expert based on their prediction correctness.
        
        Args:
            expert_name (str): The name of the expert whose weight should be updated
            is_correct (bool): Whether the expert's prediction was correct
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")

        if not is_correct:
            # Only decrease weight if the prediction was incorrect
            old_weight = self.experts[expert_name]
            self.experts[expert_name] = old_weight * (1 - self.epsilon)

    def calculate_cross_consistency(self, predictions_dict):
        """
        Calculate cross-consistency scores between experts based on their SQL predictions.
        
        Args:
            predictions_dict (dict): Dictionary mapping expert names to their SQL predictions
                {
                    "expert_1": ["SQL A", "SQL B"],
                    "expert_2": ["SQL B", "SQL C"],
                    ...
                }
                
        Returns:
            dict: Dictionary mapping expert names to their consistency scores
        """
        if not predictions_dict or len(predictions_dict) <= 1:
            return {expert: 0.0 for expert in predictions_dict}
        
        # Initialize consistency scores
        consistency_scores = {expert: 0.0 for expert in predictions_dict}
        
        # Calculate agreement between each pair of experts
        for expert1, sqls1 in predictions_dict.items():
            for expert2, sqls2 in predictions_dict.items():
                if expert1 == expert2:
                    continue
                
                # Calculate Jaccard similarity between SQL sets
                set1 = set(sqls1)
                set2 = set(sqls2)
                
                if not set1 or not set2:
                    continue
                    
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                if union > 0:
                    similarity = intersection / union
                    consistency_scores[expert1] += similarity
        
        # Normalize scores by the number of other experts
        num_other_experts = len(predictions_dict) - 1
        if num_other_experts > 0:
            for expert in consistency_scores:
                consistency_scores[expert] /= num_other_experts
                
                # Update agreement history
                self.expert_agreement_history[expert].append(consistency_scores[expert])
                # Keep only the last 10 entries to avoid unbounded growth
                if len(self.expert_agreement_history[expert]) > 10:
                    self.expert_agreement_history[expert] = self.expert_agreement_history[expert][-10:]
        
        return consistency_scores

    def apply_consistency_bonus(self, consistency_scores):
        """
        Apply consistency bonus to expert weights based on their cross-consistency scores.
        
        Args:
            consistency_scores (dict): Dictionary mapping expert names to their consistency scores
        """
        for expert, score in consistency_scores.items():
            if expert in self.experts:
                # Apply bonus proportional to consistency score
                self.experts[expert] *= (1 + self.consistency_bonus * score)
                logger.debug(f"Applied consistency bonus to {expert}: score={score:.4f}, new weight={self.experts[expert]:.4f}")

    def get_expert_consistency_trend(self, expert_name):
        """
        Get the trend of consistency scores for a specific expert.
        
        Args:
            expert_name (str): Name of the expert
            
        Returns:
            float: Average consistency score over recent history, or 0 if no history
        """
        history = self.expert_agreement_history.get(expert_name, [])
        if not history:
            return 0.0
        return sum(history) / len(history)

    def weighted_majority_vote(self, predictions_dict, apply_consistency=True):
        """
        Perform a weighted majority vote where each expert provides a list of SQLs.
        Optionally applies cross-consistency bonuses to expert weights.
        
        Args:
            predictions_dict (dict): Dictionary mapping expert names to their SQL predictions
                {
                    "expert_1": ["SQL A", "SQL B"],
                    "expert_2": ["SQL B", "SQL C"],
                    ...
                }
            apply_consistency (bool): Whether to apply consistency bonuses to weights
            
        Returns:
            tuple: (best_sql, chosen_experts, best_weight)
                - best_sql: 獲勝SQL
                - chosen_experts: 推薦這條SQL的專家列表
                - best_weight: 這條SQL累積的總加權分數
        """
        sql_to_weight = {}
        sql_to_experts = {}

        # Check if predictions_dict is empty
        if not predictions_dict:
            logger.error("No SQL predictions received from any expert.")
            return None, [], 0.0
        
        # Calculate cross-consistency scores
        consistency_scores = self.calculate_cross_consistency(predictions_dict)
        
        # Apply consistency bonus if requested
        if apply_consistency:
            self.apply_consistency_bonus(consistency_scores)
            
        # Log consistency scores and current weights
        for expert, score in consistency_scores.items():
            logger.debug(f"Expert {expert} consistency score: {score:.4f}, weight: {self.experts.get(expert, 1.0):.4f}")
        
        # Accumulate weights for each SQL
        for expert_name, sql_list in predictions_dict.items():
            expert_weight = self.experts.get(expert_name, 1.0)
            # 每條候選SQL都獲得該專家的全部權重 (Group Voting)
            for sql_str in sql_list:
                if sql_str not in sql_to_weight:
                    sql_to_weight[sql_str] = 0.0
                    sql_to_experts[sql_str] = []
                sql_to_weight[sql_str] += expert_weight
                sql_to_experts[sql_str].append(expert_name)
                
        # If no valid SQLs were added, return a safe fallback
        if not sql_to_weight:
            logger.error("No valid SQLs received from experts.")
            return None, [], 0.0
            
        # 選出加權分數最高的SQL
        best_sql = max(sql_to_weight, key=sql_to_weight.get)
        best_weight = sql_to_weight[best_sql]
        chosen_experts = list(set(sql_to_experts[best_sql]))

        return best_sql, chosen_experts, best_weight

    def get_weights(self):
        """
        Get current weights of all experts.
        
        Returns:
            dict: Dictionary mapping expert names to their current weights
        """
        return self.experts.copy()
        
    def get_expert_weight(self, expert_name: str) -> float:
        """
        Get the current weight of a specific expert.
        
        Args:
            expert_name (str): Name of the expert
            
        Returns:
            float: Current weight of the expert
            
        Raises:
            ValueError: If expert_name is not found
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")
        return self.experts[expert_name]
        
    def get_consistency_scores(self):
        """
        Get the average consistency scores for all experts.
        
        Returns:
            dict: Dictionary mapping expert names to their average consistency scores
        """
        return {expert: sum(scores)/len(scores) if scores else 0.0 
                for expert, scores in self.expert_agreement_history.items()}
