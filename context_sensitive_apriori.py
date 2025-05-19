from collections import defaultdict
from itertools import combinations
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class ContextMetadata:
    """Context metadata for a transaction."""
    temporal: Dict[str, str]  # time-based context (time of day, day of week, season)
    spatial: Dict[str, str]   # location-based context
    user: Dict[str, str]      # user demographic and behavior context

@dataclass
class ContextualTransaction:
    """A transaction with its associated context."""
    items: List[str]
    context: ContextMetadata
    timestamp: datetime

class ContextSensitiveApriori:
    """
    Context-Sensitive Apriori Algorithm Implementation
    Time Complexity: O(N + C×N×M×w + C×∑(|L(k-1)|²))
    Space Complexity: O(C×NM + C×w×|C| + N×K)
    where:
    N = number of transactions
    M = number of unique items
    w = average transaction width
    C = number of contexts
    K = context-specific parameters
    |L(k-1)| = size of frequent itemsets at level k-1
    |C| = size of candidate set
    """
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.contexts = {}  # Store identified contexts
        self.context_itemsets = defaultdict(dict)  # Frequent itemsets per context
        self.context_rules = defaultdict(list)  # Rules per context
        self.performance_stats = {}
    
    def _extract_contexts(self, transactions: List[ContextualTransaction]) -> None:
        """
        Extract and organize contexts from transactions.
        Time Complexity: O(N)
        """
        # Group transactions by context combinations
        for transaction in transactions:
            # Create a context key by combining relevant context attributes
            context_key = self._create_context_key(transaction.context)
            if context_key not in self.contexts:
                self.contexts[context_key] = []
            self.contexts[context_key].append(transaction)
    
    def _create_context_key(self, context: ContextMetadata) -> str:
        """Create a unique key for a context combination."""
        temporal_key = ','.join(f"{k}={v}" for k, v in sorted(context.temporal.items()))
        spatial_key = ','.join(f"{k}={v}" for k, v in sorted(context.spatial.items()))
        user_key = ','.join(f"{k}={v}" for k, v in sorted(context.user.items()))
        return f"{temporal_key}|{spatial_key}|{user_key}"
    
    def fit(self, transactions: List[ContextualTransaction]) -> None:
        """
        Fit the Context-Sensitive Apriori algorithm to the transaction data.
        Time Complexity: O(N + C×N×M×w)
        """
        self.transactions = transactions
        self._extract_contexts(transactions)
        
        # Process each context separately
        for context_key, context_transactions in self.contexts.items():
            # Initialize context-specific counts
            item_counts = defaultdict(int)
            n_transactions = len(context_transactions)
            
            # Generate frequent 1-itemsets for this context
            for transaction in context_transactions:
                for item in transaction.items:
                    item_counts[frozenset([item])] += 1
            
            # Filter by minimum support
            self.context_itemsets[context_key][1] = {
                itemset: count for itemset, count in item_counts.items()
                if count / n_transactions >= self.min_support
            }
            
            # Generate frequent k-itemsets for this context
            k = 2
            while self.context_itemsets[context_key][k-1]:
                self._generate_context_frequent_itemsets(
                    context_key,
                    context_transactions,
                    k
                )
                k += 1
            
            # Generate rules for this context
            self._generate_context_rules(context_key, n_transactions)
    
    def _generate_context_frequent_itemsets(
        self,
        context_key: str,
        context_transactions: List[ContextualTransaction],
        k: int
    ) -> None:
        """
        Generate frequent k-itemsets for a specific context.
        Time Complexity: O(|L(k-1)|²) + O(N×w)
        """
        # Generate candidates
        candidates = self._generate_candidates(context_key, k)
        
        # Count support for candidates
        n_transactions = len(context_transactions)
        candidate_counts = defaultdict(int)
        
        for transaction in context_transactions:
            transaction_set = frozenset(transaction.items)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
        
        # Filter by minimum support
        self.context_itemsets[context_key][k] = {
            itemset: count for itemset, count in candidate_counts.items()
            if count / n_transactions >= self.min_support
        }
    
    def _generate_candidates(self, context_key: str, k: int) -> Set[frozenset]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets for a context.
        Time Complexity: O(|L(k-1)|²)
        """
        frequent_items = list(self.context_itemsets[context_key][k-1].keys())
        candidates = set()
        
        for i in range(len(frequent_items)):
            for j in range(i + 1, len(frequent_items)):
                items1 = list(frequent_items[i])
                items2 = list(frequent_items[j])
                
                if items1[:-1] == items2[:-1]:
                    new_candidate = frozenset(items1 + [items2[-1]])
                    
                    if all(
                        frozenset(subset) in self.context_itemsets[context_key][k-1]
                        for subset in [
                            new_candidate - {item}
                            for item in new_candidate
                        ]
                    ):
                        candidates.add(new_candidate)
        
        return candidates
    
    def _generate_context_rules(self, context_key: str, n_transactions: int) -> None:
        """
        Generate association rules for a specific context.
        Time Complexity: O(∑(2^k × |L(k)|))
        """
        context_rules = []
        
        # For each k-itemset where k >= 2
        for k in range(2, len(self.context_itemsets[context_key]) + 1):
            if k not in self.context_itemsets[context_key]:
                continue
                
            for itemset, support_count in self.context_itemsets[context_key][k].items():
                # Generate all possible antecedents
                for i in range(1, k):
                    for antecedent in self._get_subsets(itemset, i):
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        confidence = (
                            support_count /
                            self.context_itemsets[context_key][len(antecedent)][antecedent]
                        )
                        
                        if confidence >= self.min_confidence:
                            support = support_count / n_transactions
                            context_rules.append((
                                antecedent,
                                consequent,
                                support,
                                confidence,
                                context_key
                            ))
        
        self.context_rules[context_key] = context_rules
    
    def _get_subsets(self, itemset: frozenset, length: int) -> List[frozenset]:
        """Generate all subsets of given length from itemset."""
        from itertools import combinations
        return [
            frozenset(combo)
            for combo in combinations(itemset, length)
        ]
    
    def get_rules(self, context_key: Optional[str] = None) -> List[Tuple]:
        """
        Return generated association rules, optionally filtered by context.
        Each rule is a tuple: (antecedent, consequent, support, confidence, context)
        """
        if context_key:
            return self.context_rules.get(context_key, [])
        
        # Combine rules from all contexts
        all_rules = []
        for rules in self.context_rules.values():
            all_rules.extend(rules)
        return all_rules
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Return performance statistics for the algorithm."""
        stats = {
            'num_contexts': len(self.contexts),
            'avg_rules_per_context': sum(
                len(rules) for rules in self.context_rules.values()
            ) / len(self.contexts) if self.contexts else 0,
            'total_rules': sum(len(rules) for rules in self.context_rules.values()),
        }
        return stats

# Example usage
if __name__ == "__main__":
    # Create sample contextual transactions
    sample_transactions = [
        ContextualTransaction(
            items=["bread", "milk", "eggs"],
            context=ContextMetadata(
                temporal={"time_of_day": "morning", "day_of_week": "monday", "season": "summer"},
                spatial={"location_type": "supermarket", "region": "urban"},
                user={"demographic": "family", "behavior_segment": "regular"}
            ),
            timestamp=datetime.now()
        ),
        # Add more transactions...
    ]

    # Initialize and run algorithm
    cs_apriori = ContextSensitiveApriori(min_support=0.3, min_confidence=0.5)
    cs_apriori.fit(sample_transactions)
    
    # Get and print rules
    rules = cs_apriori.get_rules()
    stats = cs_apriori.get_performance_stats()
    
    print("\nPerformance Statistics:")
    for metric, value in stats.items():
        print(f"{metric}: {value:.4f}") 