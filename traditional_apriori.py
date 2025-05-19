from collections import defaultdict
from itertools import combinations
from typing import List, Set, Dict, Tuple

class AprioriAlgorithm:
    """
    Traditional Apriori Algorithm Implementation
    Time Complexity: O(N×M×w + ∑(|L(k-1)|²))
    Space Complexity: O(NM + w×|C|)
    where:
    N = number of transactions
    M = number of unique items
    w = average transaction width
    |L(k-1)| = size of frequent itemsets at level k-1
    |C| = size of candidate set
    """
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.item_counts = defaultdict(int)
        self.frequent_itemsets = {}
        self.rules = []
        
    def fit(self, transactions: List[List[str]]) -> None:
        """
        Fit the Apriori algorithm to the transaction data.
        Time Complexity: O(N×M×w)
        """
        self.transactions = transactions
        self.n_transactions = len(transactions)
        
        # Generate frequent 1-itemsets
        # Time: O(N×w) where w is average transaction width
        for transaction in transactions:
            for item in transaction:
                self.item_counts[frozenset([item])] += 1
        
        # Filter by minimum support
        # Time: O(M) where M is number of unique items
        self.frequent_itemsets[1] = {
            itemset: count for itemset, count in self.item_counts.items()
            if count / self.n_transactions >= self.min_support
        }
        
        # Generate frequent k-itemsets
        k = 2
        while self.frequent_itemsets[k-1]:
            self._generate_frequent_itemsets(k)
            k += 1
            
        self._generate_rules()
    
    def _generate_frequent_itemsets(self, k: int) -> None:
        """
        Generate frequent k-itemsets.
        Time Complexity: O(|L(k-1)|²) + O(N×w)
        """
        # Generate candidates
        # Time: O(|L(k-1)|²)
        candidates = self._generate_candidates(k)
        
        # Count support for candidates
        # Time: O(N×w)
        candidate_counts = defaultdict(int)
        for transaction in self.transactions:
            transaction_set = frozenset(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
        
        # Filter by minimum support
        self.frequent_itemsets[k] = {
            itemset: count for itemset, count in candidate_counts.items()
            if count / self.n_transactions >= self.min_support
        }
    
    def _generate_candidates(self, k: int) -> Set[frozenset]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets.
        Time Complexity: O(|L(k-1)|²)
        """
        frequent_items = list(self.frequent_itemsets[k-1].keys())
        candidates = set()
        
        for i in range(len(frequent_items)):
            for j in range(i + 1, len(frequent_items)):
                items1 = list(frequent_items[i])
                items2 = list(frequent_items[j])
                
                # Items must share first k-2 items to be joinable
                if items1[:-1] == items2[:-1]:
                    # Create new candidate
                    new_candidate = frozenset(items1 + [items2[-1]])
                    
                    # Check if all (k-1) subsets are frequent
                    if all(
                        frozenset(subset) in self.frequent_itemsets[k-1]
                        for subset in [
                            new_candidate - {item}
                            for item in new_candidate
                        ]
                    ):
                        candidates.add(new_candidate)
        
        return candidates
    
    def _generate_rules(self) -> None:
        """
        Generate association rules from frequent itemsets.
        Time Complexity: O(∑(2^k × |L(k)|))
        where |L(k)| is the number of frequent k-itemsets
        """
        self.rules = []
        
        # For each k-itemset where k >= 2
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, support_count in self.frequent_itemsets[k].items():
                # Generate all possible antecedents
                for i in range(1, k):
                    for antecedent in self._get_subsets(itemset, i):
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        confidence = (
                            support_count /
                            self.frequent_itemsets[len(antecedent)][antecedent]
                        )
                        
                        if confidence >= self.min_confidence:
                            support = support_count / self.n_transactions
                            self.rules.append((
                                antecedent,
                                consequent,
                                support,
                                confidence
                            ))
    
    def _get_subsets(self, itemset: frozenset, length: int) -> List[frozenset]:
        """Generate all subsets of given length from itemset."""
        return [
            frozenset(combo)
            for combo in combinations(itemset, length)
        ]
    
    def get_rules(self) -> List[Tuple[frozenset, frozenset, float, float]]:
        """
        Return generated association rules.
        Each rule is a tuple: (antecedent, consequent, support, confidence)
        """
        return self.rules

def print_rules(rules: List[Tuple[frozenset, frozenset, float, float]]) -> None:
    """Pretty print the generated rules."""
    print("\nGenerated Association Rules:")
    print("============================")
    for antecedent, consequent, support, confidence in rules:
        print(f"{set(antecedent)} => {set(consequent)}")
        print(f"Support: {support:.3f}")
        print(f"Confidence: {confidence:.3f}")
        print("----------------------------")

# Example usage
if __name__ == "__main__":
    # Sample transaction database
    transactions = [
        ["bread", "milk", "eggs"],
        ["bread", "butter"],
        ["milk", "butter"],
        ["bread", "milk", "butter"],
        ["bread", "milk"],
    ]
    
    # Initialize and run Apriori algorithm
    apriori = AprioriAlgorithm(min_support=0.3, min_confidence=0.5)
    apriori.fit(transactions)
    
    # Get and print the rules
    rules = apriori.get_rules()
    print_rules(rules) 