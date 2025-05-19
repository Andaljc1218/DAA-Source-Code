import sys
import time
import random
from datetime import datetime
from memory_profiler import profile
from typing import List, Tuple

sys.path.append('../')
from algorithm.traditional_apriori import AprioriAlgorithm
from algorithm.context_sensitive_apriori import ContextSensitiveApriori, ContextualTransaction, ContextMetadata

def generate_synthetic_data(
    num_transactions: int,
    num_items: int,
    max_items_per_transaction: int,
    num_contexts: int
) -> Tuple[List[List[str]], List[ContextualTransaction]]:
    """Generate synthetic transaction data for both algorithms."""
    
    # Generate items
    items = [f"item_{i}" for i in range(num_items)]
    
    # Generate contexts
    contexts = []
    time_of_day = ['morning', 'afternoon', 'evening']
    regions = ['urban', 'suburban', 'rural']
    demographics = ['young', 'adult', 'senior']
    
    for _ in range(num_contexts):
        context = ContextMetadata(
            temporal={"time_of_day": random.choice(time_of_day)},
            spatial={"region": random.choice(regions)},
            user={"demographic": random.choice(demographics)}
        )
        contexts.append(context)
    
    # Generate transactions
    traditional_transactions = []
    contextual_transactions = []
    
    for _ in range(num_transactions):
        # Generate random transaction
        transaction_items = random.sample(
            items,
            random.randint(1, max_items_per_transaction)
        )
        traditional_transactions.append(transaction_items)
        
        # Generate contextual transaction
        context = random.choice(contexts)
        contextual_transactions.append(
            ContextualTransaction(
                items=transaction_items,
                context=context,
                timestamp=datetime.now()
            )
        )
    
    return traditional_transactions, contextual_transactions

@profile
def benchmark_traditional(transactions: List[List[str]], min_support: float, min_confidence: float) -> dict:
    """Benchmark traditional Apriori algorithm."""
    start_time = time.time()
    memory_start = sys.getsizeof(transactions)
    
    # Run algorithm
    apriori = AprioriAlgorithm(min_support=min_support, min_confidence=min_confidence)
    apriori.fit(transactions)
    rules = apriori.get_rules()
    
    # Calculate metrics
    end_time = time.time()
    memory_end = sys.getsizeof(transactions) + sys.getsizeof(rules)
    
    return {
        'execution_time': end_time - start_time,
        'memory_usage': memory_end - memory_start,
        'num_rules': len(rules)
    }

@profile
def benchmark_context_sensitive(
    transactions: List[ContextualTransaction],
    min_support: float,
    min_confidence: float
) -> dict:
    """Benchmark context-sensitive Apriori algorithm."""
    start_time = time.time()
    memory_start = sys.getsizeof(transactions)
    
    # Run algorithm
    cs_apriori = ContextSensitiveApriori(min_support=min_support, min_confidence=min_confidence)
    cs_apriori.fit(transactions)
    rules = cs_apriori.get_rules()
    stats = cs_apriori.get_performance_stats()
    
    # Calculate metrics
    end_time = time.time()
    memory_end = sys.getsizeof(transactions) + sys.getsizeof(rules)
    
    return {
        'execution_time': end_time - start_time,
        'memory_usage': memory_end - memory_start,
        'num_rules': len(rules),
        'detailed_stats': stats
    }

def run_benchmarks():
    """Run comprehensive benchmarks with different dataset sizes."""
    
    # Test configurations
    configurations = [
        (1000, 100, 10, 5),    # Small dataset
        (5000, 200, 15, 10),   # Medium dataset
        (10000, 300, 20, 15),  # Large dataset
    ]
    
    print("Running Benchmarks...")
    print("=" * 50)
    
    for num_trans, num_items, max_items, num_contexts in configurations:
        print(f"\nDataset Configuration:")
        print(f"Transactions: {num_trans}")
        print(f"Unique Items: {num_items}")
        print(f"Max Items/Transaction: {max_items}")
        print(f"Number of Contexts: {num_contexts}")
        print("-" * 30)
        
        # Generate data
        trad_trans, ctx_trans = generate_synthetic_data(
            num_trans, num_items, max_items, num_contexts
        )
        
        # Run traditional benchmark
        print("\nTraditional Apriori:")
        trad_results = benchmark_traditional(trad_trans, 0.01, 0.2)
        print(f"Execution Time: {trad_results['execution_time']:.3f} seconds")
        print(f"Memory Usage: {trad_results['memory_usage']/1024:.2f} KB")
        print(f"Rules Generated: {trad_results['num_rules']}")
        
        # Run context-sensitive benchmark
        print("\nContext-Sensitive Apriori:")
        cs_results = benchmark_context_sensitive(ctx_trans, 0.01, 0.2)
        print(f"Execution Time: {cs_results['execution_time']:.3f} seconds")
        print(f"Memory Usage: {cs_results['memory_usage']/1024:.2f} KB")
        print(f"Rules Generated: {cs_results['num_rules']}")
        print("\nDetailed Performance Metrics:")
        for metric, value in cs_results['detailed_stats'].items():
            print(f"{metric}: {value:.3f}")
        
        # Calculate relative performance
        time_ratio = cs_results['execution_time'] / trad_results['execution_time'] if trad_results['execution_time'] != 0 else float('inf')
        memory_ratio = cs_results['memory_usage'] / trad_results['memory_usage'] if trad_results['memory_usage'] != 0 else float('inf')
        if trad_results['num_rules'] == 0:
            rules_ratio = float('inf')
        else:
            rules_ratio = cs_results['num_rules'] / trad_results['num_rules']
        
        print("\nRelative Performance (Context-Sensitive / Traditional):")
        print(f"Time Ratio: {time_ratio:.2f}x")
        print(f"Memory Ratio: {memory_ratio:.2f}x")
        print(f"Rules Ratio: {rules_ratio:.2f}x")
        print("=" * 50)

if __name__ == "__main__":
    run_benchmarks() 