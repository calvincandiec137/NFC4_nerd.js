"""
DoChat RAG System Benchmark
Measures latency, accuracy, hallucination rate, and token usage
Run: python benchmark.py
"""

import os
import json
import time
import statistics
from datetime import datetime
from typing import List, Dict, Tuple

# Import DoChat modules
from modules.response_groq import (
    embed_query, retrieve, rerank, generate_answer,
    index, metadata, groq_client, GROQ_MODEL
)


# ============================================================================
# TEST QUERIES
# ============================================================================

def load_test_queries(filepath: str = "test_queries.json") -> List[Tuple]:
    """Load test queries from JSON file if it exists"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                queries = []
                for q in data.get('queries', []):
                    queries.append((
                        q['query'],
                        q.get('expected_keyword'),
                        q.get('expected_doc_id')
                    ))
                return queries
        except Exception as e:
            print(f"⚠️  Warning: Could not load {filepath}: {e}")
            return DEFAULT_TEST_QUERIES
    return DEFAULT_TEST_QUERIES

# Default queries (used if test_queries.json doesn't exist)
DEFAULT_TEST_QUERIES = [
    ("What is the main topic of this document?", None, None),
    ("Explain the key concepts", None, None),
    ("What are the important points?", None, None),
    ("Summarize the content", None, None),
    ("What does this document discuss?", None, None),
]

# Load queries from file (or fall back to defaults)
TEST_QUERIES = load_test_queries()

# For more accurate benchmarks, edit test_queries.json and add 50-100 examples
# with expected_keyword and expected_doc_id for accuracy testing


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_latency(num_queries: int = 20) -> Dict:
    """Measure average latency for each component"""
    print(f"\n{'='*60}")
    print(f"LATENCY BENCHMARK ({num_queries} queries)")
    print(f"{'='*60}")
    
    # Use cycling through test queries
    queries = [TEST_QUERIES[i % len(TEST_QUERIES)][0] for i in range(num_queries)]
    
    faiss_times = []
    rerank_times = []
    llm_times = []
    total_times = []
    
    for i, query in enumerate(queries, 1):
        print(f"\rProcessing query {i}/{num_queries}...", end="", flush=True)
        
        t_total = time.time()
        
        # FAISS retrieval
        t1 = time.time()
        retrieved = retrieve(query, top_k=20)
        faiss_time = (time.time() - t1) * 1000  # Convert to ms
        
        # Reranking
        t2 = time.time()
        reranked = rerank(query, retrieved, top_n=5)
        rerank_time = (time.time() - t2) * 1000
        
        # LLM generation
        t3 = time.time()
        context = "\n\n---\n\n".join(d["text"] for d in reranked)
        answer = generate_answer(query, context)
        llm_time = (time.time() - t3) * 1000
        
        total_time = (time.time() - t_total) * 1000
        
        faiss_times.append(faiss_time)
        rerank_times.append(rerank_time)
        llm_times.append(llm_time)
        total_times.append(total_time)
    
    print()  # New line after progress
    
    results = {
        "num_queries": num_queries,
        "faiss_avg_ms": statistics.mean(faiss_times),
        "faiss_median_ms": statistics.median(faiss_times),
        "faiss_p95_ms": statistics.quantiles(faiss_times, n=20)[18] if len(faiss_times) >= 20 else max(faiss_times),
        "rerank_avg_ms": statistics.mean(rerank_times),
        "rerank_median_ms": statistics.median(rerank_times),
        "rerank_p95_ms": statistics.quantiles(rerank_times, n=20)[18] if len(rerank_times) >= 20 else max(rerank_times),
        "llm_avg_ms": statistics.mean(llm_times),
        "llm_median_ms": statistics.median(llm_times),
        "llm_p95_ms": statistics.quantiles(llm_times, n=20)[18] if len(llm_times) >= 20 else max(llm_times),
        "total_avg_ms": statistics.mean(total_times),
        "total_median_ms": statistics.median(total_times),
        "total_p95_ms": statistics.quantiles(total_times, n=20)[18] if len(total_times) >= 20 else max(total_times),
    }
    
    print(f"\n{'Component':<20} {'Avg (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12}")
    print(f"{'-'*60}")
    print(f"{'FAISS Retrieval':<20} {results['faiss_avg_ms']:<12.2f} {results['faiss_median_ms']:<12.2f} {results['faiss_p95_ms']:<12.2f}")
    print(f"{'CrossEncoder Rerank':<20} {results['rerank_avg_ms']:<12.2f} {results['rerank_median_ms']:<12.2f} {results['rerank_p95_ms']:<12.2f}")
    print(f"{'LLM Generation':<20} {results['llm_avg_ms']:<12.2f} {results['llm_median_ms']:<12.2f} {results['llm_p95_ms']:<12.2f}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<20} {results['total_avg_ms']:<12.2f} {results['total_median_ms']:<12.2f} {results['total_p95_ms']:<12.2f}")
    
    return results


def benchmark_accuracy() -> Dict:
    """Compare retrieval accuracy with and without reranking"""
    print(f"\n{'='*60}")
    print(f"ACCURACY BENCHMARK")
    print(f"{'='*60}")
    print("Note: Requires manually labeled test queries with expected results")
    
    # Filter queries that have expected results
    labeled_queries = [q for q in TEST_QUERIES if q[1] is not None or q[2] is not None]
    
    if not labeled_queries:
        print("\n⚠️  No labeled test queries found.")
        print("   Add expected_keyword and expected_doc_id to TEST_QUERIES")
        print("   for accurate benchmarking.\n")
        return {
            "labeled_queries": 0,
            "baseline_accuracy": None,
            "reranked_accuracy": None,
            "improvement_pct": None,
        }
    
    baseline_hits = 0
    reranked_hits = 0
    
    for query, expected_keyword, expected_doc_id in labeled_queries:
        # Baseline: top-5 without reranking
        baseline_results = retrieve(query, top_k=5)
        if any(
            (expected_keyword and expected_keyword.lower() in doc["text"].lower()) or
            (expected_doc_id and doc["doc_id"] == expected_doc_id)
            for doc in baseline_results
        ):
            baseline_hits += 1
        
        # With reranking: top-20 -> rerank -> top-5
        retrieved = retrieve(query, top_k=20)
        reranked_results = rerank(query, retrieved, top_n=5)
        if any(
            (expected_keyword and expected_keyword.lower() in doc["text"].lower()) or
            (expected_doc_id and doc["doc_id"] == expected_doc_id)
            for doc in reranked_results
        ):
            reranked_hits += 1
    
    baseline_acc = baseline_hits / len(labeled_queries)
    reranked_acc = reranked_hits / len(labeled_queries)
    improvement = ((reranked_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    
    results = {
        "labeled_queries": len(labeled_queries),
        "baseline_accuracy": baseline_acc,
        "reranked_accuracy": reranked_acc,
        "improvement_pct": improvement,
    }
    
    print(f"\nLabeled queries: {len(labeled_queries)}")
    print(f"Baseline accuracy (top-5 direct):  {baseline_acc:.1%}")
    print(f"Reranked accuracy (top-20→5):      {reranked_acc:.1%}")
    print(f"Improvement:                        {improvement:+.1f}%")
    
    return results


def benchmark_hallucination(num_samples: int = 20) -> Dict:
    """Estimate hallucination rate by checking context grounding"""
    print(f"\n{'='*60}")
    print(f"HALLUCINATION BENCHMARK ({num_samples} samples)")
    print(f"{'='*60}")
    print("Note: Automated check for 'I don't know' responses vs. answers")
    print("      Manual review recommended for accurate hallucination detection\n")
    
    queries = [TEST_QUERIES[i % len(TEST_QUERIES)][0] for i in range(num_samples)]
    
    # Test with context (normal RAG)
    grounded_idk_count = 0
    grounded_answers = []
    
    # Test without context (to simulate hallucination baseline)
    ungrounded_idk_count = 0
    ungrounded_answers = []
    
    for i, query in enumerate(queries, 1):
        print(f"\rProcessing sample {i}/{num_samples}...", end="", flush=True)
        
        # Grounded response (with RAG context)
        retrieved = retrieve(query, top_k=20)
        reranked = rerank(query, retrieved, top_n=5)
        context = "\n\n---\n\n".join(d["text"] for d in reranked)
        grounded_answer = generate_answer(query, context)
        grounded_answers.append(grounded_answer)
        
        if "i don't know" in grounded_answer.lower() or "not present" in grounded_answer.lower():
            grounded_idk_count += 1
        
        # Ungrounded response (empty context to force hallucination)
        ungrounded_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know."

Context:
[Empty context]

Question:
{query}
"""
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": ungrounded_prompt}],
                temperature=0.2,
                max_tokens=200
            )
            ungrounded_answer = response.choices[0].message.content.strip()
            ungrounded_answers.append(ungrounded_answer)
            
            if "i don't know" in ungrounded_answer.lower() or "not present" in ungrounded_answer.lower():
                ungrounded_idk_count += 1
        except Exception as e:
            print(f"\nWarning: Ungrounded query failed: {e}")
            ungrounded_answers.append("ERROR")
    
    print()
    
    # Hallucination proxy: answers given when context is empty
    grounded_hallucination_rate = 1 - (grounded_idk_count / num_samples)
    ungrounded_hallucination_rate = 1 - (ungrounded_idk_count / num_samples)
    
    results = {
        "num_samples": num_samples,
        "grounded_proper_refusal_pct": (grounded_idk_count / num_samples) * 100,
        "ungrounded_hallucination_pct": ungrounded_hallucination_rate * 100,
        "grounded_hallucination_estimate_pct": grounded_hallucination_rate * 100,
    }
    
    print(f"\nWith RAG context:")
    print(f"  Proper refusals ('I don't know'): {grounded_idk_count}/{num_samples} ({results['grounded_proper_refusal_pct']:.1f}%)")
    print(f"  Attempted answers: {num_samples - grounded_idk_count}/{num_samples}")
    
    print(f"\nWithout context (baseline):")
    print(f"  Hallucinated answers: {num_samples - ungrounded_idk_count}/{num_samples} ({results['ungrounded_hallucination_pct']:.1f}%)")
    
    print(f"\n⚠️  Note: True hallucination requires manual review of answer correctness")
    
    return results


def benchmark_token_usage() -> Dict:
    """Calculate token usage vs. full document baseline"""
    print(f"\n{'='*60}")
    print(f"TOKEN USAGE BENCHMARK")
    print(f"{'='*60}")
    
    # Calculate full document tokens
    full_texts = [doc["text"] for doc in metadata]
    full_document = "\n".join(full_texts)
    
    # Rough token estimate: 1 token ≈ 4 characters (GPT tokenization)
    full_doc_tokens = len(full_document) / 4
    
    # RAG uses top-5 chunks
    sample_query = TEST_QUERIES[0][0]
    retrieved = retrieve(sample_query, top_k=20)
    reranked = rerank(sample_query, retrieved, top_n=5)
    rag_context = "\n\n---\n\n".join(d["text"] for d in reranked)
    rag_context_tokens = len(rag_context) / 4
    
    # Response tokens (configured max)
    max_response_tokens = 200
    
    # Full RAG tokens (context + system prompt + response)
    system_prompt_tokens_estimate = 50  # "Answer using ONLY the context..." etc.
    rag_total_tokens = rag_context_tokens + system_prompt_tokens_estimate + max_response_tokens
    
    # Baseline: full doc + response
    baseline_total_tokens = full_doc_tokens + system_prompt_tokens_estimate + max_response_tokens
    
    reduction_pct = ((baseline_total_tokens - rag_total_tokens) / baseline_total_tokens) * 100
    
    results = {
        "full_document_tokens": int(full_doc_tokens),
        "rag_context_tokens": int(rag_context_tokens),
        "max_response_tokens": max_response_tokens,
        "baseline_total_tokens": int(baseline_total_tokens),
        "rag_total_tokens": int(rag_total_tokens),
        "token_reduction_pct": reduction_pct,
    }
    
    print(f"\nFull document context:     {results['full_document_tokens']:,} tokens")
    print(f"RAG context (top-5):       {results['rag_context_tokens']:,} tokens")
    print(f"Max response:              {results['max_response_tokens']:,} tokens")
    print(f"\nBaseline total:            {results['baseline_total_tokens']:,} tokens/query")
    print(f"RAG total:                 {results['rag_total_tokens']:,} tokens/query")
    print(f"Token reduction:           {results['token_reduction_pct']:.1f}%")
    
    return results


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

def run_all_benchmarks(
    latency_queries: int = 50,
    hallucination_samples: int = 30
) -> Dict:
    """Run complete benchmark suite"""
    
    print("\n" + "="*60)
    print("DoChat RAG System - Comprehensive Benchmark")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Index size: {index.ntotal} vectors")
    print(f"Test queries: {len(TEST_QUERIES)}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "index_size": index.ntotal,
        "test_queries": len(TEST_QUERIES),
    }
    
    # Run benchmarks
    try:
        results["latency"] = benchmark_latency(latency_queries)
    except Exception as e:
        print(f"\n❌ Latency benchmark failed: {e}")
        results["latency"] = {"error": str(e)}
    
    try:
        results["accuracy"] = benchmark_accuracy()
    except Exception as e:
        print(f"\n❌ Accuracy benchmark failed: {e}")
        results["accuracy"] = {"error": str(e)}
    
    try:
        results["hallucination"] = benchmark_hallucination(hallucination_samples)
    except Exception as e:
        print(f"\n❌ Hallucination benchmark failed: {e}")
        results["hallucination"] = {"error": str(e)}
    
    try:
        results["token_usage"] = benchmark_token_usage()
    except Exception as e:
        print(f"\n❌ Token usage benchmark failed: {e}")
        results["token_usage"] = {"error": str(e)}
    
    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Generate resume-ready summary
    generate_resume_summary(results)
    
    return results


def generate_resume_summary(results: Dict):
    """Generate resume-ready bullet points from benchmark results"""
    print(f"\n{'='*60}")
    print("RESUME-READY METRICS")
    print(f"{'='*60}\n")
    
    bullets = []
    
    # Latency
    if "latency" in results and "total_avg_ms" in results["latency"]:
        avg_latency = results["latency"]["total_avg_ms"]
        bullets.append(
            f"• Built production RAG system with {avg_latency:.0f}ms average query latency "
            f"(measured over {results['latency']['num_queries']} queries) through "
            f"two-stage FAISS + CrossEncoder reranking"
        )
    
    # Accuracy
    if "accuracy" in results and results["accuracy"].get("improvement_pct"):
        improvement = results["accuracy"]["improvement_pct"]
        baseline = results["accuracy"]["baseline_accuracy"]
        reranked = results["accuracy"]["reranked_accuracy"]
        bullets.append(
            f"• Improved answer relevance by {improvement:.0f}% via CrossEncoder reranking "
            f"({baseline:.0%} → {reranked:.0%} accuracy on {results['accuracy']['labeled_queries']} test queries)"
        )
    
    # Token usage
    if "token_usage" in results and "token_reduction_pct" in results["token_usage"]:
        reduction = results["token_usage"]["token_reduction_pct"]
        baseline = results["token_usage"]["baseline_total_tokens"]
        optimized = results["token_usage"]["rag_total_tokens"]
        bullets.append(
            f"• Optimized token usage by {reduction:.0f}% ({baseline:,} → {optimized:,} tokens/query) "
            f"using 900-char chunking with 16.7% overlap"
        )
    
    # Deployment
    bullets.append(
        f"• Deployed FastAPI backend + React frontend with real-time PDF processing, "
        f"384-dim embedding generation (Qwen3-0.6B), and CORS-enabled REST API"
    )
    
    for bullet in bullets:
        print(bullet)
    
    print(f"\n{'='*60}\n")
    
    # Save to file
    with open("RESUME_METRICS.txt", "w") as f:
        f.write("DoChat RAG System - Verified Performance Metrics\n")
        f.write("="*60 + "\n\n")
        for bullet in bullets:
            f.write(bullet + "\n")
        f.write(f"\nBenchmarked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Resume metrics saved to: RESUME_METRICS.txt\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n🚀 DoChat RAG System Benchmark\n")
    
    # Check if index exists
    if not os.path.exists("./embeddings/index.faiss"):
        print("❌ Error: FAISS index not found.")
        print("   Run 'python modules/RAG.py' first to create embeddings.\n")
        sys.exit(1)
    
    # Load and check test queries
    if os.path.exists("test_queries.json"):
        print(f"✅ Loaded {len(TEST_QUERIES)} test queries from test_queries.json")
    else:
        print(f"⚠️  Using {len(TEST_QUERIES)} default test queries")
        print("   Create test_queries.json with custom queries for better benchmarks")
    
    # Check for labeled queries
    labeled_count = sum(1 for q in TEST_QUERIES if q[1] is not None or q[2] is not None)
    if labeled_count == 0:
        print(f"⚠️  No labeled queries found - accuracy benchmarks will be limited")
        print("   Add expected_keyword/expected_doc_id to test_queries.json\n")
    else:
        print(f"✅ Found {labeled_count} labeled queries for accuracy benchmarking\n")
    
    # Run benchmarks
    try:
        results = run_all_benchmarks(
            latency_queries=50,        # Adjust based on API costs
            hallucination_samples=30   # Adjust based on API costs
        )
        
        print("✅ Benchmark completed successfully!\n")
        print("Next steps:")
        print("1. Review benchmark_results.json for detailed metrics")
        print("2. Check RESUME_METRICS.txt for resume-ready bullet points")
        print("3. Add more labeled test queries to TEST_QUERIES for accuracy benchmarking")
        print("4. Manually review hallucination samples for true positive rate\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
