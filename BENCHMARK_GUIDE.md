# DoChat Benchmark Results

## Executive Summary

**Benchmark Date:** March 1, 2026  
**Index Size:** 22 chunks  
**Test Queries:** 5 unique queries, tested 50 times for latency analysis

---

## ⚡ Latency Performance (50 queries)

| Metric | Average | Median | P95 |
|--------|---------|--------|-----|
| **FAISS Search** | 189.6 ms | 173.0 ms | 238.5 ms |
| **CrossEncoder Reranking** | 410.6 ms | 409.1 ms | 476.8 ms |
| **LLM Generation (Groq)** | 7,825.4 ms | 9,685.0 ms | 10,492.3 ms |
| **Total Query Time** | **8,425.6 ms** | **10,220.5 ms** | **11,048.3 ms** |

### Analysis
- **FAISS similarity search** takes ~190ms on average
- **Reranking** adds ~410ms overhead for quality improvement
- **LLM generation** is the primary bottleneck at ~7.8 seconds
- **End-to-end latency**: **~8.4 seconds per query** (median: 10.2s)

### Interpretation
The high LLM latency is primarily due to:
1. Network latency to Groq API
2. Model inference time for llama-3.1-8b-instant
3. Response token generation (200 token limit)

---

## 🎯 Token Optimization

| Metric | Tokens | Notes |
|--------|--------|-------|
| **Full Document** | 3,435 tokens | Baseline without RAG |
| **RAG Context** | 659 tokens | Top-5 chunks after reranking |
| **Max Response** | 200 tokens | LLM output limit |
| **Baseline Total** | 3,685 tokens | Full doc + response |
| **RAG Total** | 909 tokens | RAG context + response |
| **Reduction** | **75.3%** | Token savings with RAG |

### Analysis
- RAG reduces context from 3,435 to 659 tokens (80.8% reduction)
- Overall token usage reduced by **75.3%** per query
- Cost savings scale linearly with token reduction

---

## 🛡️ Hallucination Analysis (30 samples)

| Category | Percentage | Description |
|----------|-----------|-------------|
| **Grounded Proper Refusal** | 40.0% | Correctly refused when no context |
| **Ungrounded Hallucination** | 0.0% | Made up facts unprompted |
| **Grounded Hallucination (estimate)** | 60.0% | Answered without proper grounding |

### Interpretation
- **40% proper refusal rate**: Model correctly says "I don't know" when context is insufficient
- **0% ungrounded hallucination**: No completely fabricated information
- **60% grounded hallucination**: Model provides answers that may not be fully supported by retrieved context

### Recommendations
1. Improve prompt engineering to increase proper refusal rate
2. Add confidence thresholds for reranking scores
3. Implement citation tracking to verify answer grounding

---

## 📊 Accuracy Metrics

| Metric | Value |
|--------|-------|
| **Labeled Queries** | 0 |
| **Baseline Accuracy** | N/A |
| **Reranked Accuracy** | N/A |
| **Improvement** | N/A |

**Status:** Accuracy testing requires manually labeled ground truth queries. Not yet implemented.

---

## 🔧 How to Run Benchmarks

### Prerequisites
```bash
# Ensure embeddings exist
python modules/RAG.py

# Install dependencies
pip install -r requirements.txt
```

### Run Benchmark
```bash
python benchmark.py
```

### What Gets Measured
1. ✅ **Latency**: FAISS search, reranking, LLM generation (50 iterations)
2. ✅ **Token Usage**: Context size comparison (RAG vs. full document)
3. ⚠️ **Hallucination**: Estimated via prompt-based sampling (30 samples)
4. ❌ **Accuracy**: Requires labeled test dataset (not implemented)

### Output
Results saved to `benchmark_results.json` with:
- Timestamp
- Index metadata (number of chunks)
- Detailed latency statistics (avg/median/p95)
- Token usage breakdown
- Hallucination estimates
- Accuracy metrics (if available)

---

## 📈 Performance Optimization Tips

### Reduce Latency
1. **Use local LLM**: Replace Groq API with local inference (Ollama, vLLM)
2. **Batch queries**: Process multiple queries in parallel
3. **Cache frequent queries**: Add Redis/in-memory caching layer
4. **Optimize reranking**: Reduce candidate pool or use faster CrossEncoder model

### Improve Answer Quality
1. **Increase chunk overlap**: Current 150 chars (16.7%) → try 200-300 chars
2. **Tune reranking threshold**: Filter low-score results before LLM
3. **Add query expansion**: Rephrase user queries for better retrieval
4. **Implement hybrid search**: Combine semantic + keyword (BM25) search

### Reduce Hallucination
1. **Stricter prompts**: Emphasize "only use provided context"
2. **Add citation requirement**: Force model to quote source text
3. **Confidence scoring**: Return uncertainty estimates with answers
4. **Multi-stage validation**: Add answer verification step

---

## 🎓 Benchmark Limitations

1. **Small test set**: Only 5 unique queries used
2. **No accuracy labels**: Can't measure retrieval/answer quality objectively
3. **Network dependency**: Groq API latency varies by region/time
4. **Hallucination estimation**: Requires manual review for precision
5. **Single document type**: Results may not generalize to all PDF types

---

## 📝 Next Steps

- [ ] Create labeled test dataset (50+ queries with ground truth)
- [ ] Add answer correctness scoring (RAGAS, BERTScore)
- [ ] Benchmark against different embedding models
- [ ] Test local LLM performance vs. Groq API
- [ ] Implement A/B testing framework for prompt variations

---

## 🔗 Related Files

- [benchmark.py](benchmark.py) - Benchmark script
- [test_queries.json](test_queries.json) - Test query definitions
- [benchmark_results.json](benchmark_results.json) - Raw JSON output
- [README.md](README.md) - Project overview
- [modules/RAG.py](modules/RAG.py) - RAG implementation

---

**Note**: These metrics are based on actual benchmark runs with your current setup. Performance will vary based on:
- Hardware (CPU/GPU)
- Network latency to Groq API
- Document size and complexity
- Number of indexed chunks
