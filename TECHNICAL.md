# Technical Deep Dive

## Architecture Overview

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│  Text Preprocessing              │
│  - Lowercase                     │
│  - Strip accents (bỏ dấu)        │
│  - Tokenization                  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  BM25 Retrieval                  │
│  - Compute query TF              │
│  - Calculate BM25 scores         │
│  - Structural boosting (optional)│
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  MMR Selection                   │
│  - Select top-K candidates       │
│  - Maximize relevance+diversity  │
│  - Extract step outline          │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Context Building                │
│  - Token budget management       │
│  - Chunk concatenation           │
│  - Add extracted outline         │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  LLM Generation                  │
│  - Qwen3 inference               │
│  - Temperature sampling          │
│  - Stop token detection          │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Post-processing                 │
│  - Filter <think> tags           │
│  - Language validation           │
│  - Auto-translation if needed    │
└──────┬───────────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

---

## 1. BM25 Algorithm Details

### Formula

```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
             qi∈Q
```

### Components

**Inverse Document Frequency (IDF)**:
```python
IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + ε)
```
- `N`: Total number of documents
- `df(qi)`: Document frequency of term qi
- `ε = 1e-12`: Smoothing factor

**Term Frequency (TF)**:
```python
TF_score = (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × normalization)
```
- `f(qi, D)`: Frequency of qi in document D
- `k1 = 1.5`: Term frequency saturation parameter
- `normalization = 1 - b + b × |D|/avgdl`

**Document Length Normalization**:
```python
norm = 1 - b + b × (|D| / avgdl)
```
- `b = 0.75`: Length normalization parameter
- `|D|`: Length of document D
- `avgdl`: Average document length in collection

### Why BM25 for Vietnamese?

1. **Language-agnostic**: Works with any tokenization scheme
2. **No embedding required**: Saves memory and computation
3. **Interpretable**: Can debug why certain docs are retrieved
4. **Fast**: O(n) retrieval with proper indexing
5. **Effective for keyword matching**: Vietnamese technical docs

---

## 2. MMR (Maximal Marginal Relevance)

### Motivation

BM25 alone có thể retrieve nhiều chunks tương tự nhau (low diversity).
MMR giải quyết bằng cách balance giữa:
- **Relevance**: Similarity với query
- **Diversity**: Khác biệt với chunks đã chọn

### Algorithm

```python
def mmr_select(chunks, scores, k, λ=0.7):
    selected = []
    candidates = list(range(len(chunks)))

    while len(selected) < k and candidates:
        best_idx = arg_max(
            λ × relevance(i) - (1-λ) × max_similarity(i, selected)
            for i in candidates
        )
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected
```

### Parameters

- **λ = 0.65**: Trade-off parameter
  - λ=1: Pure relevance (giống BM25)
  - λ=0: Pure diversity (có thể mất relevant docs)
  - λ=0.65: Balance tốt cho Vietnamese docs

### Similarity Metric

Sử dụng **Jaccard Similarity**:

```python
J(A, B) = |A ∩ B| / |A ∪ B|
```

Pros:
- Simple và fast
- Works well với tokenized sets
- No need for embeddings

Cons:
- Không capture semantic similarity
- Sensitive to vocabulary mismatch

---

## 3. Structural Boosting

### Motivation

Với queries về "quy trình", "các bước", chunks có cấu trúc liệt kê (Bước 1, 2, 3...) thường relevant hơn.

### Implementation

```python
def structural_boost_score(text: str) -> float:
    score = 0.0

    # Pattern 1: "Bước 1:", "Bước 2:"
    if re.search(r"^Bước\s+\d+", text, flags=re.MULTILINE):
        score += 0.3

    # Pattern 2: Keywords like "quy trình", "trình tự"
    if any(kw in text.lower() for kw in ["quy trình", "trình tự", "bao gồm"]):
        score += 0.15

    return score
```

### When to Apply

```python
boost_query = any(kw in query.lower() for kw in ["các bước", "quy trình", "trình tự"])

if boost_query:
    scores = [bm25_score(i) + structural_boost(chunks[i]) for i in range(n)]
```

---

## 4. Vietnamese Text Processing

### Challenges

1. **Diacritics (dấu)**: à, á, ả, ã, ạ, â, ă...
2. **Multi-word tokens**: "máy phay" vs "máy" + "phay"
3. **Spelling variations**: "Bước" vs "buoc"

### Solutions

#### Accent Removal (NFKD Normalization)

```python
def strip_accents(s: str) -> str:
    # Normalize to NFKD: "á" → "a" + "combining acute accent"
    s_nfkd = unicodedata.normalize("NFKD", s)

    # Filter out combining characters
    return "".join(c for c in s_nfkd if not unicodedata.combining(c))
```

Example:
- Input: `"Các bước vận hành"`
- After normalize: `"cac buoc van hanh"`

#### Tokenization

```python
def tokenize_vi(s: str) -> List[str]:
    s = s.lower()
    s = strip_accents(s)
    return re.findall(r"[a-z0-9]+", s)
```

Example:
- Input: `"Bước 1: Làm sạch"`
- Output: `["buoc", "1", "lam", "sach"]`

**Limitations**:
- Không handle compound words ("máy_phay")
- Future: Integrate VnCoreNLP hoặc pyvi

---

## 5. Token Budget Management

### Problem

Context window limited: `N_CTX = 2048 tokens`

Budget allocation:
```
Total = Prompt + Context + Generation
2048  = 150    + (2048-150-320) + 320
      = 150    + 1578            + 320
```

### Strategy

```python
def limit_context_by_tokens(chunks, budget=1578):
    ctx = "\n\n---\n\n".join(chunks)
    used = count_tokens(ctx)

    # Iteratively remove least relevant chunk (last one)
    while used > budget and len(chunks) > 1:
        chunks.pop()
        ctx = "\n\n---\n\n".join(chunks)
        used = count_tokens(ctx)

    return chunks
```

### Token Counting

Sử dụng **real tokenizer** của model:

```python
def count_tokens(text: str) -> int:
    # Chính xác hơn heuristics như len(text) / 4
    ids = MODEL.tokenize(text.encode("utf-8"))
    return len(ids)
```

**Tại sao không dùng approximation?**
- Vietnamese có ratio khác English (thường ~0.7 words/token)
- Heuristics sai số ±20%
- Real tokenizer: 100% accurate

---

## 6. Prompt Engineering

### Structure

```
<|im_start|>system
{system_instructions}
<|im_end|>

<|im_start|>user
NGỮ CẢNH:
{retrieved_context}

SƯỜN TRÍCH XUẤT:
{outline}

YÊU CẦU:
- Câu hỏi: {question}
- Chỉ dùng thông tin trong ngữ cảnh
- Không thêm kiến thức ngoài
- Trả lời bằng tiếng Việt
<|im_end|>

<|im_start|>assistant
```

### Key Techniques

1. **Explicit constraints**:
   - "CHỈ dùng thông tin trong NGỮ CẢNH"
   - "Không nêu suy nghĩ trung gian"
   - "Luôn trả lời bằng tiếng Việt"

2. **Structured input**:
   - Clear sections: NGỮ CẢNH, SƯỜN, YÊU CẦU
   - Separator: `---` giữa chunks

3. **Task specification**:
   - "Nếu hỏi 'các bước', liệt kê đầy đủ 1), 2), 3)..."
   - "Ngắn gọn, đúng trọng tâm"

### Stop Tokens

```python
stop=["<|im_end|>"]
```

Prevents model from generating system/user turns.

---

## 7. Post-processing

### Filter <think> Tags

Qwen models sometimes output reasoning in `<think>` tags:

```
<think>Let me analyze...</think>
The answer is...
```

Solution:

```python
def filter_think_tags(text: str) -> str:
    # Remove complete <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Handle unclosed tags: keep only text before <think>
    if '<think>' in text.lower():
        text = re.split(r'<think>', text, flags=re.IGNORECASE)[0]

    # Remove orphan closing tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)

    return text.strip()
```

### Language Validation

```python
def ensure_vietnamese(text: str) -> str:
    # Count Vietnamese diacritics
    vi_chars = len(re.findall(r'[àáạảã...]', text.lower()))

    # Count English words
    en_words = len(re.findall(r'\b[a-z]{3,}\b', text.lower()))

    # If too much English, auto-translate
    if en_words > 10 and vi_chars < 5:
        return auto_translate_to_vietnamese(text)

    return text
```

---

## 8. Performance Optimization

### Memory Optimization

1. **GGUF quantization**: 1.5GB instead of 6GB (FP16)
2. **mmap**: Load model without duplicating in RAM
3. **No mlock**: Allow OS to swap (good for Pi)
4. **Index caching**: Pre-compute BM25 vocab/IDF

### Compute Optimization

1. **Batch operations**: NumPy vectorization
2. **Early stopping**: MMR chỉ xét top-50 candidates
3. **Lazy loading**: Chỉ load model khi cần
4. **Token budget**: Giới hạn context size

### Latency Breakdown (Pi 4)

| Stage | Time | % |
|-------|------|---|
| Load index | 0.2s | 2% |
| BM25 retrieval | 0.1s | 1% |
| MMR selection | 0.1s | 1% |
| Token counting | 0.3s | 3% |
| **LLM generation** | **8-10s** | **90%** |
| Post-processing | 0.1s | 1% |

**Bottleneck**: LLM inference (unavoidable với CPU)

---

## 9. Future Improvements

### Vector Database Integration

**Current**: BM25 (lexical matching)
**Future**: Hybrid search (BM25 + semantic embeddings)

```python
# Pseudo-code
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Index với embeddings
vectors = embed_model.encode(chunks)
client.upsert(collection_name="docs", vectors=vectors)

# Hybrid retrieval
dense_results = client.search(query_vector, top_k=20)
sparse_results = bm25_search(query, top_k=20)
final = rerank(dense_results, sparse_results)
```

**Benefits**:
- Semantic matching ("máy phay" ~ "thiết bị cắt gọt")
- Cross-lingual retrieval
- Better recall for paraphrased queries

### LangChain Integration

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

result = qa({"query": "Các bước vận hành máy phay?"})
```

**Benefits**:
- Conversation memory
- Agent workflows với tools
- Production-ready abstractions

---

## References

1. [BM25 Original Paper](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
2. [MMR for Diversity](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
3. [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
4. [llama.cpp Quantization](https://github.com/ggerganov/llama.cpp)

---

**Questions?** Open an issue hoặc contact: your.email@example.com
