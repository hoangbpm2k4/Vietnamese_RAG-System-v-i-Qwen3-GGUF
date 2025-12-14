# rag_ultimate.py - ULTIMATE RAG cho Raspberry Pi 4
# Kết hợp TỐT NHẤT từ rag.py + rag_better.py
#
# Tính năng:
# ✅ BM25 + MMR + Boost cấu trúc
# ✅ Strip accents (bỏ dấu) để tăng match tiếng Việt
# ✅ Auto-translate nếu model trả lời tiếng Anh
# ✅ Token counting CHÍNH XÁC (dùng tokenizer thật)
# ✅ Filter <think> tags sạch sẽ
# ✅ Environment variables config
# ✅ Error handling đầy đủ
#
# Chạy: python rag_ultimate.py "Câu hỏi của bạn"

import os, re, json, math, sys, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from llama_cpp import Llama

# ================= CONFIG với Environment Variables =================
GEN_MODEL_PATH = os.getenv("GEN_MODEL_PATH", "qwen3_06b.gguf")
DOC_PATH       = os.getenv("DOC_PATH", "quytrinh.txt")

INDEX_DIR  = Path("./index_rag_ultimate")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_JSON = INDEX_DIR / "bm25_index.json"
CHUNKS_JSON = INDEX_DIR / "chunks.json"

# Pi 4 optimized defaults
N_CTX        = int(os.getenv("N_CTX", "2048"))
MAX_TOK_OUT  = int(os.getenv("MAX_TOK_OUT", "320"))  # Tăng từ 256 → 320
N_THREADS    = int(os.getenv("N_THREADS", "4"))
N_BATCH      = int(os.getenv("N_BATCH", "128"))

# Retrieval params
K_TOP        = int(os.getenv("K_TOP", "3"))
K_CAND       = int(os.getenv("K_CAND", "20"))  # Tăng từ 10 → 20 để extract outline tốt hơn
LAMBDA_MMR   = float(os.getenv("LAMBDA_MMR", "0.65"))  # Giảm từ 0.7 → 0.65 để tăng diversity
STRUCT_BOOST = float(os.getenv("STRUCT_BOOST", "0.3"))
ENFORCE_VI   = bool(int(os.getenv("ENFORCE_VI", "1")))

# Chunking
MAX_CHARS    = int(os.getenv("CHUNK_CHARS", "700"))
OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "100"))

# Generation params
_DEFAULT_TEMPERATURE = 0.2
_temp_str = os.getenv("TEMPERATURE")
if _temp_str is None:
    _temp_str = os.getenv("TEMP")
try:
    TEMPERATURE = float(_temp_str) if _temp_str is not None else _DEFAULT_TEMPERATURE
except (ValueError, TypeError):
    TEMPERATURE = _DEFAULT_TEMPERATURE

TOP_K        = int(os.getenv("TOP_K", "50"))
TOP_P        = float(os.getenv("TOP_P", "0.95"))
REPEAT_PEN   = float(os.getenv("REPEAT_P", "1.1"))
SEED         = int(os.getenv("SEED", "42"))

# Global model
MODEL: Optional[Llama] = None

# ================= Text Processing Utilities =================

def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt để tăng BM25 match"""
    s_nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s_nfkd if not unicodedata.combining(c))

def tokenize_vi(s: str) -> List[str]:
    """Tokenize tiếng Việt: lowercase + bỏ dấu + split"""
    s = s.lower()
    s = strip_accents(s)
    return re.findall(r"[a-z0-9]+", s)

def split_into_chunks(text: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
    """Chia văn bản thành chunks, ưu tiên tách theo Bước"""
    blocks = re.split(
        r"(?=^[-–]*\s*B[uư]ớc\s*\d+[:：]?)|(?=^B[uư]ớc\s+\d+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    pieces: List[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(b) <= max_chars:
            pieces.append(b)
        else:
            if overlap >= max_chars:
                overlap = max_chars // 2
            start = 0
            while start < len(b):
                end = min(len(b), start + max_chars)
                pieces.append(b[start:end])
                if end >= len(b):
                    break
                start = end - overlap
                if start <= 0:
                    start = end
    return [p for p in pieces if p.strip()]

def has_structured_content(text: str) -> bool:
    """Kiểm tra chunk có cấu trúc liệt kê không"""
    patterns = [
        r'B[uư]ớc\s+\d+',
        r'\d+\)',
        r'\d+\.',
        r'[-–]\s*B[uư]ớc',
        r'quy trình',
        r'bao gồm',
        r'trình tự',
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def structural_boost_score(text: str) -> float:
    """Tính điểm boost cho chunks có cấu trúc"""
    score = 0.0
    t = text.lower()
    if re.search(r"^b[uư]ớc\s+\d+", t, flags=re.MULTILINE):
        score += STRUCT_BOOST
    if any(x in t for x in ["quy trình", "5 bước", "9 bước", "bao gồm", "trình tự"]):
        score += STRUCT_BOOST * 0.5
    return score

# ================= BM25 Implementation =================

@dataclass
class BM25Index:
    vocab: Dict[str, int]
    df: List[int]
    idf: List[float]
    avgdl: float
    doc_lengths: List[int]

def build_bm25(chunks: List[str]) -> Tuple[BM25Index, List[List[str]]]:
    """Xây dựng BM25 index"""
    tokenized: List[List[str]] = [tokenize_vi(c) for c in chunks]
    N = len(tokenized)

    # Document frequency
    vocab: Dict[str, int] = {}
    df_counts: Dict[int, int] = {}

    for doc in tokenized:
        seen = set()
        for w in doc:
            if w not in vocab:
                vocab[w] = len(vocab)
            wid = vocab[w]
            if wid not in seen:
                df_counts[wid] = df_counts.get(wid, 0) + 1
                seen.add(wid)

    df = [0] * len(vocab)
    for wid, cnt in df_counts.items():
        df[wid] = cnt

    # IDF (BM25 formula)
    idf = [math.log((N - df_i + 0.5) / (df_i + 0.5) + 1e-12) for df_i in df]
    doc_lengths = [len(doc) for doc in tokenized]
    avgdl = sum(doc_lengths) / max(1, N)

    return BM25Index(
        vocab=vocab,
        df=df,
        idf=idf,
        avgdl=avgdl,
        doc_lengths=doc_lengths
    ), tokenized

def bm25_scores(
    query: List[str],
    bm25: BM25Index,
    tokenized_docs: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75
) -> List[float]:
    """Tính BM25 scores"""
    # Query term frequency
    q_tf: Dict[int, int] = {}
    for w in query:
        if w in bm25.vocab:
            wid = bm25.vocab[w]
            q_tf[wid] = q_tf.get(wid, 0) + 1

    scores = [0.0] * len(tokenized_docs)
    for doc_id, doc in enumerate(tokenized_docs):
        dl = bm25.doc_lengths[doc_id]
        denom = 1 - b + b * dl / max(1e-9, bm25.avgdl)

        # Term frequency
        tf_count: Dict[int, int] = {}
        for w in doc:
            wid = bm25.vocab.get(w)
            if wid is not None:
                tf_count[wid] = tf_count.get(wid, 0) + 1

        s = 0.0
        for wid, qcnt in q_tf.items():
            f = tf_count.get(wid, 0)
            if f == 0:
                continue
            idf_val = bm25.idf[wid]
            s += idf_val * (f * (k1 + 1)) / (f + k1 * denom)
        scores[doc_id] = s

    return scores

def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Jaccard similarity cho MMR"""
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def mmr_select(
    chunks: List[str],
    tokenized: List[List[str]],
    scores: List[float],
    k: int,
    lambda_: float = 0.7
) -> List[int]:
    """MMR selection để tăng diversity"""
    candidates = list(range(len(chunks)))
    selected: List[int] = []

    # Sort by score để speed up
    candidates.sort(key=lambda i: scores[i], reverse=True)

    while candidates and len(selected) < k:
        best_idx, best_val = None, -1e9

        # Chỉ xét top-50 candidates để nhanh hơn
        for i in candidates[:50]:
            rel = scores[i]
            div = 0.0 if not selected else max(
                jaccard_similarity(tokenized[i], tokenized[j])
                for j in selected
            )
            val = lambda_ * rel - (1 - lambda_) * div

            if val > best_val:
                best_val, best_idx = val, i

        if best_idx is not None:
            selected.append(best_idx)
            candidates.remove(best_idx)
        else:
            break

    return selected

# ================= Model & Prompting =================

def init_model():
    """Load model 1 lần duy nhất"""
    global MODEL
    if MODEL is None:
        print("Dang load model...", flush=True)
        try:
            MODEL = Llama(
                model_path=GEN_MODEL_PATH,
                n_ctx=N_CTX,
                embedding=False,
                n_threads=N_THREADS,
                n_batch=N_BATCH,
                use_mmap=True,
                use_mlock=False,
                logits_all=False,
                verbose=False
            )
            print("OK - Model da san sang!\n")
        except Exception as e:
            print(f"LOI khi load model: {e}")
            sys.exit(1)

def build_chat_prompt(context: str, question: str, outline: Optional[str] = None) -> str:
    """
    Tạo prompt Qwen3 với ràng buộc tiếng Việt + SƯỜN TRÍCH XUẤT.
    outline: Sườn các bước trích từ candidates (tính năng từ better.py)
    """
    system_msg = (
        "Bạn là trợ lý kỹ thuật. CHỈ dùng thông tin trong NGỮ CẢNH và SƯỜN TRÍCH XUẤT để trả lời. "
        "Không nêu suy nghĩ trung gian, không xuất thẻ <think>. "
        "Nếu không đủ thông tin, hãy nói 'Không thấy trong tài liệu'. "
        "Luôn trả lời HOÀN TOÀN bằng tiếng Việt."
    )

    # Thêm sườn nếu có
    outline_block = f"\nSƯỜN TRÍCH XUẤT (các bước tìm thấy):\n{outline}\n" if outline else ""

    user_msg = (
        "NGỮ CẢNH (trích từ tài liệu, có thể gồm nhiều đoạn):\n"
        f"{context}\n"
        f"{outline_block}\n"
        "YÊU CẦU:\n"
        f"- Câu hỏi: {question}\n"
        "- Nếu câu hỏi hỏi 'các bước', hãy liệt kê đầy đủ theo dạng 1), 2), 3)...\n"
        "- Không thêm kiến thức ngoài ngữ cảnh.\n"
        "- Không dùng tiếng Anh.\n"
        "- Ngắn gọn, đúng trọng tâm."
    )
    return (
        "<|im_start|>system\n" + system_msg + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user_msg + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def count_tokens(text: str) -> int:
    """Đếm tokens CHÍNH XÁC bằng tokenizer thực của model"""
    if MODEL is None:
        # Fallback nếu chưa load model
        words = len(re.findall(r'\w+', text))
        return int(words / 0.7)

    try:
        ids = MODEL.tokenize(text.encode("utf-8"))
        return len(ids)
    except:
        # Fallback nếu lỗi
        words = len(re.findall(r'\w+', text))
        return int(words / 0.7)

def filter_think_tags(text: str) -> str:
    """
    Lọc bỏ <think> tags - IMPROVED để xử lý unclosed tags.
    Kết hợp từ rag_ultimate.py + better.py improvements.
    """
    # 1. Xóa các đoạn <think>...</think> hoàn chỉnh
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Nếu còn <think> không close, xóa TẤT CẢ từ <think> đến cuối
    if '<think>' in text.lower():
        parts = re.split(r'<think>', text, flags=re.IGNORECASE)
        text = parts[0]  # Chỉ lấy phần trước <think>

    # 3. Xóa các tag lẻ còn sót (</think> orphan)
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)

    return text.strip()

def extract_step_outline(candidates: List[str]) -> Optional[str]:
    """
    Trích xuất "sườn" các bước từ top candidates.
    Tính năng VÀNG từ better.py!

    Ví dụ output:
    1) Làm sạch bề mặt chi tiết
    2) Lắp đặt dao
    3) Offset dao
    ...
    """
    found = {}

    for text in candidates:
        # Match "Bước 1:", "Bước 2:", etc.
        for m in re.finditer(r"(?mi)^\s*B[uư]ớc\s*(\d+)\s*[:：]?\s*(.*)$", text):
            num = int(m.group(1))
            title = m.group(2).strip()

            # Normalize whitespace
            title = re.sub(r"\s+", " ", title)

            # Chỉ thêm nếu chưa có và title không rỗng
            if num not in found and title:
                found[num] = title

    if not found:
        return None

    # Sắp xếp theo số bước
    items = []
    for n in sorted(found.keys()):
        items.append(f"{n}) {found[n]}")

    return "\n".join(items)

def ensure_vietnamese(text: str, question: str) -> str:
    """
    Hậu kiểm và tự động dịch nếu model trả lời tiếng Anh.
    Kết hợp cả 2 approaches: warning + auto-translate.
    """
    if not ENFORCE_VI:
        return text

    # Đếm Vietnamese vs English
    vietnamese_chars = len(re.findall(
        r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
        text.lower()
    ))
    english_words = len(re.findall(r'\b[a-z]{3,}\b', text.lower()))

    # Nếu quá nhiều tiếng Anh
    if english_words > 10 and vietnamese_chars < 5:
        print("CANH BAO: Model tra loi toan tieng Anh!")
        print("  -> Dang tu dong dich lai...", flush=True)

        try:
            # Tự động dịch lại
            fix_prompt = (
                "<|im_start|>system\n"
                "Dịch chính xác sang tiếng Việt, giữ ý nghĩa và cấu trúc.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n{text}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            out = MODEL(
                fix_prompt,
                max_tokens=max(64, MAX_TOK_OUT // 2),
                temperature=0.1,
                top_k=TOP_K,
                top_p=TOP_P,
                seed=SEED,
                stop=["<|im_end|>"]
            )
            translated = out["choices"][0]["text"].strip()
            translated = filter_think_tags(translated)

            # Kiểm tra xem dịch có thành công không
            vi_chars_after = len(re.findall(
                r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
                translated.lower()
            ))

            if vi_chars_after > 3:
                print("  OK - Da dich lai thanh cong!\n")
                return translated
            else:
                print("  CANH BAO: Dich lai that bai!")
                print("  => Giu nguyen output goc.\n")
                return text + "\n\n[LUU Y: Model tra loi bang tieng Anh. Nen dung model tot hon.]"

        except Exception as e:
            print(f"  LOI khi dich: {e}")
            return text + "\n\n[LOI: Khong the tu dong dich. Vui long dung model tot hon.]"

    return text

# ================= Index Management =================

def build_index_if_needed():
    """Build index nếu chưa có"""
    if INDEX_JSON.exists() and CHUNKS_JSON.exists():
        return

    print(f"Doc document tu: {DOC_PATH}")
    try:
        text = Path(DOC_PATH).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"LOI: Khong tim thay file: {DOC_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"LOI doc file: {e}")
        sys.exit(1)

    print("Dang chia chunks...")
    chunks = split_into_chunks(text, MAX_CHARS, OVERLAP)
    print(f"OK - {len(chunks)} chunks")

    print("Dang xay dung BM25 index...")
    bm25, tokenized = build_bm25(chunks)

    # Lưu index
    try:
        INDEX_JSON.write_text(json.dumps({
            "vocab": bm25.vocab,
            "df": bm25.df,
            "idf": bm25.idf,
            "avgdl": bm25.avgdl,
            "doc_lengths": bm25.doc_lengths
        }), encoding="utf-8")

        CHUNKS_JSON.write_text(json.dumps({
            "chunks": chunks,
            "tokenized": tokenized
        }, ensure_ascii=False), encoding="utf-8")

        print(f"OK - Da luu index tai {INDEX_JSON}\n")
    except Exception as e:
        print(f"LOI khi luu index: {e}")
        sys.exit(1)

def load_index() -> Tuple[BM25Index, List[str], List[List[str]]]:
    """Load index từ file"""
    try:
        ij = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        cj = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("LOI: Khong tim thay index! Chay build_index() truoc.")
        sys.exit(1)
    except Exception as e:
        print(f"LOI load index: {e}")
        sys.exit(1)

    bm25 = BM25Index(
        vocab={k: int(v) for k, v in ij["vocab"].items()},
        df=[int(x) for x in ij["df"]],
        idf=[float(x) for x in ij["idf"]],
        avgdl=float(ij["avgdl"]),
        doc_lengths=[int(x) for x in ij["doc_lengths"]],
    )
    return bm25, cj["chunks"], cj["tokenized"]

# ================= Retrieval & Answer =================

def retrieve(
    query: str,
    k: int = K_TOP,
    k_cand: int = K_CAND
) -> Tuple[List[str], List[int], List[float], Optional[str]]:
    """Retrieve top-k chunks với BM25 + Boost + MMR + extract outline"""
    bm25, chunks, tokenized = load_index()
    q_tokens = tokenize_vi(query)

    # BM25 scores
    scores = bm25_scores(q_tokens, bm25, tokenized)

    # Structural boost
    boost_query = any(x in query.lower() for x in ["các bước", "quy trình", "trình tự"])
    if boost_query:
        print("Dang tim kiem (BM25 + MMR + Boost cau truc)...", flush=True)
    else:
        print("Dang tim kiem (BM25 + MMR)...", flush=True)

    boosted_scores = [
        s + (structural_boost_score(chunks[i]) if boost_query else 0)
        for i, s in enumerate(scores)
    ]

    # Chọn ứng viên
    cand_idx = np.argsort(boosted_scores)[::-1][:max(k, k_cand)]
    cand_idx = cand_idx.tolist()

    # Trích xuất sườn từ ALL candidates (tính năng VÀNG từ better.py!)
    candidate_chunks = [chunks[i] for i in cand_idx]
    outline = extract_step_outline(candidate_chunks)
    if outline:
        print("OK - Trích xuất được sườn các bước từ candidates")

    # MMR để đa dạng hóa
    sel = mmr_select(
        candidate_chunks,
        [tokenized[i] for i in cand_idx],
        [boosted_scores[i] for i in cand_idx],
        k=k,
        lambda_=LAMBDA_MMR
    )

    final_idx = [cand_idx[i] for i in sel]
    final_scores = [boosted_scores[i] for i in final_idx]
    final_chunks = [chunks[i] for i in final_idx]

    print(f"OK - Tim thay {len(final_chunks)} chunks")
    return final_chunks, final_idx, final_scores, outline

def limit_context_by_tokens(chunks: List[str], budget: int) -> List[str]:
    """Cắt chunks nếu vượt token budget"""
    ctx = "\n\n---\n\n".join(chunks)
    used = count_tokens(ctx)

    while used > budget and len(chunks) > 1:
        chunks.pop()  # Bỏ chunk ít liên quan nhất (cuối)
        ctx = "\n\n---\n\n".join(chunks)
        used = count_tokens(ctx)

    if used > budget:
        print(f"CANH BAO: Context van qua dai ({used} tokens), nhung da giam toi thieu.")

    return chunks

def answer(question: str) -> Dict:
    """Trả lời câu hỏi với full pipeline"""
    # Đảm bảo model đã load
    init_model()

    # Retrieve chunks + outline
    chunks, idx, scores, outline = retrieve(question, k=K_TOP, k_cand=K_CAND)

    # Limit context by token budget
    budget = max(256, N_CTX - MAX_TOK_OUT - 150)  # 150 = prompt overhead
    chunks = limit_context_by_tokens(chunks, budget)
    context = "\n\n---\n\n".join(chunks)

    # Build prompt WITH outline (tính năng VÀNG!)
    prompt = build_chat_prompt(context, question, outline=outline)

    # Generate
    print("Dang sinh cau tra loi...", flush=True)
    try:
        out = MODEL(
            prompt,
            max_tokens=MAX_TOK_OUT,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PEN,
            seed=SEED,
            stop=["<|im_end|>"]
        )
        text = out["choices"][0]["text"].strip()
    except Exception as e:
        print(f"LOI khi generate: {e}")
        return {
            "question": question,
            "answer": f"[LOI] {e}",
            "retrieved_indices": idx,
            "retrieved_scores": scores,
            "retrieved_chunks": chunks,
            "outline": outline
        }

    # Post-processing
    text = filter_think_tags(text)
    text = ensure_vietnamese(text, question)

    print("OK - Da tao cau tra loi!\n")

    return {
        "question": question,
        "answer": text,
        "retrieved_indices": idx,
        "retrieved_scores": [float(f"{s:.3f}") for s in scores],
        "retrieved_chunks": chunks,
        "outline": outline
    }

# ================= Main =================

def main():
    # Fix encoding cho Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("RAG ULTIMATE - Best of Both Worlds")
    print("=" * 60)

    # Build index if needed
    if not INDEX_JSON.exists() or not CHUNKS_JSON.exists():
        print("Index chua ton tai, dang xay dung...\n")
        build_index_if_needed()
    else:
        print("Index da ton tai, bo qua build.\n")

    # Get question
    q = "Các bước vận hành máy phay CNC gồm những gì?"
    if len(sys.argv) >= 2:
        q = " ".join(sys.argv[1:])

    print("=" * 60)
    print(f"Cau hoi: {q}")
    print("=" * 60 + "\n")

    # Answer
    res = answer(q)

    # Display
    print("=" * 60)
    print("TRA LOI")
    print("=" * 60)
    print(res["answer"])

    # Display outline if available
    if res.get("outline"):
        print("\n" + "=" * 60)
        print("SUON TRICH XUAT (các bước tìm thấy)")
        print("=" * 60)
        print(res["outline"])

    print("\n" + "=" * 60)
    print("NGU CANH SU DUNG")
    print("=" * 60)
    for i, (sc, ch) in enumerate(zip(res["retrieved_scores"], res["retrieved_chunks"]), 1):
        print(f"\n[Chunk #{i}] (Diem: {sc})")
        print("-" * 60)
        preview = ch[:400] + ("..." if len(ch) > 400 else "")
        print(preview)

if __name__ == "__main__":
    main()
