import hashlib
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field


VECTOR_SIZE = 1024
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
CLUE_KEYWORDS = {
    "人物": ["亲属", "外甥", "姐姐", "姐夫", "朋友"],
    "金额": ["现金", "礼金", "回扣", "购物卡", "好处费", "资金"],
    "关系": ["关系户", "代持", "关联公司", "打招呼", "利益输送"],
    "行为": ["承揽", "串通", "套取", "虚报", "指定", "优亲厚友"],
    "项目": ["工程", "采购", "改造", "安置房", "项目"],
}

app = FastAPI(title="mock-model-server")


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[List[str], str]
    encoding_format: Optional[str] = None


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: int = 20
    return_documents: bool = False


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.1


@app.get("/")
@app.get("/health")
@app.get("/v1/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
@app.get("/v1/models")
async def models() -> Dict[str, List[Dict[str, str]]]:
    return {
        "object": "list",
        "data": [
            {"id": "qwen3-8b-awq", "object": "model"},
            {"id": "bge-m3", "object": "model"},
            {"id": "bge-reranker-v2-m3", "object": "model"},
        ],
    }


@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(payload: EmbeddingRequest) -> Dict[str, object]:
    texts = payload.input if isinstance(payload.input, list) else [payload.input]
    data = []
    for index, text in enumerate(texts):
        dense_vector, sparse_vector = embed_text(text)
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": dense_vector,
                "sparse_embedding": sparse_vector,
            }
        )
    return {"object": "list", "data": data, "model": payload.model}


@app.post("/rerank")
@app.post("/v1/rerank")
async def rerank(payload: RerankRequest) -> Dict[str, object]:
    scored = []
    query_tokens = tokenize(payload.query)
    for index, document in enumerate(payload.documents):
        score = similarity_score(query_tokens, tokenize(document))
        scored.append({"index": index, "relevance_score": round(score, 6)})
    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    return {"results": scored[: payload.top_n]}


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest) -> Dict[str, object]:
    user_prompt = extract_user_prompt(payload.messages)
    if "ranked_cases" in user_prompt:
        content = build_duplicate_response(user_prompt)
    else:
        content = build_clue_response(user_prompt)
    return {
        "id": "mock-chatcmpl",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def extract_user_prompt(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str):
                return message.content
            return "".join(part.get("text", "") for part in message.content if isinstance(part, dict))
    return ""


def build_duplicate_response(prompt: str) -> str:
    new_case = extract_json_block(prompt, "新案件：", "候选案件：")
    candidates = extract_json_block(prompt, "候选案件：", "输出格式：")
    query = json.loads(new_case or "{}")
    candidate_items = json.loads(candidates or "[]")

    query_tokens = tokenize(" ".join(
        [
            query.get("location", ""),
            query.get("reporter") or "",
            " ".join(query.get("reported_persons", [])),
            query.get("description", ""),
        ]
    ))
    query_persons = set(query.get("reported_persons", []))
    ranked_cases = []
    for index, item in enumerate(candidate_items, start=1):
        candidate_tokens = tokenize(" ".join(
            [
                item.get("location", ""),
                item.get("reporter") or "",
                " ".join(item.get("reported_persons", [])),
                item.get("description_text", ""),
            ]
        ))
        overlap_score = similarity_score(query_tokens, candidate_tokens)
        same_location = item.get("location") == query.get("location")
        overlap_persons = query_persons.intersection(set(item.get("reported_persons", [])))
        combined = overlap_score + (0.18 if same_location else 0.0) + min(0.12 * len(overlap_persons), 0.24)
        similarity = max(0, min(100, int(round(combined * 100))))
        if similarity < 45:
            continue
        reason_parts = []
        if overlap_persons:
            reason_parts.append("被举报人重合: {0}".format("、".join(sorted(overlap_persons))))
        if same_location:
            reason_parts.append("属地一致")
        reason_parts.append("案情词项相似度较高")
        ranked_cases.append(
            {
                "case_id": item["case_id"],
                "similarity_score": similarity,
                "rank": index,
                "reason": "，".join(reason_parts) + "。",
                "_combined": combined,
            }
        )

    ranked_cases.sort(key=lambda item: item["_combined"], reverse=True)
    for rank, item in enumerate(ranked_cases, start=1):
        item["rank"] = rank
        item.pop("_combined", None)

    body = {
        "is_duplicate": bool(ranked_cases),
        "ranked_cases": ranked_cases[:5],
    }
    return json.dumps(body, ensure_ascii=False)


def build_clue_response(prompt: str) -> str:
    new_case = extract_json_block(prompt, "新案件：", "重复案件：")
    candidates = extract_json_block(prompt, "重复案件：", "输出格式：")
    query = json.loads(new_case or "{}")
    candidate_items = json.loads(candidates or "[]")
    base_text = " ".join(query.get("reported_persons", [])) + " " + query.get("description", "")
    clues = []

    for item in candidate_items:
        if item.get("location") != query.get("location"):
            continue
        if not set(query.get("reported_persons", [])).intersection(set(item.get("reported_persons", []))):
            continue
        candidate_text = item.get("description_text", "")
        for clue_type, keywords in CLUE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in candidate_text and keyword not in base_text:
                    clues.append(
                        {
                            "source_case_id": item["case_id"],
                            "clue_type": clue_type,
                            "description": "历史案件 {0} 提到“{1}”相关情节，可作为延伸核查线索。".format(
                                item["case_id"], keyword
                            ),
                            "risk_level": "高" if clue_type in {"金额", "关系", "行为"} else "中",
                        }
                    )
                    break
            if len(clues) >= 3:
                break
        if len(clues) >= 3:
            break

    return json.dumps({"new_clues": clues[:3]}, ensure_ascii=False)


def extract_json_block(text: str, start_label: str, end_label: str) -> str:
    start_index = text.find(start_label)
    if start_index == -1:
        return ""
    start_index += len(start_label)
    end_index = text.find(end_label, start_index)
    if end_index == -1:
        end_index = len(text)
    return text[start_index:end_index].strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def embed_text(text: str) -> Tuple[List[float], Dict[str, List[Union[float, int]]]]:
    counts = Counter(tokenize(text))
    dense = [0.0] * VECTOR_SIZE
    sparse_buckets = Counter()
    for token, count in counts.items():
        bucket = token_bucket(token)
        dense[bucket] += float(count)
        sparse_buckets[bucket] += float(count)
    norm = math.sqrt(sum(value * value for value in dense)) or 1.0
    dense = [round(value / norm, 6) for value in dense]
    ordered = sorted(sparse_buckets.items(), key=lambda item: item[0])
    return dense, {
        "indices": [bucket for bucket, _ in ordered],
        "values": [round(value, 6) for _, value in ordered],
    }


def token_bucket(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % VECTOR_SIZE


def similarity_score(tokens_a: List[str], tokens_b: List[str]) -> float:
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    shared = set(counter_a).intersection(counter_b)
    numerator = sum(min(counter_a[token], counter_b[token]) for token in shared)
    denominator = max(sum(counter_a.values()), 1)
    coverage = numerator / denominator
    jaccard_denominator = len(set(counter_a).union(counter_b)) or 1
    jaccard = len(shared) / jaccard_denominator
    return round(0.7 * coverage + 0.3 * jaccard, 6)
