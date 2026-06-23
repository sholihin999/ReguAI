from langchain_community.vectorstores import FAISS

from src.embedding.embedding_model import get_embedding_model
from src.retrieval.retrieval_utils import (
    TOP_K_INITIAL,
    TOP_K_FINAL,
    build_search_query,
    rerank_results,
    is_in_scope,
)
from src.utils.config import VECTOR_DB_PATH, TOP_K_DEFINITION


def is_definition_question(query: str) -> bool:
    q = query.lower().strip()
    return any(pattern in q for pattern in [
        "apa itu",
        "apa definisi",
        "definisi",
        "pengertian",
        "apa yang dimaksud",
        "yang dimaksud dengan",
    ])


test_cases = [
    {"question": "Apa itu data pribadi?", "expected_pasals": ["1"], "expected_doc_keyword": "Pelindungan Data Pribadi"},
    {"question": "Apa itu subjek data pribadi?", "expected_pasals": ["1"], "expected_doc_keyword": "Pelindungan Data Pribadi"},
    {"question": "Apa itu pengendali data pribadi?", "expected_pasals": ["1"], "expected_doc_keyword": "Pelindungan Data Pribadi"},
    {"question": "Apa itu informasi elektronik?", "expected_pasals": ["1"], "expected_doc_keyword": "Nomor 11 Tahun 2008"},
    {"question": "Apa itu dokumen elektronik?", "expected_pasals": ["1"], "expected_doc_keyword": "Nomor 11 Tahun 2008"},
    {"question": "Apa itu sistem elektronik?", "expected_pasals": ["1"], "expected_doc_keyword": "Nomor 11 Tahun 2008"},
    {"question": "Apa itu transaksi elektronik?", "expected_pasals": ["1"], "expected_doc_keyword": "Nomor 11 Tahun 2008"},
    {"question": "Apa yang dimaksud dengan tindak pidana?", "expected_pasals": ["12"], "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana"},
    {"question": "Apa sanksi penyalahgunaan data pribadi?", "expected_pasals": ["65", "66", "67", "68", "69"], "expected_doc_keyword": "Pelindungan Data Pribadi"},
    {"question": "Apa sanksi doxing?", "expected_pasals": ["65", "66", "67", "68", "69"], "expected_doc_keyword": "Pelindungan Data Pribadi"},
    {"question": "Apa hak subjek data pribadi?", "expected_pasals": ["5", "6", "7", "8", "9", "10", "11", "12", "13"], "expected_doc_keyword": "Pelindungan Data Pribadi"},
    {"question": "Apa sanksi pencemaran nama baik di media sosial?", "expected_pasals": ["27A", "45", "45A", "45B"], "expected_doc_keyword": "ITE"},
    {"question": "Kategori II itu dendanya berapa?", "expected_pasals": ["79"], "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana"},
    {"question": "Apa sanksi pencurian?", "expected_pasals": ["476", "477", "478"], "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana"},
    {"question": "Apa sanksi penipuan?", "expected_pasals": ["492"], "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana"},
    {"question": "Apa itu hukum?", "expected_pasals": [], "expected_doc_keyword": "", "expected_empty": True},
    {"question": "Apa definisi negara hukum?", "expected_pasals": [], "expected_doc_keyword": "", "expected_empty": True},
]


def is_guardrail_question(case):
    return case.get("expected_empty", False)


print("Loading embedding model...")

embeddings = get_embedding_model()

print("Loading FAISS...")

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

top1_correct = 0
top3_correct = 0
top5_correct = 0
mrr_total = 0.0
precision5_total = 0.0
recall5_total = 0.0
guardrail_correct = 0
guardrail_total = 0

print("\n=== EVALUASI RETRIEVAL REGUAI ===\n")

for idx, case in enumerate(test_cases, 1):
    question = case["question"]
    expected_pasals = case["expected_pasals"]
    expected_doc_keyword = case["expected_doc_keyword"].lower()
    expected_empty = is_guardrail_question(case)

    retrieved = []

    if expected_empty and not is_in_scope(question):
        final_results = []
    else:
        retrieval_k = TOP_K_DEFINITION if is_definition_question(question) else TOP_K_INITIAL

        initial_results = vectorstore.similarity_search_with_score(
            build_search_query(question),
            k=retrieval_k,
        )

        final_results = rerank_results(question, initial_results, top_k=TOP_K_FINAL)

    for doc, score, k_score, final_score in final_results:
        retrieved.append({
            "pasal": str(doc.metadata.get("pasal", "")),
            "doc_name": doc.metadata.get("nama_dokumen", ""),
            "score": score,
            "keyword_score": k_score,
            "final_score": final_score,
        })

    def is_correct(item):
        return (
            item["pasal"] in expected_pasals
            and expected_doc_keyword in item["doc_name"].lower()
        )

    if expected_empty:
        guardrail_total += 1
        passed_guardrail = len(retrieved) == 0
        guardrail_correct += int(passed_guardrail)

        top1 = passed_guardrail
        top3 = passed_guardrail
        top5 = passed_guardrail
        reciprocal_rank = 1.0 if passed_guardrail else 0.0
        precision_at_5 = 1.0 if passed_guardrail else 0.0
        recall_at_5 = 1.0 if passed_guardrail else 0.0
    else:
        top1 = any(is_correct(item) for item in retrieved[:1])
        top3 = any(is_correct(item) for item in retrieved[:3])
        top5 = any(is_correct(item) for item in retrieved[:5])

        correct_positions = [
            rank for rank, item in enumerate(retrieved, 1)
            if is_correct(item)
        ]

        reciprocal_rank = 1 / correct_positions[0] if correct_positions else 0.0

        precision_at_5 = (
            sum(1 for item in retrieved[:5] if is_correct(item)) / 5
            if retrieved
            else 0.0
        )

        recall_at_5 = (
            min(
                sum(1 for item in retrieved[:5] if is_correct(item)) / len(expected_pasals),
                1.0,
            )
            if expected_pasals
            else 0.0
        )

    top1_correct += int(top1)
    top3_correct += int(top3)
    top5_correct += int(top5)
    mrr_total += reciprocal_rank
    precision5_total += precision_at_5
    recall5_total += recall_at_5

    print("=" * 80)
    print(f"{idx}. Pertanyaan: {question}")
    print(f"Expected Pasal: {expected_pasals if expected_pasals else '-'}")
    print(f"Expected Dokumen: {case['expected_doc_keyword'] if case['expected_doc_keyword'] else '-'}")
    print(f"Guardrail Test: {expected_empty}")
    print("\nHasil Retrieval:")

    if not retrieved:
        print("Tidak ada hasil retrieval.")

    for rank, item in enumerate(retrieved, 1):
        status = "✅" if is_correct(item) else "❌"
        print(
            f"{rank}. {status} Pasal {item['pasal']} | "
            f"{item['doc_name']} | "
            f"Distance: {item['score']:.4f} | "
            f"Keyword: {item['keyword_score']:.2f} | "
            f"Final: {item['final_score']:.4f}"
        )

    print()
    print(f"Top-1 Correct : {top1}")
    print(f"Top-3 Correct : {top3}")
    print(f"Top-5 Correct : {top5}")
    print(f"MRR           : {reciprocal_rank:.4f}")
    print(f"Precision@5   : {precision_at_5:.4f}")
    print(f"Recall@5      : {recall_at_5:.4f}")

    if expected_empty:
        print(f"Guardrail Pass: {passed_guardrail}")

    print()

total = len(test_cases)

print("\n=== HASIL AKHIR ===")
print(f"Total test case     : {total}")
print(f"Top-1 Accuracy      : {top1_correct}/{total} = {top1_correct / total:.2%}")
print(f"Top-3 Accuracy      : {top3_correct}/{total} = {top3_correct / total:.2%}")
print(f"Top-5 Accuracy      : {top5_correct}/{total} = {top5_correct / total:.2%}")
print(f"MRR                 : {mrr_total / total:.4f}")
print(f"Precision@5         : {precision5_total / total:.4f}")
print(f"Recall@5            : {recall5_total / total:.4f}")

if guardrail_total:
    print(
        f"Guardrail Accuracy  : "
        f"{guardrail_correct}/{guardrail_total} = {guardrail_correct / guardrail_total:.2%}"
    )
else:
    print("Guardrail Accuracy  : Tidak ada test case guardrail")