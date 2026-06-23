import pandas as pd
import streamlit as st
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

st.set_page_config(page_title="ReguAI Evaluation", page_icon="📊", layout="wide")


def is_definition_question(query: str) -> bool:
    q = query.lower().strip()
    return any(
        pattern in q
        for pattern in [
            "apa itu",
            "apa definisi",
            "definisi",
            "pengertian",
            "apa yang dimaksud",
            "yang dimaksud dengan",
        ]
    )


test_cases = [
    {
        "question": "Apa itu data pribadi?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Pelindungan Data Pribadi",
    },
    {
        "question": "Apa itu subjek data pribadi?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Pelindungan Data Pribadi",
    },
    {
        "question": "Apa itu pengendali data pribadi?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Pelindungan Data Pribadi",
    },
    {
        "question": "Apa itu informasi elektronik?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Nomor 11 Tahun 2008",
    },
    {
        "question": "Apa itu dokumen elektronik?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Nomor 11 Tahun 2008",
    },
    {
        "question": "Apa itu sistem elektronik?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Nomor 11 Tahun 2008",
    },
    {
        "question": "Apa itu transaksi elektronik?",
        "expected_pasals": ["1"],
        "expected_doc_keyword": "Nomor 11 Tahun 2008",
    },
    {
        "question": "Apa yang dimaksud dengan tindak pidana?",
        "expected_pasals": ["12"],
        "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana",
    },
    {
        "question": "Apa sanksi penyalahgunaan data pribadi?",
        "expected_pasals": ["65", "66", "67", "68", "69"],
        "expected_doc_keyword": "Pelindungan Data Pribadi",
    },
    {
        "question": "Apa sanksi doxing?",
        "expected_pasals": ["65", "66", "67", "68", "69"],
        "expected_doc_keyword": "Pelindungan Data Pribadi",
    },
    {
        "question": "Apa hak subjek data pribadi?",
        "expected_pasals": ["5", "6", "7", "8", "9", "10", "11", "12", "13"],
        "expected_doc_keyword": "Pelindungan Data Pribadi",
    },
    {
        "question": "Apa sanksi pencemaran nama baik di media sosial?",
        "expected_pasals": ["27A", "45", "45A", "45B"],
        "expected_doc_keyword": "ITE",
    },
    {
        "question": "Kategori II itu dendanya berapa?",
        "expected_pasals": ["79"],
        "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana",
    },
    {
        "question": "Apa sanksi pencurian?",
        "expected_pasals": ["476", "477", "478"],
        "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana",
    },
    {
        "question": "Apa sanksi penipuan?",
        "expected_pasals": ["492"],
        "expected_doc_keyword": "Kitab Undang-Undang Hukum Pidana",
    },
    {
        "question": "Apa itu hukum?",
        "expected_pasals": [],
        "expected_doc_keyword": "",
        "expected_empty": True,
    },
    {
        "question": "Apa definisi negara hukum?",
        "expected_pasals": [],
        "expected_doc_keyword": "",
        "expected_empty": True,
    },
]


@st.cache_resource
def load_vectorstore():
    embeddings = get_embedding_model()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def is_correct_result(item, expected_pasals, expected_doc_keyword):
    return (
        item["pasal"] in expected_pasals
        and expected_doc_keyword.lower() in item["doc_name"].lower()
    )


def is_guardrail_question(case):
    return case.get("expected_empty", False)


st.title("📊 ReguAI Retrieval Evaluation Dashboard")
st.caption(
    "Evaluasi retrieval berbasis pasal, dokumen, ranking, dan guardrail untuk pertanyaan di luar cakupan basis data."
)

vectorstore = load_vectorstore()

rows = []
detail_results = []

top1_correct = 0
top3_correct = 0
top5_correct = 0
mrr_total = 0.0
precision5_total = 0.0
recall5_total = 0.0
guardrail_correct = 0
guardrail_total = 0

for idx, case in enumerate(test_cases, 1):
    question = case["question"]
    expected_pasals = case["expected_pasals"]
    expected_doc_keyword = case["expected_doc_keyword"]
    expected_empty = is_guardrail_question(case)

    retrieved = []

    if expected_empty and not is_in_scope(question):
        initial_results = []
        final_results = []
    else:
        retrieval_k = (
            TOP_K_DEFINITION if is_definition_question(question) else TOP_K_INITIAL
        )

        initial_results = vectorstore.similarity_search_with_score(
            build_search_query(question),
            k=retrieval_k,
        )

        final_results = rerank_results(question, initial_results, top_k=TOP_K_FINAL)

    for rank, (doc, distance_score, k_score, final_score) in enumerate(
        final_results, 1
    ):
        item = {
            "rank": rank,
            "question": question,
            "pasal": str(doc.metadata.get("pasal", "")),
            "doc_name": doc.metadata.get("nama_dokumen", ""),
            "distance_score": float(distance_score),
            "keyword_score": float(k_score),
            "final_score": float(final_score),
            "content": doc.page_content.replace("passage: ", "")[:1500],
        }

        item["correct"] = is_correct_result(
            item,
            expected_pasals,
            expected_doc_keyword,
        )

        retrieved.append(item)
        detail_results.append(item)

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
        evaluation_type = "Guardrail"
    else:
        top1 = any(item["correct"] for item in retrieved[:1])
        top3 = any(item["correct"] for item in retrieved[:3])
        top5 = any(item["correct"] for item in retrieved[:5])

        correct_positions = [
            rank for rank, item in enumerate(retrieved, 1) if item["correct"]
        ]

        reciprocal_rank = 1 / correct_positions[0] if correct_positions else 0.0

        precision_at_5 = (
            sum(1 for item in retrieved[:5] if item["correct"]) / 5
            if retrieved
            else 0.0
        )

        recall_at_5 = (
            min(
                sum(1 for item in retrieved[:5] if item["correct"])
                / len(expected_pasals),
                1.0,
            )
            if expected_pasals
            else 0.0
        )

        evaluation_type = "Retrieval"

    top1_correct += int(top1)
    top3_correct += int(top3)
    top5_correct += int(top5)
    mrr_total += reciprocal_rank
    precision5_total += precision_at_5
    recall5_total += recall_at_5

    rows.append(
        {
            "No": idx,
            "Jenis": evaluation_type,
            "Pertanyaan": question,
            "Expected Pasal": ", ".join(expected_pasals) if expected_pasals else "-",
            "Expected Dokumen": expected_doc_keyword if expected_doc_keyword else "-",
            "Top-1": top1,
            "Top-3": top3,
            "Top-5": top5,
            "MRR": round(reciprocal_rank, 4),
            "Precision@5": round(precision_at_5, 4),
            "Recall@5": round(recall_at_5, 4),
            "Ranking 1": (
                f"Pasal {retrieved[0]['pasal']} - {retrieved[0]['doc_name']}"
                if retrieved
                else "Tidak ada hasil"
            ),
        }
    )

total = len(test_cases)

top1_accuracy = top1_correct / total if total else 0
top3_accuracy = top3_correct / total if total else 0
top5_accuracy = top5_correct / total if total else 0
mrr_score = mrr_total / total if total else 0
precision5_score = precision5_total / total if total else 0
recall5_score = recall5_total / total if total else 0
guardrail_score = guardrail_correct / guardrail_total if guardrail_total else 0

st.markdown("### Ringkasan Metrik")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Top-1 Accuracy", f"{top1_accuracy:.2%}")
col2.metric("Top-3 Accuracy", f"{top3_accuracy:.2%}")
col3.metric("Top-5 Accuracy", f"{top5_accuracy:.2%}")
col4.metric("MRR", f"{mrr_score:.4f}")

col5, col6, col7 = st.columns(3)
col5.metric("Precision@5", f"{precision5_score:.4f}")
col6.metric("Recall@5", f"{recall5_score:.4f}")
col7.metric("Guardrail Accuracy", f"{guardrail_score:.2%}")

st.divider()

st.markdown("### Ringkasan Evaluasi")

df_rows = pd.DataFrame(rows)

with st.container():
    st.dataframe(
        df_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Top-1": st.column_config.CheckboxColumn("Top-1"),
            "Top-3": st.column_config.CheckboxColumn("Top-3"),
            "Top-5": st.column_config.CheckboxColumn("Top-5"),
            "MRR": st.column_config.NumberColumn("MRR", format="%.4f"),
            "Precision@5": st.column_config.NumberColumn("Precision@5", format="%.4f"),
            "Recall@5": st.column_config.NumberColumn("Recall@5", format="%.4f"),
        },
    )

st.info(
    "Precision@5 dapat terlihat lebih rendah pada pertanyaan definisi karena hanya satu pasal yang menjadi sumber utama. "
    "Top-1 Accuracy dan MRR lebih representatif untuk melihat apakah sumber utama ditemukan pada peringkat pertama. "
    "Guardrail Accuracy mengukur apakah sistem menolak pertanyaan yang berada di luar cakupan basis data."
)

st.divider()

st.markdown("### Detail Retrieval")

selected_question = st.selectbox(
    "Pilih pertanyaan:",
    [case["question"] for case in test_cases],
)

selected_case = next(
    case for case in test_cases if case["question"] == selected_question
)

filtered = [item for item in detail_results if item["question"] == selected_question]

if not filtered:
    if selected_case.get("expected_empty", False):
        st.success(
            "Tidak ada hasil retrieval. Ini sesuai untuk pertanyaan guardrail/out-of-scope."
        )
    else:
        st.warning("Tidak ada hasil retrieval untuk pertanyaan ini.")

for item in filtered:
    status = "✅ Benar" if item["correct"] else "❌ Tidak sesuai"

    with st.expander(
        f"Rank {item['rank']} | {status} | Pasal {item['pasal']} | "
        f"Distance {item['distance_score']:.4f} | "
        f"Keyword {item['keyword_score']:.2f} | "
        f"Final {item['final_score']:.4f}"
    ):
        st.write(f"**Dokumen:** {item['doc_name']}")
        st.write(f"**Pasal:** {item['pasal']}")
        st.write(f"**Distance Score:** {item['distance_score']:.4f}")
        st.write(f"**Keyword Score:** {item['keyword_score']:.2f}")
        st.write(f"**Final Score:** {item['final_score']:.4f}")
        st.markdown("**Potongan isi:**")
        st.write(item["content"])
