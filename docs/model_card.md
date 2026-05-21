# Model Card — Policy Q&A RAG Chain

**Model Name:** `policy_qa_rag_chain`
**Version:** 1.0.0
**Type:** Retrieval-Augmented Generation (RAG) chain
**Author:** SNS Square
**Last Updated:** 2026-05-21
**Registry:** Unity Catalog — `{catalog}.{schema}.policy_qa_rag_chain`

---

## Intended Use

**Primary use:** Answer natural-language questions about enterprise policy documents (HR, IT security, expenses, data privacy, remote work) by retrieving relevant document chunks and generating grounded, cited answers.

**Intended users:** Enterprise employees seeking policy information; HR teams reducing repetitive query volume.

**Out-of-scope uses:**
- Legal advice or binding HR decisions
- Questions outside the ingested policy document corpus
- Real-time regulatory compliance checking
- Processing of sensitive personal data beyond what is described in the guardrails section

---

## Model Architecture

| Component | Details |
|---|---|
| Retriever | Mosaic AI Vector Search — BGE-Large-EN-v1.5 embeddings, cosine similarity, top-5 chunks |
| LLM | Llama 3.1 8B Instruct (dev) / 70B Instruct (production demo) via Foundation Models API |
| Chain | LangChain LCEL: retriever → context formatter → ChatPromptTemplate → ChatDatabricks → StrOutputParser |
| Guardrails | Input: PII redaction, injection detection, length limit. Output: PII scan, verbatim reproduction check, similarity disclaimer |

---

## Training Data

This is a RAG system — no model training is performed. The LLM (Llama 3.1) is used as-is from the Mosaic AI Foundation Models API. The retrieval corpus consists of:

- **Sample data:** 5 synthetic policy documents (leave, IT security, expenses, data privacy, remote work) — no real employee data
- **Production:** Customer-supplied policy documents uploaded to Unity Catalog Volumes

---

## Evaluation Results

Evaluated on 20 Q&A pairs covering all 5 sample policy domains.

| Metric | Value | Notes |
|---|---|---|
| Keyword Precision (avg) | ≥ 0.80 | Fraction of expected keywords present in response |
| Pass Rate (precision ≥ 0.5) | ≥ 0.90 | Fraction of questions answered correctly |
| Avg Latency | < 3,000 ms | End-to-end P50 latency |
| Not-found handling | 100% | Out-of-scope questions correctly deflected |
| Citation accuracy | Manual review | LLM cites filename + page from retrieved chunks |

*Exact values logged in MLflow experiment `/Shared/policy_qa_assistant` under run `04_rag_chain_evaluation`.*

---

## Known Biases and Limitations

1. **Corpus dependency:** Answers are only as good as the ingested documents. Outdated policies produce outdated answers.
2. **Language:** Optimized for English. Non-English documents will work but retrieval quality may degrade.
3. **Scanned PDFs:** Image-based PDFs produce poor text extraction. OCR pre-processing required.
4. **Hallucination risk:** Despite grounding prompts and similarity thresholds, LLMs can occasionally generate plausible-sounding but incorrect citations. Always verify critical answers.
5. **LLM biases:** Llama 3.1 inherits biases from its pre-training data. Policy Q&A is low-risk for demographic bias, but responses should be reviewed for any sensitive HR topics.
6. **Chunk boundary effects:** Answers that span chunk boundaries may be incomplete. Overlap (50 tokens) mitigates but does not eliminate this.

---

## Guardrails

| Guardrail | Implementation |
|---|---|
| PII detection (input) | Regex patterns for email, phone, SSN, credit card — redacted before LLM |
| Prompt injection (input) | Regex patterns for common injection phrases — query blocked |
| Input length limit | 2,000 tokens max — longer queries rejected with clear message |
| Similarity threshold | Score < 0.70 → disclaimer added to response |
| PII scan (output) | Same regex patterns applied to LLM response |
| Verbatim reproduction | Chunks > 200 chars reproduced verbatim are replaced with summary notice |
| Not-found response | Explicit "I could not find..." response when answer not in documents |

**Production:** Unity AI Gateway provides additional guardrails: PII guardrail, prompt injection guardrail, hallucination guard (Beta), payload logging, rate limiting.

---

## Human-in-the-Loop

Model outputs are **recommendations only**, not automated decisions. All answers affecting employees or individuals must be verified with HR or the relevant department. This is stated explicitly in the Gradio UI and in the README.

---

## Data Retention

- `qa_interactions` (Gold): retained per workspace data retention policy (default: indefinite Delta table)
- `qa_feedback` (Gold): retained per workspace data retention policy
- Raw documents (Volume): retained until manually deleted
- Recommended production retention: 2 years for interactions, 5 years for source documents

## Right to Erasure

If personal data is present in interactions (e.g., user queries containing PII that was not fully redacted):

```sql
-- Delete interactions for a specific user session
DELETE FROM catalog.schema.qa_interactions
WHERE session_id = '<session_id>';

-- Delete feedback for a specific interaction
DELETE FROM catalog.schema.qa_feedback
WHERE interaction_id = '<interaction_id>';
```

For GDPR / India DPDP Act 2023 compliance, document deletion requests in your DPO system and confirm with `VACUUM` after deletion.
