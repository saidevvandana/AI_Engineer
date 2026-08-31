# AI Presales Agent — SOW Generator

Structured deal data → retrieval-grounded knowledge → a structured Statement
of Work (SOW), built as a multi-step LangGraph workflow with a FastAPI
backend and a Vue3 frontend.

- **Backend**: Python, FastAPI, LangGraph + LangChain, FAISS
- **Frontend**: Vue3 (Vite)
- **LLM**: Anthropic Claude or OpenAI (pluggable via env var)
- **Embeddings**: local `sentence-transformers` by default (no API key needed
  to run retrieval), or OpenAI embeddings if preferred

---

## 1. Architecture

```
┌─────────────┐      HTTP/JSON      ┌────────────────────────────────────────┐
│  Vue3 UI    │◀───────────────────▶│              FastAPI app                │
│ (frontend/) │  /api/sow/generate   │  app/api/routes.py                      │
│             │  /api/sow/compare    │                                          │
└─────────────┘                     │   ┌──────────────────────────────────┐   │
                                     │   │   LangGraph workflow (compiled)  │   │
                                     │   │   app/workflow/graph.py          │   │
                                     │   │                                  │   │
                                     │   │  normalize_input                 │   │
                                     │   │       │                          │   │
                                     │   │  plan_queries                    │   │
                                     │   │       │                          │   │
                                     │   │  [use_rag?]──▶ retrieve_knowledge│   │
                                     │   │       │              │           │   │
                                     │   │       └─▶ skip_retrieval          │   │
                                     │   │                     │            │   │
                                     │   │            generate_sections     │   │
                                     │   │                     │            │   │
                                     │   │           validate_and_refine    │   │
                                     │   │                     │            │   │
                                     │   │            assemble_sources      │   │
                                     │   └──────────────────────────────────┘   │
                                     │         │                    │            │
                                     │   ┌─────▼─────┐      ┌───────▼───────┐   │
                                     │   │  FAISS     │      │  Chat model    │   │
                                     │   │  vector    │      │  (Claude /     │   │
                                     │   │  store     │      │   OpenAI)      │   │
                                     │   └─────┬──────┘      └────────────────┘   │
                                     │         │                                  │
                                     │   knowledge/data/*.md (chunked at startup) │
                                     └────────────────────────────────────────────┘
```

Directory layout:

```
backend/
  app/
    config.py            # single source of settings (env vars)
    models/schemas.py     # Pydantic request/response contracts
    knowledge/
      data/*.md            # the knowledge base (5 source docs)
      loader.py             # markdown-aware chunking
      embeddings.py          # embeddings factory (local / OpenAI)
      vectorstore.py          # FAISS build/load/search
    workflow/
      state.py              # typed LangGraph state
      prompts.py             # per-section prompt templates
      nodes.py               # one function per pipeline step
      graph.py               # StateGraph wiring + conditional RAG branch
    services/
      llm.py                 # chat model factory (Anthropic / OpenAI)
      sow_generator.py        # graph.invoke() -> typed SOWResult
    api/routes.py           # FastAPI endpoints
    main.py                # app + CORS
  tests/                  # unit tests (no API key required)
scripts/
  demo_with_without_rag.py # WITH vs WITHOUT RAG validation report
frontend/
  src/
    components/            # DealForm, SOWOutput, SourcesPanel, ComparisonView
    api/client.js           # fetch wrapper
```

## 2. End-to-end flow

1. User fills in deal data in the Vue3 form (or loads the clean/messy sample)
   and clicks **Generate SOW** (or **Compare**).
2. The frontend POSTs the raw deal JSON to `/api/sow/generate`.
3. `sow_generator.generate()` invokes the compiled LangGraph graph with that
   JSON as `raw_input`.
4. The graph runs its nodes in order (see §4 below), calling out to the FAISS
   retriever and the chat model as needed.
5. The final graph state is converted into a typed `SOWResult` (sections,
   sources used, which fields were inferred, validation notes) and returned
   as JSON.
6. The frontend renders the five SOW sections, the inferred-fields panel, the
   validation log, and the retrieved-sources panel (with per-chunk relevance
   score and originating file/section — the citations).

## 3. Running it

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # (or `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env        # then fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
python run.py                 # serves http://localhost:8000, docs at /docs
```

On first run, the FAISS index is built automatically from
`app/knowledge/data/*.md` and persisted to `app/knowledge/index/`. Delete
that folder (or call `POST /api/knowledge/reindex`) after editing the
knowledge base to rebuild it.

Run the unit tests (these do **not** require an LLM API key — they cover
input normalization, chunking, retrieval relevance, and graph structure):

```bash
pytest
```

### Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173, proxies /api to :8000
```

### WITH vs WITHOUT RAG validation report

Requires a real LLM key configured in `backend/.env` (this makes live model
calls):

```bash
cd backend
python ../scripts/demo_with_without_rag.py
```

This writes `docs/validation_with_without_rag.md`, running the *identical*
pipeline for both the clean and messy sample inputs, once with retrieval and
once without, so the two outputs are directly comparable. The same
comparison is also available live in the UI via **Compare With / Without
RAG**, and via `POST /api/sow/compare`.

---

## 4. RAG Implementation

**This is a hard requirement of the assessment, so it's worth being explicit
about what "real" means here**: the SOW sections are generated by a prompt
that is only handed retrieved chunks as its "Reference material" — the model
is instructed to ground its answer in that material, and the
`skip_retrieval` branch (used for the without-RAG comparison) passes an
empty context block with the literal string `(no reference material
retrieved)`, so there's no way for retrieval to be a no-op that the LLM
ignores in favor of parametric knowledge alone — you can see the difference
directly in `docs/validation_with_without_rag.md` or the Compare view.

### Knowledge base
Five markdown documents under `backend/app/knowledge/data/`:
`sow_templates.md`, `cloud_migration_best_practices.md`, `hipaa_compliance.md`,
`delivery_frameworks.md`, `modernization_and_estimation_guidelines.md`. These
cover exactly the four categories the assessment asks for (SOW templates,
migration best practices, compliance guidelines, delivery frameworks), plus a
fifth doc specifically written to ground the *messy-input* path (generic
"Modernization" project types, unknown budget/timeline defaulting guidance) —
this exists because a RAG demo that only retrieves well for the one clean
sample input isn't actually demonstrating retrieval.

### Chunking strategy
Two-stage split (`app/knowledge/loader.py`):
1. **`MarkdownHeaderTextSplitter`** on `#`/`##` headers first, so a chunk
   never straddles two unrelated topics (e.g. HIPAA "Technical Safeguards"
   won't get merged with "Administrative Safeguards" at an arbitrary
   character boundary). Each resulting piece keeps a `section` metadata path
   like `HIPAA Compliance Guidelines for Cloud Engagements > Technical
   Safeguards (Security Rule)`.
2. **`RecursiveCharacterTextSplitter`** (chunk_size=900, chunk_overlap=150)
   within each header section, so no single chunk is too large to be a
   precise retrieval unit, while the overlap keeps sentences that straddle a
   split boundary intelligible on both sides.

Every chunk carries `source` (filename) and `section` (header path) metadata,
which is what the frontend's "Retrieved Sources" panel displays as citations.

### Retrieval method
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` by default
  (via `langchain-huggingface`) — chosen so the RAG pipeline works with zero
  API cost/key regardless of which LLM provider is configured. Swappable to
  OpenAI embeddings via `EMBEDDINGS_PROVIDER=openai`.
- **Store**: FAISS (`langchain_community.vectorstores.FAISS`), built once at
  first use and persisted to disk (`app/knowledge/index/`).
- **Query construction**: retrieval is **per-section**, not one global query
  for the whole SOW. `plan_queries` (`app/workflow/nodes.py`) builds a
  distinct query per SOW section from the normalized deal fields — e.g. the
  Assumptions query includes "risks and client responsibilities", the
  Timeline query includes the requested duration, and every query gets a
  `HIPAA compliance safeguards` hint appended when the industry matches
  `health.*`. This is deliberate: a single generic query ("cloud migration
  for healthcare") retrieves a much less targeted set of chunks than five
  section-specific queries, and it means the Scope of Work section and the
  Assumptions section are grounded in different, more relevant slices of the
  knowledge base rather than sharing one context blob.
- **Top-k**: `RAG_TOP_K` (default 4) chunks per section query, deduplicated
  across sections before being surfaced as `sources` in the API response.
- **No metadata filtering** is applied (the KB is small enough that pure
  similarity search is precise) — see §6 for what changes at scale.

### How retrieval improves the output — what the evidence actually shows
`docs/validation_with_without_rag.md` is generated output from a real run
(Claude Sonnet), not a hypothetical. Worth being precise about what it shows,
rather than overselling it: with a frontier model, the WITHOUT-RAG baseline
is **not** garbage — Claude already has broad training knowledge of AWS
migrations, SOW structure, and HIPAA, so both columns are competent. The
measurable differences are:

1. **Grounded provenance, not just plausible text.** Every WITH-RAG claim
   traces to a specific retrieved chunk (visible in the `sources` list with
   file + section + similarity score); the WITHOUT-RAG version is equally
   fluent but *unattributable* — you cannot tell a client "this deliverable
   comes from our documented methodology" versus "the model guessed
   something reasonable." For a presales tool that's the difference between
   a defensible SOW and a plausible-sounding one.
2. **House terminology and structure, not the model's own.** WITH-RAG scope
   sections use this org's exact phase names (`Assess / Mobilize / Migrate &
   Modernize / Operate & Optimize`, straight from `cloud_migration_best_practices.md`)
   and exact deliverable names (`HIPAA Security Risk Assessment Report`,
   `Compliance Gap Remediation Plan` — named verbatim in `hipaa_compliance.md`).
   WITHOUT-RAG invents its own structurally-similar but differently-named
   workstreams (e.g. "Target Architecture and Roadmap Design" instead of
   "Mobilize") — plausible, but not this org's methodology, and inconsistent
   from run to run.
3. **Where it matters most: information the model can't already know.**
   `hipaa_compliance.md`'s specific guidance to extend timelines 10-20% for
   compliance validation, or `modernization_and_estimation_guidelines.md`'s
   specific default-estimation policy, are house conventions — a general
   LLM has no way to know this org's specific estimation policy or a
   proprietary/updated compliance interpretation without retrieval. That's
   the scenario RAG is actually load-bearing for, more than raw prose
   quality on a well-known domain like "cloud migration."

In short: on a strong model, RAG's win here is *consistency, attribution,
and encoding house-specific knowledge* rather than turning bad output good —
and that's the honest claim to make, not "WITHOUT-RAG is noticeably worse."
See `docs/validation_with_without_rag.md` for the full generated text side by
side.

---

## 5. Agent Workflow (LangGraph)

The pipeline is a `langgraph.graph.StateGraph` over a typed `SOWState`
(`app/workflow/state.py`), not a single prompt:

| Node | Responsibility |
|---|---|
| `normalize_input` | Parses the raw (possibly messy) JSON into `DealInput`, infers a default for every missing/placeholder field ("Unknown", `null`, empty), and records **why** each default was chosen. |
| `plan_queries` | Builds one retrieval query per SOW section from the normalized deal. |
| `retrieve_knowledge` / `skip_retrieval` | Conditional branch on `state["use_rag"]` — the only place RAG is switched on/off, which is what makes the with/without comparison a clean A/B rather than two different code paths. |
| `generate_sections` | Calls the LLM once per SOW section, each with its own instructions and its own retrieved context — not one "write the whole SOW" prompt. |
| `validate_and_refine` | Runs cheap heuristic checks per section (too short; healthcare deal but Assumptions doesn't mention HIPAA; Scope of Work missing an Out-of-Scope boundary; Deliverables not actually a list) and re-prompts the LLM to fix only the sections that fail, with the specific problem named. |
| `assemble_sources` | Dedupes and ranks the retrieved chunks into the citation list returned to the UI. |

### Why structured this way
- **Section-level generation + validation** means a problem in one section
  (e.g. Assumptions missing HIPAA) triggers a targeted, cheap re-prompt of
  just that section — not a full-document regeneration, and not a silent
  pass-through of a bad section.
- **A conditional edge, not an `if` inside one function**, for the RAG
  toggle keeps the graph the unit of orchestration truth — inspecting
  `graph.get_graph()` shows the two branches explicitly, and it's the
  natural point to later add e.g. a filtered-retrieval branch or a
  human-in-the-loop interrupt without touching generation code.
- **Typed `SOWState`, single-purpose node functions** (`state -> partial
  update`) mirror exactly how this would sit in a larger LangGraph
  deployment: each node here is already a pure function suitable for
  `add_node`, the state is already the shape a checkpointer would persist,
  and `route_after_queries` is already the shape of a `add_conditional_edges`
  router. Dropping this graph into an existing LangGraph/LangChain
  orchestration layer means: reusing this `StateGraph` as a subgraph (e.g.
  invoked from a larger "opportunity intake" graph), swapping `get_llm()` /
  `get_embeddings()` for the platform's existing model-routing layer, and
  replacing the in-process FAISS store with whatever retriever the platform
  already exposes as a `VectorStoreRetriever` — none of `nodes.py` would need
  to change, since it only depends on the `similarity_search()` function
  signature, not on FAISS specifically.

### Validation requirement: WITH vs WITHOUT RAG
Both `POST /api/sow/generate` (with `use_rag: false`) and
`POST /api/sow/compare` (which runs both branches back-to-back on the same
input) are available; the frontend's **Compare With / Without RAG** button
uses the latter and renders both outputs side by side per section. See §4
for what the difference looks like and why.

---

## 6. Tradeoffs

### What was simplified
- **In-memory/local FAISS index**, rebuilt on first import and cached with
  `functools.lru_cache` — fine for a five-document knowledge base and a
  single-process demo; not a multi-tenant or horizontally-scaled store.
- **Heuristic validation** (`_validate_section`) rather than an LLM-as-judge
  or a second model call to score groundedness — cheap and fast, but it
  checks for specific known failure patterns rather than general quality.
- **No persistence of generated SOWs** — every request regenerates from
  scratch; there's no deal history, no versioning, no "edit and regenerate
  section 3 only" workflow.
- **No auth / multi-tenancy** — single shared backend, single shared
  knowledge base, no per-user or per-org data isolation.
- **Synchronous request/response** — `/api/sow/generate` blocks for the
  duration of ~5-10 LLM calls (5 generation + up to 5 refinement); there's no
  streaming or background job with a job-status endpoint.
- **A small, hand-written knowledge base** (5 docs) rather than ingesting a
  real corpus of past SOWs, contracts, and compliance documents.

### What would break at scale
See the final question below — this is really the same list, restated for
"thousands of deals":

1. **The FAISS index is process-local and rebuilt in-process.** At scale you
   need a managed/shared vector store (e.g. pgvector, Pinecone, Weaviate,
   OpenSearch) so multiple API instances share one index, support incremental
   upserts as the knowledge base grows, and don't each pay a cold-start
   embedding cost.
2. **Synchronous generation won't hold up under concurrent load.** ~5-10
   sequential LLM calls per SOW (worse with refinement retries) means a
   single request can take 30-90s; at thousands of concurrent deals this
   needs to become an async job queue (e.g. Celery/RQ or a LangGraph
   checkpointer-backed run you can poll/resume) with the UI polling or
   subscribing for status, plus parallelizing the independent section-
   generation calls (they don't depend on each other today and are only
   sequential because of a simple `for` loop in `generate_sections`).
3. **No caching of embeddings or repeated retrieval queries** — at scale,
   caching embeddings for repeated/similar deal shapes (e.g. many "Cloud
   Migration + Healthcare" deals) would cut both cost and latency.
4. **Heuristic validation doesn't catch semantic errors** (a factually wrong
   but well-formatted section passes). At scale, worth adding an LLM-judge
   validation pass and/or human review gating before a SOW is sent to a
   client — this is a natural `interrupt()` point in LangGraph.
5. **No rate limiting / cost controls per tenant** — thousands of deals from
   many users needs per-org quotas and model-cost tracking, not a single
   shared API key.
6. **No observability** — at scale you need per-node tracing (LangSmith or
   equivalent) to see which node/section is slow, failing validation
   repeatedly, or retrieving poor-quality chunks, rather than reading logs.
7. **Single hand-maintained knowledge base file set** doesn't scale to a real
   enterprise corpus — needs an ingestion pipeline (dedup, freshness/version
   tracking, access control per document, incremental re-embedding on
   change) rather than "delete the index folder and restart".

---

## 7. "If this had to run at enterprise scale (thousands of deals), what would break?"

In priority order:

1. **Retrieval infrastructure** — swap process-local FAISS for a managed
   vector store with proper indexing, sharding, and incremental updates;
   separate the embedding/ingestion pipeline from the request path entirely
   (a background indexer, not "build on first import").
2. **Latency & throughput** — move SOW generation to an async job (LangGraph
   supports this natively via checkpointers + `interrupt`/resume), parallelize
   the five independent section-generation calls, and add response streaming
   so the UI can show sections as they complete instead of blocking on the
   whole graph.
3. **Cost** — thousands of deals × ~5-10 LLM calls each adds up fast; add
   prompt caching (static system/reference-doc content), a cheaper model for
   the validation/refine pass, and per-tenant budget controls.
4. **Reliability** — add retries with backoff on LLM/embedding calls,
   circuit-breaking if a provider degrades, and a fallback provider (this
   codebase already isolates the LLM behind `get_llm()` for exactly this
   reason).
5. **Quality at scale** — heuristic validation catches known failure shapes
   but not novel ones; add LLM-judge scoring, sampling-based human review,
   and feedback loops (accepted/edited SOWs feeding back into few-shot
   examples or fine-tuning signal).
6. **Multi-tenancy & governance** — per-org knowledge bases (a healthcare
   client's SOWs shouldn't retrieve a competitor's), auth, audit logging of
   what was generated/retrieved for compliance purposes, and versioned
   knowledge base documents so retrieval results are reproducible/explainable
   after the fact.
7. **Observability** — distributed tracing per LangGraph node (LangSmith or
   OpenTelemetry), retrieval-quality metrics (are top-k chunks actually being
   used in the final text?), and alerting on validation failure rates as an
   early signal the knowledge base or prompts have drifted from reality.

---

## 8. Sample inputs used throughout

**Clean:**
```json
{
  "client_name": "Acme Health",
  "industry": "Healthcare",
  "project_type": "Cloud Migration",
  "objectives": ["Migrate legacy systems to AWS", "Improve scalability", "Ensure HIPAA compliance"],
  "timeline": "6 months",
  "budget_range": "$250k-$500k"
}
```

**Messy:**
```json
{
  "client_name": "Unknown",
  "project_type": "Modernization",
  "objectives": ["Improve systems"],
  "timeline": null
}
```

Both are available as one-click examples in the UI, and both are covered by
`backend/tests/test_input_parsing.py` and `scripts/demo_with_without_rag.py`.
