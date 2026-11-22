# Corpus Sources — RAG-BenchLab

**Domain:** LLM systems, Retrieval-Augmented Generation (RAG), evaluation and agents.
**Goal:** exercise factual precision, multi-passage synthesis, relational reasoning (graph) and abstention.

## Licensing rules
- Only **CC BY 4.0** documents are allowed. CC BY-SA, CC BY-ND and the default arXiv non-exclusive license are excluded.
- Every license below was read on the arXiv abstract page on **2026-11-21**. `scripts/download_corpus.py` re-checks it before each download and refuses anything that is not CC BY 4.0.
- PDFs go in `corpus/raw/` (git-ignored). Only this list is versioned.
- The specification mentions "public reports"; every report considered had a non-free license, so three table-heavy CC BY benchmark papers (section 2) take their place.

The download script parses the rows below: keep the format `| ID | title | `arXiv id` | ... |`.

## 1. Papers (arXiv, CC BY 4.0)

### Track A — RAG architectures, retrieval, graph
| ID | Title | arXiv ID | License | Role |
| :-- | :-- | :-- | :-- | :-- |
| A01 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | `2310.11511` | CC BY 4.0 | Reflection tokens, PopQA, PubHealth |
| A02 | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | `2401.18059` | CC BY 4.0 | Hierarchical summaries |
| A03 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | `2404.16130` | CC BY 4.0 | GraphRAG, communities, global queries |
| A04 | GRAG: Graph Retrieval-Augmented Generation | `2405.16506` | CC BY 4.0 | Graph retrieval |
| A05 | Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation | `2410.20724` | CC BY 4.0 | KG-based RAG |
| A06 | Domain-specific Question Answering with Hybrid Search | `2412.03736` | CC BY 4.0 | Hybrid search (P2) |
| A07 | How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models | `2508.16757` | CC BY 4.0 | Rerankers, comparison tables (P2) |
| A08 | In Defense of RAG in the Era of Long-Context Language Models | `2409.01666` | CC BY 4.0 | RAG vs long context |
| A09 | Long Context vs. RAG for LLMs: An Evaluation and Revisits | `2501.01880` | CC BY 4.0 | RAG vs long context |

### Track B — Evaluation, faithfulness, abstention
| ID | Title | arXiv ID | License | Role |
| :-- | :-- | :-- | :-- | :-- |
| B01 | Ragas: Automated Evaluation of Retrieval Augmented Generation | `2309.15217` | CC BY 4.0 | Faithfulness, answer relevance, context precision |
| B02 | ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | `2311.09476` | CC BY 4.0 | Synthetic judges, PPI |
| B03 | Evaluating Retrieval Quality in Retrieval-Augmented Generation | `2404.13781` | CC BY 4.0 | Retrieval evaluation |
| B04 | DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge | `2610.28340` | CC BY 4.0 | Abstention (no-answer questions) |

### Track C — Agents, planning, tools
| ID | Title | arXiv ID | License | Role |
| :-- | :-- | :-- | :-- | :-- |
| C01 | ReAct: Synergizing Reasoning and Acting in Language Models | `2210.03629` | CC BY 4.0 | Thought-action-observation loop |
| C02 | Reflexion: Language Agents with Verbal Reinforcement Learning | `2303.11366` | CC BY 4.0 | Self-reflection |
| C03 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | `2305.10601` | CC BY 4.0 | Search over thoughts |
| C04 | Gorilla: Large Language Model Connected with Massive APIs | `2305.15334` | CC BY 4.0 | Tool/API use |
| C05 | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | `2305.04091` | CC BY 4.0 | Task decomposition |
| C06 | Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models | `2411.19443` | CC BY 4.0 | Agentic RAG (P4) |
| C07 | Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems | `2505.22571` | CC BY 4.0 | Agentic RAG (P4) |

## 2. Table-heavy benchmark papers (CC BY 4.0)
| ID | Title | arXiv ID | License | Role |
| :-- | :-- | :-- | :-- | :-- |
| R01 | MMTEB: Massive Multilingual Text Embedding Benchmark | `2502.13595` | CC BY 4.0 | Large embedding comparison tables |
| R02 | T^2-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation | `2506.12071` | CC BY 4.0 | Text + table RAG |
| R03 | Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? | `2406.13121` | CC BY 4.0 | RAG vs long-context tables |

## 3. Excluded after license check
Not CC BY 4.0, so not usable: RAG (Lewis et al.), DPR, ColBERT, HyDE, CRAG, MT-Bench, Lost in the Middle, FactScore, RGB, CRUD-RAG, Toolformer, Agentic RAG survey (2501.09136), MTEB (2210.07316), LongBench (all arXiv non-exclusive license); BEIR (CC BY-SA 4.0).
