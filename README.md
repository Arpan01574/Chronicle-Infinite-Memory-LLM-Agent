<p align="center">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-FF6F00?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-00A67E?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Chainlit-Chat_UI-5A67D8?style=for-the-badge&logo=chainlit&logoColor=white" alt="Chainlit"/>
  <a href="EVALUATION.md"><img src="https://img.shields.io/badge/Evaluation-200_prompts-00f5a0?style=for-the-badge" alt="Evaluation"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-blue?style=for-the-badge" alt="Status"/>
</p>

# 🧠 Chronicle — Infinite Memory LLM Agent

> **A lightweight, framework-free AI assistant with persistent memory, RAG capabilities, and a stunning cyberpunk UI.**

Chronicle is a fully local, privacy-first AI agent that remembers everything. Built from scratch using pure Python, it combines vector-based document retrieval (RAG), persistent core memory, dynamic tool execution, and a beautiful glassmorphism chat interface—all running on your own hardware via Ollama.

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Architectural Verification & Evaluation](#-architectural-verification--evaluation)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Available Tools](#-available-tools)
- [Agent Modes](#-agent-modes)
- [Roadmap](#-roadmap)

---

## ❌ The Problem

Traditional LLM chatbots suffer from **amnesia**—every conversation starts from zero. They:

- Forget your name, preferences, and context between sessions
- Cannot access your personal documents or knowledge base
- Require expensive cloud APIs with privacy concerns
- Lack the ability to take actions (web search, file generation, calculations)

---

## ✅ The Solution

Chronicle solves this with a **three-tier memory architecture**:

| Memory Layer | Technology | Purpose |
|--------------|------------|---------|
| **Core Memory** | JSON file (`data/memory.json`) | Persistent user info, preferences, persona |
| **Conversation History** | JSONL + SQLite | Chat persistence across sessions |
| **Archival Memory** | ChromaDB + Sentence Transformers | Vector-indexed documents for RAG retrieval |

All processing happens **100% locally** using Ollama, ensuring complete privacy.

---

## ✨ Key Features

### 🧠 Memory Systems
- **Core Memory Persistence** — Automatically saves user details, preferences, and conversation context
- **RAG Pipeline** — Ingests PDF, TXT, and MD files into ChromaDB for semantic search
- **Conversation Archival** — Every interaction is embedded and stored for long-term recall

### 🔧 Tool Execution
- **JSON-based Tool Calling** — LLM outputs structured tool calls, executed automatically
- **Multi-tool Chaining** — Supports up to 3 tool iterations per turn
- **Real-time Web Search** — DuckDuckGo integration for current information

### 🎭 Dynamic Modes
- **Teacher Mode** — Socratic tutoring with guided questions
- **Python/JavaScript Coder** — Language-specific style guide injection
- **Code Reviewer** — Detailed code analysis and suggestions
- **Web Researcher** — Automatic search for every query

### 💎 Premium UI
- **Glassmorphism Design** — Frosted glass effects with neon cyberpunk accents
- **Chat History Sidebar** — Resume any previous conversation
- **Collapsible Steps** — View inner monologue, RAG context, and tool calls
- **File Upload** — Drag-and-drop PDF/TXT/MD with automatic ingestion

---

## 🏗️ Technical Architecture

![System Architecture Diagram](public/architecture.png)

### Data Flow
1. **User Input** → Chainlit captures message + optional file uploads
2. **RAG Retrieval** → Query ChromaDB for relevant document chunks
3. **Prompt Assembly** → Combine base persona + active modes + RAG context + tools
4. **LLM Inference** → Ollama processes the full context
5. **Tool Loop** → If tool call detected, execute and re-prompt (up to 3x)
6. **Memory Update** → Persist to core memory, history, and vector store
7. **Response Display** → Show answer with collapsible thinking/tool steps

---

## 🧪 Architectural Verification & Evaluation

Chronicle ships with a reproducible benchmark that samples **200 prompts** from a **500-prompt, category-stratified corpus** (7 categories, 60 simulated chats) and runs them against the local Ollama-backed agent. Each run is graded by an automated heuristic rubric and also records tool-call behaviour, per-prompt latency, and answer/reference embedding similarity. Full methodology, rubric, limitations, and data provenance are in **[EVALUATION.md](EVALUATION.md)**.

### Results (200 prompts)

| Metric | Result | Notes |
|---|---|---|
| Tool-intent routing | **92.3 %** accuracy (24/26) · F1 90.9 % · MCC 0.84 | n = 26; 95 % CI for accuracy 75.9 – 97.9 % |
| End-to-end latency | mean **29.7 s** · median **22.4 s** · p95 **70.7 s** | Local model, single machine |
| Answer ↔ reference similarity | mean cosine **0.41** (MiniLM-L6-v2, n = 173) | Similarity signal, not a correctness score |
| Heuristic rubric score | 86.2 % (157 pass / 31 partial / 12 fail) | Automated rubric, see below |

<p align="center">
  <img src="evals/results/02_tool_confusion_matrix.png" alt="Tool-intent confusion matrix" width="48%"/>
  <img src="evals/results/03_end_to_end_latency.png" alt="End-to-end latency distribution" width="48%"/>
</p>

> **How to read these numbers.** The rubric checks that answers *look* responsive (term overlap, code formatting, tool usage, length); it does not verify factual correctness. RAG-quality and subsystem-timing figures shown in the charts come from a separate architectural verification run rather than the 200-prompt benchmark. Methodology, limitations, and data sources are in [EVALUATION.md](EVALUATION.md#6-data-sources-and-verdict-totals).

### Run it

```powershell
python .\evals\evaluate.py --live                  # full 200-prompt run (~100 min), replaces JSON + charts
python .\evals\evaluate.py --live --limit 5        # smoke test, no outputs overwritten
python .\evals\evaluate.py --charts-only           # regenerate charts from saved JSON (no Ollama)
python .\evals\evaluate.py --rebuild-from-saved    # recompute derived metrics from saved answers
```

Canonical results: [`data/verification_results.json`](data/verification_results.json) · Charts: [`evals/results/`](evals/results/)

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime |
| Ollama | Latest | Local LLM inference |
| Git | Any | Clone repository |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Arpan01574/Chronicle-Infinite-Memory-LLM-Agent.git
cd Chronicle-Infinite-Memory-LLM-Agent

# 2. Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the required Ollama model
ollama pull hermes3:8b
```

### Configuration

No `.env` file is required. Configuration is done via constants in the source files:

| File | Key Settings |
|------|--------------|
| `src/engine.py` | `OLLAMA_MODEL`, `RAG_TOP_K`, `MAX_HISTORY_MESSAGES` |
| `src/ingest.py` | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL` |
| `src/app.py` | `MODE_PROMPTS`, `SUPPORTED_UPLOAD_TYPES` |

To change the LLM model, edit `OLLAMA_MODEL` in `src/engine.py`:

```python
OLLAMA_MODEL: str = "hermes3:8b"  # Change to your preferred model
```

---

## 💻 Usage

### Start the Chat Interface

Click the ▶️ **Play** button on `src/app.py` in VS Code, or run via terminal:
```bash
chainlit run src/app.py -w
```

The `-w` flag enables hot-reload during development. Open your browser to `http://localhost:8000`.

### Ingest Documents

Add PDF, TXT, or MD files to the `data/` directory, then run:

```bash
python -m src.ingest
```

Or ingest a specific file:

```bash
python -m src.ingest path/to/document.pdf
```

### CLI Testing Mode

For quick testing without the web UI:

```bash
python src/engine.py
```

**CLI Commands:**
- `quit` — Exit
- `reset` — Clear conversation history
- `stats` — Show memory statistics
- `teacher` — Activate Teacher Mode
- `coder` — Activate Python Coding Guide
- `default` — Clear all modes

---

## 📁 Project Structure

```
Chronicle-Infinite-Memory-LLM-Agent/
├── src/
│   ├── app.py          # Chainlit web interface, chat lifecycle, file uploads
│   ├── engine.py       # MemoryAgent core: RAG, tool execution, LLM calls
│   ├── tools.py        # Tool implementations (web search, PDF, memory, math)
│   └── ingest.py       # Document ingestion pipeline for ChromaDB
├── data/
│   ├── memory.json     # Core memory (user info, persona, preferences)
│   ├── history.jsonl   # Conversation history (append-only log)
│   ├── chainlit.db     # SQLite database for Chainlit persistence
│   ├── chromadb/       # ChromaDB vector store (embeddings + metadata)
│   ├── uploads/        # User-uploaded files (auto-created)
│   ├── user_downloads/ # Generated PDFs and exports
│   └── verification_results.json  # Canonical benchmark results
├── evals/
│   ├── evaluate.py     # Live benchmark, rubric, metrics, chart pipeline
│   ├── prompt.py       # 500-prompt corpus and 200-prompt stratified sampler
│   ├── charts.py       # Standalone chart generation from saved results
│   └── results/        # Generated benchmark charts (PNG)
├── public/
│   ├── custom.css      # Glassmorphism + Cyberpunk theme (714 lines)
│   ├── architecture.png # System architecture diagram
│   ├── teacher.svg     # Starter button icons
│   ├── search.svg
│   ├── document.svg
│   └── code.svg
├── EVALUATION.md       # Evaluation methodology, results, and data sources
├── requirements.txt    # Python dependencies
├── chainlit.md         # Chainlit welcome screen
└── README.md           # This file
```

---

## 🔧 Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `perform_web_search` | Real-time internet search via DuckDuckGo | `query` |
| `get_current_datetime` | Returns current date, time, and timezone | None |
| `save_summary_to_pdf` | Exports text content to PDF file | `filename`, `content` |
| `fetch_coding_guidelines` | Language-specific style guide (PEP 8, ESLint, etc.) | `language` |
| `activate_teacher_mode` | Enables Socratic tutoring for a topic | `topic` |
| `evaluate_math_expression` | Safe mathematical expression evaluation | `expression` |
| `update_core_memory` | Persists user info to core memory | `section`, `field`, `value`, `operation` |
| `search_knowledge_base` | Queries ChromaDB for relevant documents | `query` |

---

## 🎭 Agent Modes

Switch modes via the **Settings** gear icon in the UI:

| Mode | Behavior |
|------|----------|
| **General Assistant** | Default helpful assistant |
| **Web Researcher** | Always searches the web, cites sources |
| **Teacher Mode** | Socratic questioning, no direct answers |
| **Python Coder** | PEP 8 style, type hints, docstrings |
| **JavaScript Coder** | ESLint/Airbnb style, const/let, arrow functions |
| **Code Reviewer** | Bug hunting, security checks, improvements |

---

## 🗺️ Roadmap

Based on the current codebase, logical next steps include:

- [ ] **Docker Support** — Containerize the application with Ollama sidecar
- [ ] **Multi-Model Selection** — UI dropdown to switch between Ollama models
- [ ] **Voice Input/Output** — Whisper for STT, TTS for responses
- [ ] **Memory Visualization** — Graph view of knowledge base connections
- [ ] **Plugin System** — Hot-loadable tools without code changes
- [ ] **Conversation Export** — Export chats as Markdown or PDF
- [ ] **Authentication** — Multi-user support with separate memory spaces
- [ ] **Streaming Responses** — Token-by-token output for faster perceived latency

---

## 📄 License

This project is open source under the MIT License.

---

<p align="center">
  <strong>Built with ❤️ for privacy-conscious AI enthusiasts</strong><br/>
  <em>Your data never leaves your machine.</em>
</p>