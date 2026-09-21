# RestoRec 🍽️

RestoRec is a retrieval-augmented generation (RAG) restaurant recommendation system focused on Croydon and surrounding areas of South London. It combines restaurant information collected from TikTok, Reddit and Google Places-style data to provide grounded restaurant recommendations through a Streamlit interface.

Rather than relying only on the general knowledge of a large language model, RestoRec retrieves relevant records from a local knowledge base before generating an answer. This helps the application produce recommendations based on the supplied datasets.

The project also contains a retrieval evaluation framework that compares dense retrieval, BM25 lexical retrieval and hybrid retrieval using manually labelled relevance judgements.

## Project overview

Restaurant information is distributed across several platforms. Google Places provides structured details such as addresses, ratings and restaurant categories, while TikTok and Reddit contain informal recommendations, opinions and descriptions of dining experiences.

RestoRec combines these sources into a single searchable knowledge base.

The application follows a traditional RAG pipeline:

1. Restaurant records are loaded from CSV files.
2. Each record is converted into a LangChain `Document`.
3. Text is encoded using the `all-MiniLM-L6-v2` sentence-transformer model.
4. Embeddings are stored in a local FAISS vector index.
5. A user enters a restaurant-related question through Streamlit.
6. The most relevant documents are retrieved from the knowledge base.
7. The retrieved context is supplied to Google Gemini.
8. Gemini generates an answer grounded in the retrieved records.

## Main features

- Combines TikTok, Reddit and Places restaurant data.
- Uses semantic embeddings to retrieve contextually relevant records.
- Stores embeddings locally using FAISS.
- Generates grounded answers with Google Gemini.
- Displays the retrieved documents used to construct an answer.
- Applies source limits to provide results from different data sources.
- Supports dense, BM25 and hybrid retrieval evaluation.
- Evaluates retrieval using Precision@8, Recall@8, MRR and nDCG@8.
- Measures average retrieval latency.
- Includes automated unit, integration and behavioural tests.
- Uses manually verified relevance labels for the evaluation questions.

## System architecture

```mermaid
flowchart TD
    A["TikTok CSV"] --> D["Document loading"]
    B["Reddit CSV"] --> D
    C["Places CSV"] --> D

    D --> E["Text representation"]
    E --> F["MiniLM embeddings"]
    F --> G["FAISS index"]

    H["User question"] --> I["Dense retrieval"]
    G --> I
    I --> J["Retrieved context"]
    J --> K["Google Gemini"]
    K --> L["Grounded recommendation"]
    L --> M["Streamlit interface"]
```

The retrieval evaluation operates separately from answer generation. This makes it possible to compare retrieval methods without the results being affected by the language model.

## Technology stack

| Technology | Purpose |
|---|---|
| Python | Core application and evaluation language |
| Streamlit | Web application interface |
| LangChain | Document, retrieval and RAG pipeline components |
| Sentence Transformers | Creation of semantic document and query embeddings |
| `all-MiniLM-L6-v2` | Sentence-embedding model |
| FAISS | Local vector similarity search |
| BM25 | Lexical keyword-based retrieval baseline |
| Reciprocal Rank Fusion | Combination of dense and BM25 rankings |
| Google Gemini | Context-grounded answer generation |
| Python CSV module | Reading and processing the TikTok, Reddit and Places datasets |
| pytest | Automated testing |
| python-dotenv | Loading API credentials from `.env` |

## Project structure

```text
RestoRec/
├── data/
│   ├── tiktok.csv
│   ├── reddit.csv
│   └── places.csv
│
├── evaluation/
│   ├── evaluation.py
│   ├── labelling_output.txt
│   └── retrieval_evaluation_results.csv
│
├── FAISS_index/
│   └── Generated local vector index
│
├── src/
│   ├── main.py
│   └── rag.py
│
├── tests/
│   ├── test_embedding_and_faiss.py
│   ├── test_functional_smoke.py
│   ├── test_llm_context.py
│   ├── test_query_behaviour.py
│   └── test_rag.py
│
├── .env
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

### Directory descriptions

#### `data/`

Contains the three source datasets:

- `tiktok.csv` contains TikTok captions, hashtags, engagement information and associated place details.
- `reddit.csv` contains restaurant-related Reddit posts and metadata.
- `places.csv` contains structured restaurant details such as name, category, address, rating, atmosphere and dietary options.

#### `src/`

Contains the main application code:

- `main.py` defines the Streamlit interface and calls the RAG functions.
- `rag.py` loads the datasets, constructs documents, creates or loads the FAISS index, retrieves relevant context and communicates with Gemini.

#### `evaluation/`

Contains the retrieval evaluation framework:

- `evaluation.py` defines the evaluation questions, relevance labels, retrieval methods and metric calculations.
- `labelling_output.txt` records candidate documents used during manual relevance assessment.
- `retrieval_evaluation_results.csv` contains the final results for dense, BM25 and hybrid retrieval.

#### `tests/`

Contains automated tests covering document loading, embeddings, FAISS retrieval, query behaviour, context construction and basic application functionality.

#### `FAISS_index/`

Contains the locally generated vector database. This directory is generated by the application and should normally be excluded from Git because it can be recreated from the source datasets.

## Installation

### 1. Clone the repository

```powershell
git clone <https://github.com/MB-C0des/RestoRec>
cd RestoRec
```


### 2. Create a virtual environment

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

When the environment is active, `(.venv)` should appear at the beginning of the PowerShell prompt.

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies include:

```text
streamlit
python-dotenv
langchain
langchain-community
langchain-google-genai
langchain-huggingface
sentence-transformers
faiss-cpu
rank-bm25
pytest
```

The exact installed dependencies should be maintained in `requirements.txt`.

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

The `.env` file contains a secret and must not be committed to Git. Confirm that `.gitignore` contains:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
FAISS_index/
```

### 5. Add the datasets

Place the source files inside the `data` directory:

```text
data/
├── tiktok.csv
├── reddit.csv
└── places.csv
```

The filenames must match those expected by `src/rag.py`. The application will raise a `FileNotFoundError` if a required file cannot be found.

## Running the application

Run Streamlit from the project root:

```powershell
streamlit run .\src\main.py
```

Streamlit should open the application in a browser. If it does not open automatically, use the local address shown in the terminal, normally:

```text
http://localhost:8501
```

Create the knowledge base before submitting a question. During initial creation, the embedding model may need to be downloaded and all documents must be encoded. Therefore, the first execution may take longer than later runs.

Example questions include:

```text
Where can I find Indian food in Croydon?
```

```text
Where can I get sushi in Purley?
```

```text
Which restaurants have vegan options near East Croydon station?
```

The application can also display the retrieved records so that the evidence supplied to Gemini can be inspected.

## Retrieval implementation

### Dense retrieval

Dense retrieval converts both the query and documents into numerical embedding vectors. FAISS compares the query vector with the stored document vectors and returns the nearest results.

The system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Dense retrieval can identify semantic similarity even when the query and document do not contain exactly the same words.

### BM25 retrieval

BM25 is used as the lexical baseline in the evaluation. It ranks documents according to shared query terms while accounting for term frequency, document frequency and document length.

BM25 is computationally efficient, but it depends more heavily on exact vocabulary overlap.

### Hybrid retrieval

The hybrid method combines dense and BM25 rankings using weighted Reciprocal Rank Fusion. The evaluation configuration gives greater weight to dense retrieval:

```text
Dense weight: 0.7
BM25 weight: 0.3
RRF constant: 60
```

The hybrid approach was included to test whether lexical evidence could improve semantic retrieval.

### Source limits

The retrieval pipeline applies source limits so that one large dataset does not completely dominate the returned context:

```python
SOURCE_LIMITS = {
    "TikTok": 3,
    "Reddit": 2,
    "Places": 3,
}
```

This produces a maximum of eight retrieved documents while preserving representation from the available sources.

## Retrieval evaluation

Run the evaluation from the project root:

```powershell
python -u .\evaluation\evaluation.py
```

To display the results and save the complete terminal output:

```powershell
python -u .\evaluation\evaluation.py 2>&1 |
    Tee-Object -FilePath .\evaluation\evaluation_output.txt
```

If a dependency warning needs to be suppressed:

```powershell
python -W ignore::DeprecationWarning -u .\evaluation\evaluation.py 2>&1 |
    Tee-Object -FilePath .\evaluation\evaluation_output.txt
```

### Manual labelling mode

The evaluation questions contain sets of manually verified relevant document IDs. During benchmark construction, labelling mode can be enabled to print candidate documents:

```python
LABELLING_MODE = True
```

After the relevance labels have been added, change it to:

```python
LABELLING_MODE = False
```

The evaluation will then run the three retrieval methods and calculate their metrics.

### Evaluation metrics

- **Precision@8** measures how many of the eight retrieved documents are relevant.
- **Recall@8** measures how many of all known relevant documents are retrieved.
- **MRR** measures the rank of the first relevant document.
- **nDCG@8** rewards relevant documents that appear nearer the top of the ranking.
- **Time (ms)** measures average retrieval latency.

### Current results

| Method | Precision@8 | Recall@8 | MRR | nDCG@8 | Time (ms) |
|---|---:|---:|---:|---:|---:|
| Dense | 0.225 | 0.273 | 0.263 | 0.258 | 75.56 |
| BM25 | 0.119 | 0.185 | 0.136 | 0.153 | 8.42 |
| Hybrid | 0.206 | 0.263 | 0.249 | 0.237 | 96.57 |

Dense retrieval achieved the highest score for all four effectiveness metrics. BM25 was considerably faster but produced the weakest relevance scores. Hybrid retrieval improved on BM25 but did not outperform dense retrieval and introduced additional latency.

Based on these results, dense retrieval is the preferred method for the current RestoRec application.

## Running the tests

Run the complete test suite from the project root:

```powershell
python -m pytest
```

For more detailed output:

```powershell
python -m pytest -v
```

Run a specific test file:

```powershell
python -m pytest .\tests\test_rag.py -v
```

The tests cover:

- CSV loading and document construction.
- Embedding generation.
- FAISS index creation and retrieval.
- Source metadata.
- Query-specific retrieval behaviour.
- LLM context construction.
- Functional smoke testing.

## Limitations

- The system is geographically focused on Croydon and nearby areas.
- The datasets may not represent every local restaurant.
- Social-media records may contain informal, incomplete or outdated information.
- Some restaurant records may be duplicated across sources.
- Restaurant status, menus and ratings can change after data collection.
- Manually created relevance judgements may contain assessor subjectivity.
- The current retrieval evaluation contains 20 questions, which limits statistical generalisation.
- The application does not currently maintain multi-turn conversational memory.
- Retrieval evaluation measures document relevance, not the factual correctness of the final generated answer.
- The Gemini API requires an internet connection and may be affected by API quotas.

## Future improvements

Possible extensions include:

- Increasing the number and diversity of evaluation questions.
- Using a second assessor to validate relevance labels.
- Reporting evaluation results by query category.
- Adding confidence intervals and statistical significance testing.
- Tuning hybrid retrieval weights on a separate development set.
- Adding reranking with a cross-encoder model.
- Detecting duplicate restaurant records across sources.
- Adding filters for location, price, dietary requirements and rating.
- Supporting multi-turn conversational recommendations.
- Adding automated data-refresh pipelines.
- Evaluating generated answers for faithfulness and recommendation quality.
- Migrating deprecated LangChain integrations to their standalone packages.

## Research context

The evaluation design is influenced by established information-retrieval benchmarks such as TREC and BEIR. The project uses a test collection consisting of queries, documents and manually assigned relevance judgements. The same questions and labels are applied to each retrieval method to support a controlled comparison.

The experiment should be considered BEIR-inspired rather than a direct reproduction of BEIR because RestoRec uses a smaller, domain-specific collection and manually developed restaurant queries.

