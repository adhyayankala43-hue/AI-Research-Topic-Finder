AI Research Topic Discovery System
An end-to-end Machine Learning web application designed to analyze scientific research corpora, surface latent thematic structures, identify interdisciplinary research gaps, evaluate keyword growth velocity, and provide domain-constrained personalized paper recommendations.

Overview
Staying ahead of rapid literature growth across diverse scientific disciplines is challenging. The AI Research Topic Discovery System automates the exploratory review phase of research workflows.

By applying unsupervised Natural Language Processing (NLP) and clustering techniques, the platform processes academic abstracts, identifies semantic clusters, computes directional momentum (growth velocity) for terminology, surfaces unexplored cross-domain research intersections, and ranks literature recommendations based on semantic vector similarity.

Core Features
Latent Topic Extraction: Unsupervised topic discovery surfacing primary semantic topics and their defining keywords across the corpus.


Topic Growth Velocity & Mentions: Dual-axis visualization comparing total term frequency against growth velocity ($\Delta \text{Mentions} / \Delta t$) to differentiate foundational baselines from emerging paradigms.


K-Means Topic Clustering: Automated partitioning of paper vector embeddings into distinct semantic clusters with collapsible abstract inspectors.


Cross-Domain Research Gap Identification: Automated semantic distance evaluation between disparate scientific disciplines to generate novel interdisciplinary hypotheses.


Domain-Constrained Recommendation Engine: Cosine similarity matching mapping user queries directly into the vector space, filtered by domain constraints.


Report Export: Real-time generation of active analytics, identified gaps, and cluster metadata into a downloadable JSON payload.


Tech Stack & Architecture
Backend
Python 3.10+ / Flask: Web application routing, session handling, and API endpoints.


Gunicorn: Production WSGI HTTP server configured with custom worker timeouts.


Scikit-Learn: TF-IDF Vectorization, Latent Dirichlet Allocation / Matrix Decomposition, K-Means clustering, and Cosine Similarity computations.


NLTK: Text preprocessing pipeline including tokenization, stop-word removal, and WordNet lemmatization.


Joblib & Gzip: Serialization and pre-computed compressed storage (processed_corpus.json.gz) designed for low-memory environments (sub-512MB RAM constraints).


Frontend
HTML5 & CSS3: Custom dark crypto UI theme (#0b0f19 canvas, #131b2e card containers) with custom SVG assets.


Bootstrap 5 & Bootstrap Icons: Responsive grid architecture, modular accordions, and UI icons.


Chart.js: Responsive bar chart rendering total mentions alongside term growth velocity.


MathJax 3: Dynamic client-side LaTeX and mathematical notation rendering.


Project Structure
Plaintext
├── app.py                      # Flask application endpoints and API routes
├── topic_engine.py             # NLP pipeline, clustering, and vector search logic
├── processed_corpus.json.gz    # Compressed, pre-computed literature dataset
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment process configuration (e.g., Render/Heroku)
├── static/
│   └── search.svg              # Navigation brand graphic
└── templates/
    └── index.html              # Interactive dashboard UI


Installation & Local Setup
1. Clone the Repository
Bash
git clone https://github.com/your-username/ai-research-topic-finder.git
cd ai-research-topic-finder


2. Set Up a Virtual Environment
Bash
# Windows
python -m venv venv
.\venv\Scripts\activate


# Linux / macOS / WSL
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies
Bash
pip install -r requirements.txt


4. Download Required NLTK Resources
Ensure the required tokenizers and corpora are available locally:

Python
import nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'])


5. Run the Application
Bash
python app.py


Open your browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:8044/).

API Reference
GET /api/analytics
Fetches topic distributions, trend metrics, cross-domain gaps, and clusters.

Query Parameters:



domain (optional): Filter data by a specific domain (e.g., Cyber Forensics, Quantum Cryptography, or All).


Response: JSON payload containing topics, trends, gaps, and clusters.


POST /api/recommend
Matches an open-ended research interest against domain literature.

Request Body:



JSON
{
  "query": "medical deep learning for diagnosis",
  "domain": "All"
}




Response:



JSON
{
  "recommendations": [
    {
      "title": "Paper Title",
      "domain": "Domain Name",
      "year": 2024,
      "score": 0.89,
      "keywords": ["deep learning", "neural network", "diagnosis"],
      "abstract": "Abstract text..."
    }
  ]
}




GET /api/export
Compiles and streams the current state of analytics, identified research gaps, and active clusters as a downloadable JSON file.

Production Deployment
This project is configured for cloud deployment platforms such as Render:

Environment: Python 3


Build Command: pip install -r requirements.txt


Start Command:



Bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120




Memory Optimization: Uses pre-computed serialized models and capped cluster subsets to ensure operation remains strictly within 512MB RAM free-tier limits.


