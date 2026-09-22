import os
import random
import re
import json
import gzip
import joblib
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Point NLTK to the local folder created during the Render build
nltk.data.path.append(os.path.abspath('nltk_data'))

class ResearchTopicFinder:
    def __init__(self):
        print("Loading pre-trained AI models into RAM...")
        self.lemmatizer = WordNetLemmatizer()
        
        # 1. Warm-up call to force NLTK to load dictionaries into RAM on boot
        try:
            self.lemmatizer.lemmatize("testing")
        except LookupError:
            print("Missing NLTK data. Downloading fallbacks...")
            nltk.download('wordnet', quiet=True, download_dir='./nltk_data')
            nltk.download('omw-1.4', quiet=True, download_dir='./nltk_data')
            
        # 2. Initialize and safely load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True, download_dir='./nltk_data')
            self.stop_words = set(stopwords.words('english'))

        # 3. Add your custom academic stopwords
        self.stop_words = self.stop_words.union({
            'using', 'paper', 'approach', 'model', 'technique', 'analysis',
            'application', 'system', 'based', 'via', 'study', 'method'
        })
        
        # 4. Load the pre-processed data and trained models instantly
        try:
            with gzip.open('processed_corpus.json.gz', 'rt', encoding='utf-8') as f:
                self.papers = json.load(f)
                
            self.vectorizer = joblib.load('pretrained_vectorizer.joblib')
            self.tfidf_matrix = joblib.load('pretrained_tfidf.joblib')
            self.lda_model = joblib.load('pretrained_lda.joblib')
            print("✅ Models loaded successfully!")
        except FileNotFoundError:
            print("ERROR: Missing .joblib or .json.gz files. Run train_backend.py first.")
            self.papers, self.vectorizer, self.tfidf_matrix, self.lda_model = [], None, None, None

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = text.split()
        return ' '.join([self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2])

    def extract_latent_topics(self, n_words=5):
        if not self.lda_model: return {}
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics[f"Topic {topic_idx + 1}"] = top_words
        return topics

    def cluster_topics(self, domain_filter=None, n_clusters=3):
        if not self.papers: return {}
        clusters = defaultdict(list)
        shuffled_papers = self.papers.copy()
        random.shuffle(shuffled_papers)
        if domain_filter and domain_filter.lower() != "all":
            shuffled_papers = [p for p in shuffled_papers if p['domain'].lower() == domain_filter.lower()]
        for paper in shuffled_papers:
            if len(clusters[paper['cluster']]) < 25:
                raw_abstract = str(paper.get('abstract', 'No abstract available.'))
                short_abstract = raw_abstract[:250] + '...' if len(raw_abstract) > 250 else raw_abstract
                
                clusters[paper['cluster']].append({
                    'title': paper['title'],
                    'domain': paper['domain'],
                    'year': paper['year'],
                    'abstract': short_abstract
                })
                
        return dict(clusters)

    def analyze_trends(self, domain=None):
        if not self.papers: return {'timeline': [], 'trends': []}
        papers = self.papers if not domain else [p for p in self.papers if p['domain'].lower() == domain.lower()]
        year_kw = defaultdict(lambda: defaultdict(int))
        years = sorted(list({p['year'] for p in papers}))

        for p in papers:
            for kw in p['keywords']:
                year_kw[kw][p['year']] += 1

        trends = []
        for kw, counts in year_kw.items():
            latest = counts.get(max(years), 0) if years else 0
            earliest = counts.get(min(years), 0) if years else 0
            velocity = latest - earliest
            trends.append({
                'keyword': kw,
                'total_mentions': sum(counts.values()),
                'growth_velocity': velocity,
                'yearly_distribution': {y: counts.get(y, 0) for y in years}
            })
        trends.sort(key=lambda x: (x['growth_velocity'], x['total_mentions']), reverse=True)
        return {'timeline': years, 'trends': trends[:8]}

    def identify_research_gaps(self):
        if not self.papers: return []
        domain_keywords = defaultdict(set)
        for p in self.papers:
            domain_keywords[p['domain']].update(p['keywords'])
            
        domains = list(domain_keywords.keys())
        if len(domains) < 2: return []
        
        gaps = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                shared = len(domain_keywords[d1].intersection(domain_keywords[d2]))
                gaps.append({
                    'domains': f"{d1} × {d2}",
                    'overlap_count': shared,
                    'hypothesis': f"Evaluate techniques from {d1} applied directly to {d2} challenges."
                })
        gaps.sort(key=lambda x: x['overlap_count'])
        return gaps[:5]

    def recommend_topics(self, query, domain_filter=None, top_k=3):
        if not self.papers or not query.strip(): return []
        
        q_vec = self.vectorizer.transform([self.clean_text(query)])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            paper = self.papers[idx]
            if domain_filter and domain_filter.lower() != "all" and paper['domain'].lower() != domain_filter.lower():
                continue
            if scores[idx] > 0:
                results.append({
                    'title': paper['title'],
                    'domain': paper['domain'],
                    'year': paper['year'],
                    'score': round(float(scores[idx]), 3),
                    'keywords': paper['keywords'],
                    'abstract': paper.get('abstract', 'No abstract available.')
                })
            if len(results) >= top_k: break
        return results
