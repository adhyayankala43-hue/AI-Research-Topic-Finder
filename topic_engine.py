import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk

# Point NLTK to the local folder created during the Render build
nltk.data.path.append(os.path.abspath('nltk_data'))


class ResearchTopicFinder:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')).union({
            'using', 'paper', 'approach', 'model', 'technique', 'analysis',
            'application', 'system', 'based', 'via', 'study', 'method'
        })
        
        # Load the newly accumulated CSV file
        self.papers = self._initialize_corpus('diverse_research_corpus.csv.gz')
        
        self.vectorizer = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.lda_model = None
        self._fit_models()

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = nltk.word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2])

    def _initialize_corpus(self, csv_filepath):
        try:
            # Read the dataset using pandas
            df = pd.read_csv(csv_filepath, compression='gzip')
            
            # Convert the dataframe into a list of dictionaries
            corpus = df.to_dict('records')
            
            for idx, item in enumerate(corpus):
                item['id'] = idx
                item['year'] = int(item.get('year', 2024))
                
                # Clean the text using both title and abstract
                item['clean_text'] = self.clean_text(f"{item.get('title', '')} {item.get('abstract', '')}")
                
                # Extract top 5 keywords
                item['keywords'] = [word for word, _ in Counter(item['clean_text'].split()).most_common(5)]
            
            return corpus
            
        except FileNotFoundError:
            print(f"Warning: {csv_filepath} not found. Ensure you ran fetch_papers.py first.")
            return [] 

    def _fit_models(self):
        if not self.papers:
            return
        docs = [p['clean_text'] for p in self.papers]
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        self.lda_model = LatentDirichletAllocation(n_components=4, random_state=42)
        self.lda_model.fit(self.tfidf_matrix)

    def extract_latent_topics(self, n_words=5):
        if not self.lda_model: 
            return {}
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics[f"Topic {topic_idx + 1}"] = top_words
        return topics

    def cluster_topics(self, n_clusters=3):
        if not self.papers: 
            return {}
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(self.tfidf_matrix)
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            # Explicitly fetch the abstract from the dataset
            paper_abstract = self.papers[idx].get('abstract', '')
            if not paper_abstract:
                paper_abstract = "No abstract available in the dataset."
                
            clusters[int(label)].append({
                'title': self.papers[idx]['title'],
                'domain': self.papers[idx]['domain'],
                'year': self.papers[idx]['year'],
                'abstract': paper_abstract  # Sending to frontend
            })
        return dict(clusters)

    def analyze_trends(self, domain=None):
        if not self.papers: 
            return {'timeline': [], 'trends': []}
            
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
        if not self.papers: 
            return []
        
        domain_keywords = defaultdict(set)
        for p in self.papers:
            domain_keywords[p['domain']].update(p['keywords'])

        domains = list(domain_keywords.keys())
        if len(domains) < 2:
            return [] # Need at least 2 domains to compare

        gaps = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                shared = domain_keywords[d1].intersection(domain_keywords[d2])
                
                gaps.append({
                    'domains': f"{d1} × {d2}",
                    'overlap_count': len(shared),
                    'hypothesis': f"Evaluate techniques from {d1} applied directly to {d2} challenges."
                })
        
        # Sort pairs by the lowest overlap count (the biggest "gaps" in research)
        gaps.sort(key=lambda x: x['overlap_count'])
        
        # Always return the top 5 largest gaps
        return gaps[:5]

    def recommend_topics(self, query, domain_filter=None, top_k=3):
        if not self.papers: 
            return []
        clean_q = self.clean_text(query)
        if not clean_q.strip():
            return []
        
        q_vec = self.vectorizer.transform([clean_q])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        ranked_indices = scores.argsort()[::-1]
        results = []
        for idx in ranked_indices:
            paper = self.papers[idx]
            if domain_filter and domain_filter.lower() != "all" and paper['domain'].lower() != domain_filter.lower():
                continue
            if scores[idx] > 0:
                # Explicitly fetch the abstract from the dataset
                paper_abstract = paper.get('abstract', '')
                if not paper_abstract:
                    paper_abstract = "No abstract available in the dataset."
                    
                results.append({
                    'title': paper['title'],
                    'domain': paper['domain'],
                    'year': paper['year'],
                    'score': round(float(scores[idx]), 3),
                    'keywords': paper['keywords'],
                    'abstract': paper_abstract  # Sending to frontend
                })
            if len(results) >= top_k:
                break
        return results
