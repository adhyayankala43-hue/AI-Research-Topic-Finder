import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import time

def fetch_diverse_arxiv_papers(topics, max_per_topic=100):
    all_papers = []
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for topic in topics:
        print(f"Fetching papers for: {topic}...")
        
        # Format query for URL
        query = urllib.parse.quote(topic)
        url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_per_topic}'
        
        try:
            response = urllib.request.urlopen(url)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            for entry in root.findall('atom:entry', namespace):
                all_papers.append({
                    'title': entry.find('atom:title', namespace).text.strip().replace('\n', ' '),
                    'abstract': entry.find('atom:summary', namespace).text.strip().replace('\n', ' '),
                    'year': entry.find('atom:published', namespace).text[:4],
                    'domain': topic.title() # Use the search topic as the domain label
                })
                
            # Sleep briefly to respect API rate limits
            time.sleep(3)
            
        except Exception as e:
            print(f"Error fetching {topic}: {e}")
            
    return pd.DataFrame(all_papers)

# Define a diverse list of research areas
diverse_topics = [
    "cyber forensics",
    "mobile device extraction",
    "acoustic feature extraction",
    "computer vision",
    "machine learning security",
    "natural language processing",
    "quantum computing",
    "forensic DNA analysis",
    "autosomal STR profiling",
    "qPCR quantification techniques",
    "computational biology",
    "epidemiology and public health",
    "climate change modeling",
    "oceanography",
    "geospatial analysis",
    "renewable energy systems",
    "neuroscience",
    "medical image segmentation",
    "pharmacogenomics",
    "cognitive behavioral psychology",
    "digital audio forensics",
    "hybrid synthetic speech detection",
    "quantum cryptography",
    "edge computing architecture",
    "interactive narrative design",
    "esports analytics and performance",
    "computational linguistics",
    "digital humanities"
]

# Fetch 100 papers from each of the 7 topics (Total: ~700 papers)
diverse_corpus_df = fetch_diverse_arxiv_papers(diverse_topics, max_per_topic=1000)

print(f"\nSuccessfully accumulated {len(diverse_corpus_df)} diverse papers.")
print("\nDomain Breakdown:")
print(diverse_corpus_df['domain'].value_counts())

# Save to CSV for your Topic Finder app
diverse_corpus_df.to_csv("diverse_research_corpus.csv", index=False)
