from flask import Flask, render_template, request, jsonify, Response
from topic_engine import ResearchTopicFinder
import json

app = Flask(__name__)
engine = ResearchTopicFinder()

@app.route('/')
def index():
    domains = sorted(list({p['domain'] for p in engine.papers}))
    return render_template('index.html', domains=domains)

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    domain = request.args.get('domain', None)
    if domain == "All":
        domain = None
    
    trend_data = engine.analyze_trends(domain=domain)
    clusters = engine.cluster_topics(n_clusters=3)
    lda_topics = engine.extract_latent_topics(n_words=4)
    gaps = engine.identify_research_gaps()

    return jsonify({
        'trends': trend_data,
        'clusters': clusters,
        'topics': lda_topics,
        'gaps': gaps
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    payload = request.get_json() or {}
    query = payload.get('query', '')
    domain = payload.get('domain', None)
    results = engine.recommend_topics(query=query, domain_filter=domain)
    return jsonify({'recommendations': results})

@app.route('/api/export', methods=['GET'])
def export_report():
    report_data = {
        'database_size': len(engine.papers),
        'lda_latent_topics': engine.extract_latent_topics(n_words=5),
        'top_emerging_trends': engine.analyze_trends()['trends'],
        'identified_research_gaps': engine.identify_research_gaps()
    }
    return Response(
        json.dumps(report_data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment;filename=research_topic_report.json'}
    )

import os

if __name__ == '__main__':
    # Fetch the dynamically assigned port from Render, fallback to local 8044
    port = int(os.environ.get('PORT', 8044))
    # Bind to 0.0.0.0 to allow external web traffic
    app.run(host='0.0.0.0', port=port, debug=False)
