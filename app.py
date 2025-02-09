# في بداية ملف app.py
    import nest_asyncio
    nest_asyncio.apply()
from flask import Flask, request, jsonify, render_template
from main import AIEnhancedDomainAnalyzer
import asyncio

app = Flask(__name__)

# تهيئة المحلل عند تشغيل التطبيق
analyzer = AIEnhancedDomainAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')  # واجهة المستخدم الرئيسية

@app.route('/analyze', methods=['POST'])
async def analyze_domain():
    data = request.json
    query = data.get('query', '')
    
    # تشغيل التحليل (باستخدام asyncio)
    suggestions = await analyzer.smart_domain_search(query)
    
    # تحويل النتائج إلى تنسيق JSON
    results = [{
        "name": s.name,
        "relevance_score": s.relevance_score,
        "availability": s.availability,
        "risk_score": s.risk_score,
        "category": s.category,
        "keywords": s.keywords
    } for s in suggestions]
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
