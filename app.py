from flask import Flask, request, jsonify, render_template
from main import AIEnhancedDomainAnalyzer
import nest_asyncio

nest_asyncio.apply()
app = Flask(__name__)
analyzer = AIEnhancedDomainAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
async def analyze_domain():
    try:
        data = request.json
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        suggestions = await analyzer.smart_domain_search(query)
        return jsonify([{
            "name": s.name,
            "relevance": f"{s.relevance_score:.2f}",
            "available": s.availability
        } for s in suggestions])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
