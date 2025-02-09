<!DOCTYPE html>
<html dir="rtl">
<head>
    <title>محلل النطاقات - eaq.io</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        input { width: 300px; padding: 10px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; }
        .result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>أدخل الكلمات المفتاحية لبحث النطاقات:</h1>
    <input type="text" id="query" placeholder="مثال: تقنية، أعمال...">
    <button onclick="analyze()">بحث</button>
    <div id="results"></div>

    <script>
        async function analyze() {
            const query = document.getElementById('query').value;
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            
            const results = await response.json();
            let html = '';
            
            results.forEach(domain => {
                html += `
                    <div class="result">
                        <h3>${domain.name}</h3>
                        <p>التوفر: ${domain.availability ? '🟢 متاح' : '🔴 غير متاح'}</p>
                        <p>الملاءمة: ${Math.round(domain.relevance_score * 100)}%</p>
                    </div>
                `;
            });
            
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
