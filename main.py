import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, pipeline
import random
import string
from typing import List, Optional
from dataclasses import dataclass
import whois
import asyncio

@dataclass
class DomainSuggestion:
    name: str
    relevance_score: float
    availability: bool
    risk_score: Optional[float] = None
    category: Optional[str] = None
    keywords: List[str] = None

class AIEnhancedDomainAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.keyword_extractor = pipeline('zero-shot-classification')
        self.domain_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=0.0,
            max_df=1.0
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.population_size = 50
        self.mutation_rate = 0.1
        self.generations = 20

    def _initialize_population(self, keywords: List[str]) -> List[str]:
        """تهيئة المجتمع الأولي باستخدام الكلمات المفتاحية"""
        base_domains = [f"{kw}.io" for kw in keywords]
        mutations = [''.join(random.choices(string.ascii_lowercase, k=6)) for _ in range(20)]
        return base_domains + mutations

    async def _check_domain_availability(self, domain: str) -> bool:
        """التحقق من توفر النطاق"""
        try:
            await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, whois.query, domain), timeout=2)
            return False
        except:
            return True

    def _calculate_similarity(self, domain: str, keyword: str) -> float:
        """حساب التشابه بين النطاق والكلمة المفتاحية"""
        vectors = self.domain_vectorizer.fit_transform([domain, keyword])
        return np.dot(vectors[0].toarray(), vectors[1].toarray().T)[0][0]

    async def _analyze_domain_relevance(self, domain: str, query: str) -> float:
        """تحليل ملاءمة النطاق"""
        domain_tokens = self.tokenizer(domain, return_tensors="tf", truncation=True)
        query_tokens = self.tokenizer(query, return_tensors="tf", truncation=True)
        
        domain_emb = self.bert_model(**domain_tokens).last_hidden_state[:, 0, :]
        query_emb = self.bert_model(**query_tokens).last_hidden_state[:, 0, :]
        
        similarity = tf.reduce_mean(tf.keras.losses.cosine_similarity(domain_emb, query_emb)).numpy()
        return float(similarity)

    async def smart_domain_search(self, query: str) -> List[DomainSuggestion]:
        """البحث الذكي مع تحسينات للأداء"""
        keywords = await self._extract_keywords(query)
        domain_candidates = await self._generate_domains_genetic(keywords)
        
        analyzed_domains = []
        for domain in domain_candidates[:10]:
            availability = await self._check_domain_availability(domain)
            if availability:
                relevance = await self._analyze_domain_relevance(domain, query)
                analyzed_domains.append(DomainSuggestion(
                    name=domain,
                    relevance_score=relevance,
                    availability=True,
                    keywords=keywords
                ))
        
        return sorted(analyzed_domains, key=lambda x: x.relevance_score, reverse=True)[:5]  # أفضل 5 نتائج فقط

# باقي الدوال (التي ذكرتها سابقًا) هنا...
