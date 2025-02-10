import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel, pipeline
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
        self.domain_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.population_size = 30  # تقليل الحجم
        self.mutation_rate = 0.1
        self.generations = 15

    async def smart_domain_search(self, query: str) -> List[DomainSuggestion]:
        try:
            keywords = await self._extract_keywords(query)
            domain_candidates = await self._generate_domains_genetic(keywords)
            analyzed_domains = []
            
            for domain in domain_candidates[:5]:  # 5 نطاقات فقط
                availability = await self._check_domain_availability(domain)
                relevance = await self._analyze_domain_relevance(domain, query)
                analyzed_domains.append(DomainSuggestion(
                    name=domain,
                    relevance_score=relevance,
                    availability=availability,
                    keywords=keywords
                ))
            
            return sorted(analyzed_domains, key=lambda x: x.relevance_score, reverse=True)
        
        except Exception as e:
            print(f"Error in smart_domain_search: {str(e)}")
            return []

    async def _extract_keywords(self, query: str) -> List[str]:
        try:
            inputs = self.tokenizer(query, return_tensors="tf", truncation=True, max_length=128)
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            labels = ['technology', 'business', 'health', 'science']
            result = self.keyword_extractor(query, candidate_labels=labels, multi_label=True)
            return [label for label, score in zip(result['labels'], result['scores']) if score > 0.3]
        
        except Exception as e:
            print(f"Keyword extraction error: {str(e)}")
            return []

    def _initialize_population(self, keywords: List[str]) -> List[str]:
        return [f"{kw}.io" for kw in keywords] + [''.join(random.choices(string.ascii_lowercase, k=6)) for _ in range(10)]

    async def _check_domain_availability(self, domain: str) -> bool:
        try:
            await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, whois.query, domain), timeout=2)
            return False
        except:
            return True

    async def _analyze_domain_relevance(self, domain: str, query: str) -> float:
        try:
            domain_inputs = self.tokenizer(domain, return_tensors="tf", truncation=True, max_length=64)
            query_inputs = self.tokenizer(query, return_tensors="tf", truncation=True, max_length=64)
            
            domain_emb = self.bert_model(**domain_inputs).last_hidden_state[:, 0, :]
            query_emb = self.bert_model(**query_inputs).last_hidden_state[:, 0, :]
            
            return float(tf.keras.losses.cosine_similarity(domain_emb, query_emb).numpy()[0])
        
        except Exception as e:
            print(f"Relevance analysis error: {str(e)}")
            return 0.0

    # ... (بقية الدوال مع إضافة try/except)
