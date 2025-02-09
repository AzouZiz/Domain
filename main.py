import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, pipeline
import random
import string
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import whois
import asyncio
import aiohttp

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
        self.domain_classifier = RandomForestClassifier()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.population_size = 50  # تقليل الحجم لتجنب المهلة
        self.mutation_rate = 0.1
        self.generations = 20  # تقليل عدد الأجيال

    async def smart_domain_search(self, query: str) -> List[DomainSuggestion]:
        keywords = await self._extract_keywords(query)
        domain_candidates = await self._generate_domains_genetic(keywords)
        
        analyzed_domains = []
        for domain in domain_candidates[:10]:  # تقليل عدد النطاقات المفحوصة
            relevance = await self._analyze_domain_relevance(domain, query)
            availability = await self._check_domain_availability(domain)
            analyzed_domains.append(DomainSuggestion(
                name=domain,
                relevance_score=relevance,
                availability=availability,
                keywords=keywords
            ))
        
        return sorted(analyzed_domains, key=lambda x: x.relevance_score, reverse=True)

    async def _extract_keywords(self, query: str) -> List[str]:
        inputs = self.tokenizer(query, return_tensors="tf", truncation=True, max_length=128)
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        labels = ['technology', 'business', 'entertainment', 'health', 'science']
        result = await self.keyword_extractor(query, candidate_labels=labels, multi_label=True)
        
        keywords = [label for label, score in zip(result['labels'], result['scores']) if score > 0.3]
        return list(set(keywords))

    async def _generate_domains_genetic(self, keywords: List[str]) -> List[str]:
        population = self._initialize_population(keywords)
        for _ in range(self.generations):
            fitness_scores = [self._evaluate_fitness(domain, keywords) for domain in population]
            parents = self._select_parents(population, fitness_scores)
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            population = offspring
        return population[:20]  # إرجاع أفضل 20 نطاق فقط

    # ... (بقية الدوال كما هي مع تعديلات الطول)
