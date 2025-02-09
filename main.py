import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from transformers import pipeline
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
    """فئة لتمثيل اقتراحات النطاقات"""
    name: str
    relevance_score: float
    availability: bool
    risk_score: Optional[float] = None
    category: Optional[str] = None
    keywords: List[str] = None

class AIEnhancedDomainAnalyzer:
    """محلل النطاقات المعزز بالذكاء الاصطناعي"""
    
    def __init__(self):
        # تهيئة نماذج الذكاء الاصطناعي
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.keyword_extractor = pipeline('zero-shot-classification')
        self.domain_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=0.0,
            max_df=1.0
        )
        self.domain_classifier = RandomForestClassifier()
        
        # تحميل نموذج BERT المدرب مسبقاً
        self.bert_model = tf.keras.models.load_weights('path_to_bert_model')
        
        # تهيئة المحلل الجيني
        self.population_size = 100
        self.mutation_rate = 0.1
        self.generations = 50
        
    async def smart_domain_search(self, query: str) -> List[DomainSuggestion]:
        """البحث الذكي عن النطاقات باستخدام الذكاء الاصطناعي"""
        
        # استخراج الكلمات المفتاحية والمفاهيم
        keywords = await self._extract_keywords(query)
        
        # توليد النطاقات باستخدام الخوارزمية الجينية
        domain_candidates = await self._generate_domains_genetic(keywords)
        
        # تحليل وتصنيف النطاقات المقترحة
        analyzed_domains = []
        for domain in domain_candidates:
            # تحليل مدى ملاءمة النطاق
            relevance = await self._analyze_domain_relevance(domain, query)
            
            # التحقق من توفر النطاق
            availability = await self._check_domain_availability(domain)
            
            # تحليل المخاطر
            risk_score = await self._analyze_domain_risk(domain)
            
            # تصنيف النطاق
            category = await self._categorize_domain(domain)
            
            analyzed_domains.append(DomainSuggestion(
                name=domain,
                relevance_score=relevance,
                availability=availability,
                risk_score=risk_score,
                category=category,
                keywords=keywords
            ))
        
        # ترتيب النتائج حسب الأهمية والملاءمة
        return sorted(analyzed_domains, key=lambda x: x.relevance_score, reverse=True)

    async def _extract_keywords(self, query: str) -> List[str]:
        """استخراج الكلمات المفتاحية باستخدام نموذج BERT"""
        # تحويل النص إلى متجهات
        embeddings = self.bert_model.encode(query)
        
        # تصنيف متعدد التسميات
        labels = [
            'technology', 'business', 'entertainment', 'health',
            'science', 'sports', 'education', 'shopping'
        ]
        
        result = await self.keyword_extractor(
            query,
            candidate_labels=labels,
            multi_label=True
        )
        
        # استخراج الكلمات المفتاحية ذات الصلة
        keywords = []
        for label, score in zip(result['labels'], result['scores']):
            if score > 0.3:  # عتبة الثقة
                keywords.extend(self._get_related_terms(label))
                
        return list(set(keywords))

    async def _generate_domains_genetic(self, keywords: List[str]) -> List[str]:
        """توليد النطاقات باستخدام الخوارزمية الجينية"""
        
        # تهيئة المجتمع الأولي
        population = self._initialize_population(keywords)
        
        for generation in range(self.generations):
            # تقييم اللياقة
            fitness_scores = [self._evaluate_fitness(domain, keywords) for domain in population]
            
            # اختيار الأفضل
            parents = self._select_parents(population, fitness_scores)
            
            # التزاوج
            offspring = self._crossover(parents)
            
            # الطفرة
            offspring = self._mutate(offspring)
            
            # تحديث المجتمع
            population = offspring
        
        # اختيار أفضل النطاقات
        return self._select_best_domains(population, keywords)

    def _evaluate_fitness(self, domain: str, keywords: List[str]) -> float:
        """تقييم مدى ملاءمة النطاق"""
        score = 0.0
        
        # تقييم الطول
        if 5 <= len(domain) <= 15:
            score += 0.3
            
        # تقييم سهولة النطق
        score += self._calculate_pronounceability(domain) * 0.2
        
        # تقييم العلاقة بالكلمات المفتاحية
        keyword_relevance = max(self._calculate_similarity(domain, keyword) for keyword in keywords)
        score += keyword_relevance * 0.5
        
        return score

    async def _analyze_domain_relevance(self, domain: str, query: str) -> float:
        """تحليل مدى ملاءمة النطاق للاستعلام"""
        # تحويل النص إلى متجهات
        domain_embedding = self.bert_model.encode(domain)
        query_embedding = self.bert_model.encode(query)
        
        # حساب التشابه
        similarity = np.dot(domain_embedding, query_embedding) / (
            np.linalg.norm(domain_embedding) * np.linalg.norm(query_embedding)
        )
        
        # تحليل المشاعر
        sentiment = await self.sentiment_analyzer(domain)
        sentiment_score = sentiment[0]['score'] if sentiment[0]['label'] == 'POSITIVE' else 0
        
        # دمج النتائج
        return 0.7 * similarity + 0.3 * sentiment_score

    def _select_parents(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        """اختيار الأفراد الأفضل للتزاوج"""
        # اختيار النخبة
        elite_size = int(0.1 * len(population))
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite = [population[i] for i in elite_indices]
        
        # اختيار الباقي باستخدام عجلة الروليت
        selection_probs = np.array(fitness_scores) / sum(fitness_scores)
        selected_indices = np.random.choice(
            len(population),
            size=len(population) - elite_size,
            p=selection_probs
        )
        
        return elite + [population[i] for i in selected_indices]

    def _crossover(self, parents: List[str]) -> List[str]:
        """عملية التزاوج بين النطاقات"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                # اختيار نقطة التقاطع
                crossover_point = random.randint(1, min(len(parents[i]), len(parents[i+1])) - 1)
                
                # إنشاء النسل
                child1 = parents[i][:crossover_point] + parents[i+1][crossover_point:]
                child2 = parents[i+1][:crossover_point] + parents[i][crossover_point:]
                
                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i])
                
        return offspring

    def _mutate(self, population: List[str]) -> List[str]:
        """إجراء الطفرات على النطاقات"""
        mutated = []
        
        for domain in population:
            if random.random() < self.mutation_rate:
                # اختيار نوع الطفرة
                mutation_type = random.choice(['substitute', 'insert', 'delete'])
                
                if mutation_type == 'substitute':
                    pos = random.randint(0, len(domain) - 1)
                    char = random.choice(string.ascii_lowercase + string.digits)
                    domain = domain[:pos] + char + domain[pos+1:]
                    
                elif mutation_type == 'insert':
                    pos = random.randint(0, len(domain))
                    char = random.choice(string.ascii_lowercase + string.digits)
                    domain = domain[:pos] + char + domain[pos:]
                    
                elif mutation_type == 'delete':
                    if len(domain) > 3:  # تجنب النطاقات القصيرة جداً
                        pos = random.randint(0, len(domain) - 1)
                        domain = domain[:pos] + domain[pos+1:]
                        
            mutated.append(domain)
            
        return mutated

    @staticmethod
    def _calculate_pronounceability(domain: str) -> float:
        """حساب مدى سهولة نطق النطاق"""
        vowels = set('aeiou')
        consonants = set('bcdfghjklmnpqrstvwxyz')
        
        score = 0.0
        prev_char = None
        
        for char in domain.lower():
            if char in vowels:
                if prev_char in consonants:
                    score += 0.1
            elif char in consonants:
                if prev_char in vowels:
                    score += 0.1
            prev_char = char
            
        return min(1.0, score)

    async def _check_domain_availability(self, domain: str) -> bool:
        """التحقق من توفر النطاق"""
        try:
            await whois.query(domain)
            return False
        except:
            return True

    def generate_report(self, suggestions: List[DomainSuggestion]) -> str:
        """إنشاء تقرير مفصل عن النطاقات المقترحة"""
        report = []
        report.append("تقرير اقتراحات النطاقات")
        report.append("=" * 50)
        
        for i, suggestion in enumerate(suggestions, 1):
            report.append(f"\n{i}. النطاق: {suggestion.name}")
            report.append(f"   درجة الملاءمة: {suggestion.relevance_score:.2f}")
            report.append(f"   الحالة: {'متاح' if suggestion.availability else 'غير متاح'}")
            if suggestion.risk_score:
                report.append(f"   درجة المخاطر: {suggestion.risk_score:.2f}")
            if suggestion.category:
                report.append(f"   التصنيف: {suggestion.category}")
            if suggestion.keywords:
                report.append(f"   الكلمات المفتاحية: {', '.join(suggestion.keywords)}")
                
        return "\n".join(report)
