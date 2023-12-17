# NLP_Archiving

★ : 읽어 봄

☆ : 관심 있음

## 1. 데이터, 전처리

Efficient Estimation of Word Representations in Vector Space(2013) ★

*	Word2Vec 논문, CBOW, Skip-gram

GloVe: Global Vectors for Word Representation(2014) ★

*	GloVe 논문, corpus들의 공동 출현 정보로 단어 임베딩

Neural Machine Translation of Rare Words with Subword Units(2016) ★

*	희귀단어를 처리하기 위한 Subword 단위 인코딩 소개

Enriching Word Vectors with Subword Information(2016) ★

*	FastText 논문, n-gram subword 표현

AEDA: An Easier Data Augmentation Technique for Text Classification (2022) ☆

* Punctuation(온점,세미콜론…)을 통한 데이터 증강

Neural Machine Translation of Rare Words with Subword Units (2022) ★

* BPE(Byte Pair Encoding)을 사용한 subword tokenizer 개발

Judging a Book by its Cover (2023) ★

* 책 표지, 장르 데이터셋 구축

<br>

## 2. 학습

Estimating the Uncertainty of Average F1 Scores (2015) ☆

*	Macro, Micro F1 score 들의 불확실성 시각화, Markov Chain Monte Carlo(MCMC) 추적으로 입증 -> 대안 제시는 안하지만 얼마나 불확실한지 분류

Dropout Reduces Underfitting (2023) ★

*	Dropout을 초기에 적용해 과대적합이 아니라 과소적합을 해결함

Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (2020) ☆

*	Fine-Tuning 분야 survey 논문

<br>

## 3. 모델

LONG SHORT-TERM MEMORY (1997) ★

*	LSTM 논문

Attention Is All You Need (2017) ★

*	Transformer 논문

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018) ★

*	BERT 논문

Improving Language Understanding by Generative Pre-Training (2018) ★

*	ELMO 논문

ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS (2020) ★

*	ELECTRA 논문, BERT에서 replaced token detection 추가, 비교적 작은 모델

ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS (2019) ☆

*	NSP(Next Sentence Prediction) 대신 SOP(Sentence Order Prediction) 사용, 파라미터 공유 및 임베딩 크기 감소로 BERT의 성능 향상

RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019) ★

*	RoBERTa 논문, 대용량 BERT

Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019) ★

*	Sentence-BERT, 문장 임베딩을 위한 BERT -> STS에 활용 가능

SpanBERT: Improving Pre-training by Representing and Predicting Spans (2019) ☆

*	SpanBERT 논문, 예측을 구 단위로 진행하는 BERT

PEGASUS Pre-training with Extracted Gap-sentences for Abstractive Summarization (2020) ★

*	Pegasus 논문, 생성 모델, 요약 작업 진행

Enriching Pre-trained Language Model with Entity Information for Relation Classification (2020) ☆

*	R-BERT 논문, 관계 분류용 언어 모델

Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling (2023) ★

*	Sparse masked modeling(SparK)를 통해 CNN을 BERT 스타일로 사전훈련 시킴

GPT-4 Technical Report (2023) ☆

*	GPT-4 논문

<br>

## 4. NLP 세부 task

Neural Machine Translation by Jointly Learning to Align and Translate (2015) ☆

*	Soft-search 아키텍처 고안, 양방향 RNN 인코더, soft-attention

An Improved Baseline for Sentence-level Relation Extraction (2017) ★

*	Typed entity marker를 통한 RE task 성능 향상

CALIBRATING SEQUENCE LIKELIHOOD IMPROVES (2018) ☆

*	언어모델 MLE를 통한 학습 후 input - 후보군 시퀀스 강화, BERTScore를 변환하여 점수 측정

Curriculum Learning for Natural Language Understanding (2020) ★

*	curriculum learning, 점진적으로 맞추기 어려운 훈련 데이터 학습

Dense Passage Retrieval for Open-Domain Question Answering (2020) ★ 

* DPR, BERT 구조로 임베딩 -> 내적으로 유사도 측정 

Learning Dense Representations of Phrases at Scale (2021) ☆

* DensePhrases, DPR과 달리 구문 단위 임베딩 하여 복잡한 문제에 특화

GPT Understands, Too (2021) ★

*	P-Tuning, 다양한 task에 대한 모델 일반화

LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS (2023) ☆

*	언어모델을 APE(자동 프롬프트 엔지니어)로 사용하는 방법 제시
