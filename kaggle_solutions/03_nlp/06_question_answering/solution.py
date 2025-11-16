"""
Question Answering System - Kaggle NLP Solution
================================================
This solution demonstrates a simple extractive question answering system
that finds relevant answers from given contexts using TF-IDF similarity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings
import re

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class QuestionAnsweringSystem:
    """Simple extractive QA system using TF-IDF and cosine similarity"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.contexts = []
        self.context_vectors = None

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\?\!]', '', text)
        return text

    def fit(self, contexts):
        """Fit the QA system on contexts"""
        self.contexts = [self.preprocess_text(c) for c in contexts]
        # Split contexts into sentences for better granularity
        self.sentences = []
        self.sentence_to_context = []

        for idx, context in enumerate(self.contexts):
            sents = sent_tokenize(context)
            self.sentences.extend(sents)
            self.sentence_to_context.extend([idx] * len(sents))

        # Create TF-IDF vectors for sentences
        self.context_vectors = self.vectorizer.fit_transform(self.sentences)

    def answer(self, question, top_k=3):
        """Find answer for the given question"""
        question = self.preprocess_text(question)
        question_vector = self.vectorizer.transform([question])

        # Calculate similarity
        similarities = cosine_similarity(question_vector, self.context_vectors)[0]

        # Get top k most similar sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        answers = []
        for idx in top_indices:
            answers.append({
                'answer': self.sentences[idx],
                'score': similarities[idx],
                'context_id': self.sentence_to_context[idx]
            })

        return answers

def generate_qa_dataset():
    """Generate realistic QA dataset"""
    contexts = [
        "Python is a high-level programming language created by Guido van Rossum. "
        "It was first released in 1991. Python emphasizes code readability and uses "
        "significant whitespace. It supports multiple programming paradigms including "
        "procedural, object-oriented, and functional programming.",

        "Machine learning is a subset of artificial intelligence that enables systems "
        "to learn and improve from experience. It focuses on developing computer programs "
        "that can access data and use it to learn for themselves. Deep learning is a "
        "specialized form of machine learning using neural networks.",

        "The Amazon rainforest covers approximately 5.5 million square kilometers. "
        "It spans across nine countries in South America. The forest is home to over "
        "400 billion individual trees and thousands of species. It plays a crucial role "
        "in regulating global climate and produces about 20% of the world's oxygen.",

        "Climate change refers to long-term shifts in global temperatures and weather patterns. "
        "Human activities, particularly burning fossil fuels, are the main driver. This releases "
        "greenhouse gases like carbon dioxide into the atmosphere. The effects include rising sea "
        "levels, extreme weather events, and ecosystem disruption.",

        "The Internet was developed in the late 1960s as ARPANET. Tim Berners-Lee invented "
        "the World Wide Web in 1989 while working at CERN. The web made the internet accessible "
        "to the general public. Today, over 5 billion people worldwide use the internet.",

        "Photosynthesis is the process by which plants convert light energy into chemical energy. "
        "Chlorophyll in plant cells absorbs sunlight. Plants use carbon dioxide from air and water "
        "from soil to produce glucose and oxygen. This process is essential for life on Earth.",

        "The Great Wall of China stretches over 21,000 kilometers. Construction began in the "
        "7th century BC and continued for centuries. It was built to protect Chinese states from "
        "invasions. Despite popular belief, it is not visible from space with the naked eye.",

        "Bitcoin is a decentralized digital currency created in 2009 by an unknown person using "
        "the pseudonym Satoshi Nakamoto. It operates without a central bank or administrator. "
        "Transactions are verified by network nodes through cryptography and recorded on a "
        "blockchain. Bitcoin can be exchanged for other currencies, products, and services."
    ]

    questions = [
        "Who created Python programming language?",
        "When was Python first released?",
        "What is machine learning?",
        "How large is the Amazon rainforest?",
        "How many countries does Amazon rainforest span?",
        "What causes climate change?",
        "Who invented the World Wide Web?",
        "When was the World Wide Web invented?",
        "What is photosynthesis?",
        "How long is the Great Wall of China?",
        "When was Bitcoin created?",
        "Who created Bitcoin?",
        "What does chlorophyll do?",
        "How much oxygen does Amazon produce?",
        "What is deep learning?",
        "Can you see Great Wall from space?"
    ]

    expected_answers = [
        "Guido van Rossum",
        "1991",
        "subset of artificial intelligence",
        "5.5 million square kilometers",
        "nine countries",
        "burning fossil fuels",
        "Tim Berners-Lee",
        "1989",
        "process by which plants convert light energy",
        "21,000 kilometers",
        "2009",
        "Satoshi Nakamoto",
        "absorbs sunlight",
        "20% of the world's oxygen",
        "specialized form of machine learning",
        "not visible from space"
    ]

    return contexts, questions, expected_answers

def evaluate_qa_system(qa_system, questions, expected_answers):
    """Evaluate the QA system"""
    results = []

    for question, expected in zip(questions, expected_answers):
        answers = qa_system.answer(question, top_k=1)
        top_answer = answers[0]['answer']
        score = answers[0]['score']

        # Simple evaluation: check if expected answer is in retrieved answer
        is_correct = expected.lower() in top_answer.lower()

        results.append({
            'question': question,
            'answer': top_answer,
            'expected': expected,
            'score': score,
            'correct': is_correct
        })

    return pd.DataFrame(results)

def create_visualizations(results_df):
    """Create visualizations for QA system performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Accuracy
    accuracy = results_df['correct'].mean() * 100
    axes[0, 0].bar(['Correct', 'Incorrect'],
                   [results_df['correct'].sum(), len(results_df) - results_df['correct'].sum()],
                   color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_title(f'Answer Accuracy: {accuracy:.1f}%', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Confidence scores distribution
    axes[0, 1].hist(results_df['score'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Confidence Scores', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Cosine Similarity Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(results_df['score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {results_df["score"].mean():.3f}')
    axes[0, 1].legend()

    # 3. Score comparison: Correct vs Incorrect
    correct_scores = results_df[results_df['correct']]['score']
    incorrect_scores = results_df[~results_df['correct']]['score']

    box_data = [correct_scores, incorrect_scores]
    axes[1, 0].boxplot(box_data, labels=['Correct', 'Incorrect'])
    axes[1, 0].set_title('Confidence Scores: Correct vs Incorrect', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Cosine Similarity Score')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Top 10 questions by score
    top_10 = results_df.nlargest(10, 'score')
    y_pos = np.arange(len(top_10))
    colors = ['green' if c else 'red' for c in top_10['correct']]

    axes[1, 1].barh(y_pos, top_10['score'], color=colors, alpha=0.7)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([q[:30] + '...' if len(q) > 30 else q
                                 for q in top_10['question']], fontsize=8)
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_title('Top 10 Questions by Confidence', fontsize=12, fontweight='bold')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('qa_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'qa_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Question Answering System - Kaggle NLP Solution")
    print("=" * 60)

    # Generate dataset
    print("\n1. Generating QA Dataset...")
    contexts, questions, expected_answers = generate_qa_dataset()
    print(f"   - Contexts: {len(contexts)}")
    print(f"   - Questions: {len(questions)}")

    # Initialize and train QA system
    print("\n2. Training QA System...")
    qa_system = QuestionAnsweringSystem()
    qa_system.fit(contexts)
    print(f"   - Sentences extracted: {len(qa_system.sentences)}")
    print(f"   - Vocabulary size: {len(qa_system.vectorizer.vocabulary_)}")

    # Evaluate system
    print("\n3. Evaluating QA System...")
    results_df = evaluate_qa_system(qa_system, questions, expected_answers)

    accuracy = results_df['correct'].mean() * 100
    avg_score = results_df['score'].mean()

    print(f"   - Accuracy: {accuracy:.1f}%")
    print(f"   - Average Confidence: {avg_score:.3f}")

    # Show sample results
    print("\n4. Sample Question-Answer Pairs:")
    print("-" * 60)
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        status = "✓" if row['correct'] else "✗"
        print(f"\n{status} Q: {row['question']}")
        print(f"   A: {row['answer'][:80]}...")
        print(f"   Expected: {row['expected']}")
        print(f"   Score: {row['score']:.3f}")

    # Create visualizations
    print("\n5. Creating Visualizations...")
    create_visualizations(results_df)

    # Interactive demo
    print("\n6. Interactive Demo:")
    print("-" * 60)
    demo_questions = [
        "What programming paradigms does Python support?",
        "What is the role of Amazon rainforest in climate?",
        "How does blockchain work with Bitcoin?"
    ]

    for q in demo_questions:
        print(f"\nQ: {q}")
        answers = qa_system.answer(q, top_k=2)
        for i, ans in enumerate(answers, 1):
            print(f"   {i}. {ans['answer']} (score: {ans['score']:.3f})")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
