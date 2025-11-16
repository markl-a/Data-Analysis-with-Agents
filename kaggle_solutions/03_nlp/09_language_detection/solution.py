"""
Language Detection - Kaggle NLP Solution
========================================
This solution demonstrates multi-language classification using character
n-grams and Naive Bayes classifier for detecting the language of text.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

class LanguageDetector:
    """Language detection using character n-grams and Naive Bayes"""

    def __init__(self, ngram_range=(1, 3)):
        """
        Args:
            ngram_range: Range of n-gram sizes (default: 1-3 characters)
        """
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=3000,
            lowercase=True
        )
        self.classifier = MultinomialNB(alpha=0.1)
        self.languages = []

    def fit(self, texts, languages):
        """Train the language detector"""
        self.languages = sorted(list(set(languages)))
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, languages)

    def predict(self, texts):
        """Predict languages for texts"""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Get prediction probabilities"""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def get_top_features(self, language, n=10):
        """Get top character n-grams for a language"""
        feature_names = self.vectorizer.get_feature_names_out()
        lang_idx = self.languages.index(language)

        # Get feature log probabilities for this language
        feature_log_prob = self.classifier.feature_log_prob_[lang_idx]

        # Get top features
        top_indices = np.argsort(feature_log_prob)[-n:][::-1]
        top_features = [(feature_names[i], feature_log_prob[i])
                       for i in top_indices]

        return top_features

def generate_language_dataset():
    """Generate multilingual dataset"""

    # Sample texts in different languages
    data = {
        'English': [
            "Hello, how are you today?",
            "The weather is beautiful this morning.",
            "I love reading books and learning new things.",
            "Technology is changing our world rapidly.",
            "Let's meet tomorrow at the coffee shop.",
            "She plays the piano very well.",
            "The cat is sleeping on the couch.",
            "We are going to the beach this weekend.",
            "Education is the key to success.",
            "The movie was absolutely fantastic!",
            "I enjoy cooking Italian food.",
            "Programming can be challenging but rewarding.",
            "The mountains look stunning in winter.",
            "Music brings people together from all cultures.",
            "Reading helps improve your vocabulary and imagination.",
        ],
        'Spanish': [
            "Hola, ¿cómo estás hoy?",
            "El clima es hermoso esta mañana.",
            "Me encanta leer libros y aprender cosas nuevas.",
            "La tecnología está cambiando nuestro mundo rápidamente.",
            "Nos vemos mañana en la cafetería.",
            "Ella toca el piano muy bien.",
            "El gato está durmiendo en el sofá.",
            "Vamos a la playa este fin de semana.",
            "La educación es la clave del éxito.",
            "¡La película fue absolutamente fantástica!",
            "Disfruto cocinando comida italiana.",
            "La programación puede ser desafiante pero gratificante.",
            "Las montañas se ven impresionantes en invierno.",
            "La música une a personas de todas las culturas.",
            "Leer ayuda a mejorar tu vocabulario e imaginación.",
        ],
        'French': [
            "Bonjour, comment allez-vous aujourd'hui?",
            "Le temps est magnifique ce matin.",
            "J'adore lire des livres et apprendre de nouvelles choses.",
            "La technologie change notre monde rapidement.",
            "Rendez-vous demain au café.",
            "Elle joue très bien du piano.",
            "Le chat dort sur le canapé.",
            "Nous allons à la plage ce week-end.",
            "L'éducation est la clé du succès.",
            "Le film était absolument fantastique!",
            "J'aime cuisiner de la nourriture italienne.",
            "La programmation peut être difficile mais gratifiante.",
            "Les montagnes sont magnifiques en hiver.",
            "La musique rassemble les gens de toutes les cultures.",
            "La lecture aide à améliorer votre vocabulaire et imagination.",
        ],
        'German': [
            "Hallo, wie geht es dir heute?",
            "Das Wetter ist heute Morgen wunderschön.",
            "Ich liebe es, Bücher zu lesen und neue Dinge zu lernen.",
            "Technologie verändert unsere Welt rasant.",
            "Lass uns morgen im Café treffen.",
            "Sie spielt sehr gut Klavier.",
            "Die Katze schläft auf dem Sofa.",
            "Wir gehen dieses Wochenende zum Strand.",
            "Bildung ist der Schlüssel zum Erfolg.",
            "Der Film war absolut fantastisch!",
            "Ich koche gerne italienisches Essen.",
            "Programmieren kann herausfordernd aber lohnend sein.",
            "Die Berge sehen im Winter atemberaubend aus.",
            "Musik bringt Menschen aus allen Kulturen zusammen.",
            "Lesen hilft, Ihren Wortschatz und Ihre Vorstellungskraft zu verbessern.",
        ],
        'Italian': [
            "Ciao, come stai oggi?",
            "Il tempo è bellissimo questa mattina.",
            "Amo leggere libri e imparare cose nuove.",
            "La tecnologia sta cambiando il nostro mondo rapidamente.",
            "Ci vediamo domani al bar.",
            "Suona il pianoforte molto bene.",
            "Il gatto sta dormendo sul divano.",
            "Andiamo in spiaggia questo fine settimana.",
            "L'istruzione è la chiave del successo.",
            "Il film era assolutamente fantastico!",
            "Mi piace cucinare cibo italiano.",
            "La programmazione può essere impegnativa ma gratificante.",
            "Le montagne sembrano stupende in inverno.",
            "La musica unisce le persone di tutte le culture.",
            "Leggere aiuta a migliorare il tuo vocabolario e immaginazione.",
        ],
        'Portuguese': [
            "Olá, como você está hoje?",
            "O tempo está lindo esta manhã.",
            "Eu amo ler livros e aprender coisas novas.",
            "A tecnologia está mudando nosso mundo rapidamente.",
            "Vamos nos encontrar amanhã no café.",
            "Ela toca piano muito bem.",
            "O gato está dormindo no sofá.",
            "Vamos à praia neste fim de semana.",
            "A educação é a chave para o sucesso.",
            "O filme foi absolutamente fantástico!",
            "Eu gosto de cozinhar comida italiana.",
            "Programação pode ser desafiadora mas gratificante.",
            "As montanhas parecem deslumbrantes no inverno.",
            "A música une pessoas de todas as culturas.",
            "Ler ajuda a melhorar seu vocabulário e imaginação.",
        ],
    }

    # Create dataset
    texts = []
    languages = []

    for lang, sentences in data.items():
        texts.extend(sentences)
        languages.extend([lang] * len(sentences))

    return texts, languages

def create_visualizations(y_true, y_pred, languages, conf_matrix, top_features_dict):
    """Create visualizations for language detection"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=languages, yticklabels=languages,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_xlabel('Predicted Language')
    ax1.set_ylabel('True Language')
    ax1.set_title('Confusion Matrix - Language Detection', fontsize=12, fontweight='bold')

    # 2. Accuracy by language
    ax2 = fig.add_subplot(gs[0, 2])
    accuracies = []
    for lang in languages:
        mask = np.array(y_true) == lang
        if mask.sum() > 0:
            acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)

    colors = plt.cm.Set3(range(len(languages)))
    bars = ax2.barh(range(len(languages)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(languages)))
    ax2.set_yticklabels(languages)
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Accuracy by Language', fontsize=10, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)

    # 3. Language distribution
    ax3 = fig.add_subplot(gs[1, 2])
    lang_counts = pd.Series(y_true).value_counts()
    ax3.pie(lang_counts.values, labels=lang_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Language Distribution', fontsize=10, fontweight='bold')

    # 4. Top features for each language (showing 3 languages)
    for idx, lang in enumerate(languages[:3]):
        ax = fig.add_subplot(gs[2, idx])
        if lang in top_features_dict:
            features = top_features_dict[lang]
            feature_names = [f[0] for f in features[:8]]
            feature_scores = [np.exp(f[1]) for f in features[:8]]  # Convert log prob to prob

            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, feature_scores, color=colors[idx], alpha=0.7, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names, fontsize=8)
            ax.set_xlabel('Probability')
            ax.set_title(f'Top N-grams: {lang}', fontsize=10, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

    plt.savefig('language_detection_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'language_detection_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Language Detection - Kaggle NLP Solution")
    print("=" * 60)

    # Generate dataset
    print("\n1. Generating Multilingual Dataset...")
    texts, languages_list = generate_language_dataset()
    unique_languages = sorted(list(set(languages_list)))

    print(f"   - Total samples: {len(texts)}")
    print(f"   - Languages: {', '.join(unique_languages)}")
    print(f"   - Samples per language: {len(texts) // len(unique_languages)}")

    # Split dataset
    print("\n2. Splitting Dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, languages_list, test_size=0.25, random_state=42, stratify=languages_list
    )
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")

    # Train language detector
    print("\n3. Training Language Detector...")
    detector = LanguageDetector(ngram_range=(1, 3))
    detector.fit(X_train, y_train)

    print(f"   - Feature dimension: {len(detector.vectorizer.get_feature_names_out())}")
    print(f"   - Character n-gram range: 1-3")

    # Make predictions
    print("\n4. Evaluating Model...")
    y_pred = detector.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"   - Overall Accuracy: {accuracy:.2%}")

    # Detailed metrics
    print("\n5. Classification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, target_names=unique_languages)
    print(report)

    # Cross-validation
    print("\n6. Cross-Validation Scores:")
    print("-" * 60)
    cv_scores = cross_val_score(
        MultinomialNB(alpha=0.1),
        detector.vectorizer.transform(X_train),
        y_train,
        cv=5
    )
    print(f"   - CV Scores: {cv_scores}")
    print(f"   - Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_languages)

    # Top features for each language
    print("\n7. Top Character N-grams by Language:")
    print("-" * 60)
    top_features_dict = {}

    for lang in unique_languages:
        top_features = detector.get_top_features(lang, n=8)
        top_features_dict[lang] = top_features
        print(f"\n{lang}:")
        for feature, score in top_features[:5]:
            print(f"   '{feature}': {np.exp(score):.4f}")

    # Create visualizations
    print("\n8. Creating Visualizations...")
    create_visualizations(y_test, y_pred, unique_languages, conf_matrix, top_features_dict)

    # Interactive demo
    print("\n9. Interactive Demo:")
    print("-" * 60)

    test_samples = [
        "Hello, how are you?",
        "Bonjour, comment ça va?",
        "Hola, ¿cómo estás?",
        "Ciao, come stai?",
        "Hallo, wie geht's?",
        "Olá, como vai?",
    ]

    for text in test_samples:
        pred_lang = detector.predict([text])[0]
        proba = detector.predict_proba([text])[0]
        confidence = proba.max()

        print(f"\nText: '{text}'")
        print(f"Detected: {pred_lang} (confidence: {confidence:.2%})")

        # Show top 3 predictions
        top_3_idx = np.argsort(proba)[-3:][::-1]
        print("Top 3 predictions:")
        for idx in top_3_idx:
            print(f"   {detector.languages[idx]}: {proba[idx]:.2%}")

    # Performance summary
    print("\n10. Performance Summary:")
    print("-" * 60)
    print(f"   - Total accuracy: {accuracy:.2%}")
    print(f"   - Cross-validation: {cv_scores.mean():.2%}")
    print(f"   - Languages supported: {len(unique_languages)}")
    print(f"   - Character n-grams used: {len(detector.vectorizer.get_feature_names_out())}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
