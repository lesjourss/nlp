import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words('indonesian'))
custom_stopwords = {'yang', 'dan', 'di', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'ke', 'nya'}
stop_words.update(custom_stopwords)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    filtered = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    stemmed = [stemmer.stem(word) for word in filtered]
    
    return {
        'original': text,
        'tokens': tokens,
        'filtered': filtered,
        'stemmed': stemmed,
        'final_text': ' '.join(stemmed)
    }


def preprocess_reviews(reviews):
    results = []
    for review in reviews:
        result = preprocess_text(review)
        results.append(result)
    
    return results


def get_statistics(preprocessed_results):
    total_reviews = len(preprocessed_results)
    
    if total_reviews == 0:
        return {
            'total_reviews': 0,
            'avg_tokens_original': 0,
            'avg_tokens_final': 0,
            'reduction_rate': 0
        }
    
    avg_tokens_original = sum(len(r['tokens']) for r in preprocessed_results) / total_reviews
    avg_tokens_final = sum(len(r['stemmed']) for r in preprocessed_results) / total_reviews
    reduction_rate = ((avg_tokens_original - avg_tokens_final) / avg_tokens_original) * 100
    
    return {
        'total_reviews': total_reviews,
        'avg_tokens_original': round(avg_tokens_original, 2),
        'avg_tokens_final': round(avg_tokens_final, 2),
        'reduction_rate': round(reduction_rate, 2)
    }


def get_word_frequency(preprocessed_results, top_n=20):
    from collections import Counter
    
    all_words = []
    for result in preprocessed_results:
        all_words.extend(result['stemmed'])
    
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)


def create_comparison_dataframe(preprocessed_results):
    data = []
    for i, result in enumerate(preprocessed_results):
        data.append({
            'No': i + 1,
            'Original': result['original'][:60] + '...',
            'Token Count': len(result['tokens']),
            'After Stopword': len(result['filtered']),
            'After Stemming': len(result['stemmed']),
            'Final Text': ' '.join(result['stemmed'][:15]) + '...'
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    
    sample_reviews = [
        "produk sesuai dengan deskripsi etalase toko. kualitas produk standar.",
        "Penjual gak jujur. Ukuran kecil dipaksa dilabel in 45."
    ]
    
 
    results = preprocess_reviews(sample_reviews)
    
    for i, result in enumerate(results):
        print(f"\n=== Ulasan {i+1} ===")
        print(f"Original: {result['original']}")
        print(f"Tokens: {result['tokens'][:10]}...")
        print(f"Filtered: {result['filtered'][:10]}...")
        print(f"Stemmed: {result['stemmed'][:10]}...")
        print(f"Final: {result['final_text']}")
    

    stats = get_statistics(results)
    print(f"\n=== Statistics ===")
    print(f"Total reviews: {stats['total_reviews']}")
    print(f"Avg tokens (original): {stats['avg_tokens_original']}")
    print(f"Avg tokens (final): {stats['avg_tokens_final']}")
    print(f"Reduction: {stats['reduction_rate']}%")