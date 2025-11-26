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
        
def analyze_sentiment(text):
    """
    Simple rule-based sentiment analysis for Indonesian text
    
    Args:
        text (str): Processed text (after stemming)
        
    Returns:
        str: 'Positif', 'Negatif', atau 'Netral'
    """
    # Kata positif bahasa Indonesia
    positive_words = {
        'bagus', 'baik', 'puas', 'senang', 'suka', 'mantap', 'oke', 
        'recommended', 'cepat', 'murah', 'kualitas', 'ori', 'original',
        'rapih', 'rapi', 'sesuai', 'terima', 'kasih', 'thanks', 'memuaskan',
        'awet', 'best', 'terbaik', 'perfect', 'sempurna', 'lengkap',
        'helpful', 'ramah', 'ResponsIf', 'fast', 'aman'
    }
    
    # Kata negatif bahasa Indonesia
    negative_words = {
        'buruk', 'jelek', 'kecewa', 'rusak', 'palsu', 'lambat', 'lama',
        'mahal', 'tidak', 'bukan', 'jangan', 'salah', 'error', 'cacat',
        'pecah', 'bocor', 'busuk', 'bau', 'kotor', 'lecet', 'penyok',
        'komplain', 'tolak', 'hangus', 'gagal', 'cancel', 'bohong',
        'tipu', 'kecil', 'beda', 'gak', 'enggak', 'jujur'
    }
    
    # Hitung skor sentiment
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Tentukan sentiment
    if positive_count > negative_count:
        return 'Positif'
    elif negative_count > positive_count:
        return 'Negatif'
    else:
        return 'Netral'


def get_sentiment_statistics(preprocessed_results):
    """
    Get sentiment statistics from preprocessed results
    
    Args:
        preprocessed_results (list): Results from preprocess_reviews()
        
    Returns:
        dict: Sentiment statistics
            - total: total reviews
            - positive: count of positive reviews
            - negative: count of negative reviews
            - neutral: count of neutral reviews
            - positive_percentage: percentage of positive
            - negative_percentage: percentage of negative
            - neutral_percentage: percentage of neutral
    """
    sentiments = []
    
    # Analyze sentiment untuk setiap review
    for result in preprocessed_results:
        sentiment = analyze_sentiment(result['final_text'])
        sentiments.append(sentiment)
    
    # Hitung statistik
    total = len(sentiments)
    positive = sentiments.count('Positif')
    negative = sentiments.count('Negatif')
    neutral = sentiments.count('Netral')
    
    # Hitung persentase
    positive_pct = round((positive / total * 100), 1) if total > 0 else 0
    negative_pct = round((negative / total * 100), 1) if total > 0 else 0
    neutral_pct = round((neutral / total * 100), 1) if total > 0 else 0
    
    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'positive_percentage': positive_pct,
        'negative_percentage': negative_pct,
        'neutral_percentage': neutral_pct,
        'sentiments': sentiments  # List sentiment per review
    }
    
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

    # Sentiment Analysis

    print("\n" + "=" * 70)
    print("SENTIMENT STATISTICS")
    print("=" * 70)
    sentiment_stats = get_sentiment_statistics(results)
    print(f"Total Reviews: {sentiment_stats['total']}")
    print(f"Positive: {sentiment_stats['positive']} ({sentiment_stats['positive_percentage']}%)")
    print(f"Negative: {sentiment_stats['negative']} ({sentiment_stats['negative_percentage']}%)")
    print(f"Neutral: {sentiment_stats['neutral']} ({sentiment_stats['neutral_percentage']}%)")
    
    # Text preprocessing statistics
    print("\n" + "=" * 70)
    print("TEXT PREPROCESSING STATISTICS")
    print("=" * 70)
    stats = get_statistics(results)
    print(f"Total reviews: {stats['total_reviews']}")
    print(f"Avg tokens (original): {stats['avg_tokens_original']}")
    print(f"Avg tokens (final): {stats['avg_tokens_final']}")
    print(f"Reduction: {stats['reduction_rate']}%")
    
    # Create DataFrame
    print("\n" + "=" * 70)
    print("RESULTS DATAFRAME")
    print("=" * 70)
    df = create_sentiment_dataframe(results)
    print(df.to_string(index=False))
    
    # Word frequency
    print("\n" + "=" * 70)
    print("TOP 10 MOST COMMON WORDS")
    print("=" * 70)
    word_freq = get_word_frequency(results, top_n=10)
    for word, count in word_freq:
        print(f"{word}: {count}")
