import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import io


from preprocessing import preprocess_reviews, get_statistics, get_sentiment_statistics

st.set_page_config(
    page_title="Analisis Topik Ulasan E-commerce",
    layout="wide"
)

st.title("Analisis Topik Ulasan Produk E-commerce")
st.markdown("**Preprocessing Text menggunakan NLP**")
st.divider()

with st.sidebar:
    st.header("Pengaturan")
    input_method = st.radio(
        "Pilih metode input:",
        ["Upload CSV", "Input Manual"]
    )
    
    st.divider()
    st.info("**Dibuat oleh:** Zahfandhika Fauzan Maldini\n\n**Mata Kuliah:** NLP")

if input_method == "Upload CSV":
    st.subheader("Upload File CSV")
    st.write("File CSV harus memiliki kolom bernama **'review'**")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=['csv'],
        help="Upload file CSV dengan kolom 'review'"
    )
    
    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")

            df = pd.read_csv(
                 io.StringIO(content),
                 names=["review"],       
                 header=0,                
                 engine="python",
                 quoting=3,             
                 on_bad_lines="skip"    
            )
            
            if 'review' not in df.columns:
                st.error("File CSV harus memiliki kolom 'review'!")
            else:
                st.success(f"Berhasil memuat {len(df)} ulasan")
                
                with st.expander("Preview Data"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                reviews = df['review'].dropna().tolist()
                
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            reviews = None
    else:
        reviews = None

else:  
    st.subheader("Input Ulasan Manual")
    
    num_reviews = st.number_input(
        "Jumlah ulasan:",
        min_value=1,
        max_value=10,
        value=3
    )
    
    reviews = []
    for i in range(num_reviews):
        review = st.text_area(
            f"Ulasan {i+1}:",
            key=f"review_{i}",
            height=80
        )
        if review:
            reviews.append(review)

if reviews and len(reviews) > 0:
    
    st.divider()
    
    if st.button("Proses Data", type="primary"):
        
        with st.spinner("Memproses data..."):
           
            results = preprocess_reviews(reviews)
            stats = get_statistics(results)
        
        st.success("Preprocessing selesai!")
        
        tab1, tab2, tab3, tab4, tab5= st.tabs([
            "Statistik",
            "Detail Preprocessing",
            "Word Cloud",
            "Analisis Topik",
            "Analysis Sentiment"
        ])
        
        with tab1:
            st.subheader("Statistik Preprocessing")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Ulasan", stats['total_reviews'])
            with col2:
                st.metric("Rata-rata Token Awal", stats['avg_tokens_original'])
            with col3:
                st.metric("Rata-rata Token Akhir", stats['avg_tokens_final'])
            with col4:
                st.metric("Efisiensi", f"{stats['reduction_rate']}%")
            
            st.divider()
            
            comparison_data = []
            for i, result in enumerate(results):
                comparison_data.append({
                    'Ulasan': f'Ulasan {i+1}',
                    'Original': len(result['tokens']),
                    'Filtered': len(result['filtered']),
                    'Stemmed': len(result['stemmed'])
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                df_comparison,
                x='Ulasan',
                y=['Original', 'Filtered', 'Stemmed'],
                title='Perbandingan Jumlah Token per Tahap',
                barmode='group',
                color_discrete_map={
                    'Original': '#ff6b6b',
                    'Filtered': '#4ecdc4',
                    'Stemmed': '#45b7d1'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Detail Preprocessing per Ulasan")
            
            for i, result in enumerate(results):
                with st.expander(f"Ulasan {i+1}"):
                    st.write("**1. Original Text:**")
                    st.info(result['original'])
                    
                    st.write(f"**2. Tokenization** ({len(result['tokens'])} tokens):")
                    st.code(result['tokens'])
                    
                    st.write(f"**3. Stopword Removal** ({len(result['filtered'])} tokens):")
                    st.code(result['filtered'])
                    
                    st.write(f"**4. Stemming** ({len(result['stemmed'])} tokens):")
                    st.code(result['stemmed'])
                    
                    st.write("**5. Final Text:**")
                    st.success(result['final_text'])
        
        
        with tab3:
            st.subheader("☁️ Word Cloud")
            
            
            all_words = []
            for result in results:
                all_words.extend(result['stemmed'])
                
            if all_words:
                 wordcloud = WordCloud(
                     width=800,
                     height=400,
                     background_color='white',
                     colormap='viridis'
                 ).generate(' '.join(all_words))
                 
                 fig, ax = plt.subplots(figsize=(12, 6))
                 ax.imshow(wordcloud, interpolation='bilinear')
                 ax.axis('off')
                 st.pyplot(fig)
                 
                 import io
                 buf = io.BytesIO()
                 fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                 buf.seek(0)
                 
                 st.download_button(
                     label="💾 Download Word Cloud",
                     data=buf,
                     file_name="wordcloud.png",
                     mime="image/png"
                 )
            else:
                st.warning("Tidak ada data untuk membuat word cloud")
        
        with tab4:
            st.subheader("Analisis Topik")
            
            all_words = []
            for result in results:
                all_words.extend(result['stemmed'])
            
            word_freq = Counter(all_words)
            top_20 = word_freq.most_common(20)
            
            if top_20:
                
                words, counts = zip(*top_20)
                
                fig = go.Figure(go.Bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    marker=dict(
                        color=counts,
                        colorscale='Blues',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title='Top 20 Kata Paling Sering Muncul',
                    xaxis_title='Frekuensi',
                    yaxis_title='Kata',
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Tabel Frekuensi Kata")
                df_freq = pd.DataFrame(top_20, columns=['Kata', 'Frekuensi'])
                st.dataframe(df_freq, use_container_width=True)
                
                csv = df_freq.to_csv(index=False)
                st.download_button(
                    "Download Hasil Analisis (CSV)",
                    data=csv,
                    file_name="analisis_topik.csv",
                    mime="text/csv"
                )
        
        with tab5:
            st.subheader("Sentiment Analysis")
            sentiment_stats = get_sentiment_statistics(results)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Positif",
                    f"{sentiment_stats['positive']} ulasan",       
                    f"{sentiment_stats['positive_percentage']}%"
                )
            
            with col2:
                st.metric(
                    "Netral",
                    f"{sentiment_stats['neutral']} ulasan",       
                    f"{sentiment_stats['neutral_percentage']}%"
                )
                
            with col3:
                st.metric(
                    "Negatif",
                    f"{sentiment_stats['negative']} ulasan",       
                    f"{sentiment_stats['negative_percentage']}%"
                )
                
            st.divider()
else:
    st.info("Silakan upload file CSV atau input ulasan manual untuk memulai analisis")

st.divider()
st.caption("Mini Project NLP - Analisis Topik Ulasan Produk E-commerce")
