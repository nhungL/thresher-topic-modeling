import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pyLDAvis
import pyLDAvis.lda_model
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import webbrowser

# Custom stopword list for school newspapers
custom_stopwords = set([
    'student', 'school', 'university', 'college', 'campus', 'class', 'teacher',
    'professor', 'faculty', 'staff', 'member', 'meeting', 'event', 'organization',
    'club', 'group', 'session', 'discussion', 'activity', 'program', 'day', 'week',
    'year', 'semester', 'term', 'today', 'tomorrow', 'yesterday', 
    'hall', 'room', 'building', 'home', 'office', 'area', 'have', 'has', 'had',
    'is', 'are', 'was', 'were', 'be', 'being', 'been', 'will', 'would', 'can',
    'could', 'should', 'may', 'might', 'also', 'another', 'any', 'all', 'each',
    'every', 'some', 'one', 'two', 'three', 'first', 'second', 'third', 'many',
    'most', 'other', 'such', 'including'
])

stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS).union(custom_stopwords)

def read_files():
    file_list = os.listdir('data')

    # Initialize an empty list to store all data
    all_data = []

    # Iterate over all files in the directory
    for file_name in file_list:
        # Create the full file path
        file_path = os.path.join('data', file_name)
        
        # Check if the file is a text file
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
                number = file_name[3:11]
                all_data.append((content, number))

    # Convert the list into a DataFrame
    df = pd.DataFrame(all_data, columns=['Content', 'Date'])
    return df

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def process_and_analyze(n):
    df = read_files()
    df['Clean_Content'] = df['Content'].apply(lambda x: clean_text(x))
    print(df)

    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=3000, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Clean_Content'])

    # LDA Model
    num_topics = n  # Choose the number of topics
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)

    # Extract topics
    def display_topics(model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic_terms = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topics.append(topic_terms)
            print(f"Topic {topic_idx}: {' '.join(topic_terms)}")
        return topics

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = display_topics(lda, tfidf_feature_names, 10)

    # Create a Gensim dictionary from the clean content
    dictionary = Dictionary(df['Clean_Content'].apply(lambda x: x.split()).tolist())

    # Calculate coherence score
    coherence_model = CoherenceModel(topics=topics, texts=df['Clean_Content'].apply(lambda x: x.split()).tolist(), dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    
    return coherence_score

    # # pyLDAvis.enable_notebook()
    # panel = pyLDAvis.lda_model.prepare(lda, tfidf_matrix, tfidf_vectorizer, mds='tsne')
    # # pyLDAvis.display(panel)
    # pyLDAvis.save_html(panel, 'lda_visualization.html')
    # webbrowser.open('lda_visualization.html')

    # # Analyze trends over time
    # df['Topic'] = lda.transform(tfidf_matrix).argmax(axis=1)
    # df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    # # Group by date and calculate the mean topic distribution
    # topic_trends = df.groupby(df['Date'].dt.to_period('M')).Topic.value_counts(normalize=True).unstack().fillna(0)

    # # Plot the trends
    # topic_trends.plot(kind='line', figsize=(12, 8))
    # plt.title('Topic Trends Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Proportion of Documents')
    # plt.legend(title='Topic')
    # plt.show()

if __name__ == "__main__":
    # scores = []
    # for i in range(2, 10):
    #     print(i)
    #     score = process_and_analyze(i)
    #     scores.append(score)
    # plt.plot(range(2, 10), scores)
    # plt.xlabel("Number of Topics")
    # plt.ylabel("Coherence Score")
    # plt.title("Coherence Scores by Number of Topics")
    # plt.show()
        
    process_and_analyze(3)

    # Visualize the topics
    
