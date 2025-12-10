#here we are just importing neccesary packages.
import matplotlib.pyplot as plt
import pandas as pd


# We allow for the use of either Wordcloud or Seaborn for visualization
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


from pyspark.ml.linalg import DenseVector, SparseVector


#here we create a function to create visualzations based on the results we get from LDA
def visualize_topics(lda_model, vocabulary, df_transformed=None, num_words=10):

    topics_df = lda_model.describeTopics(maxTermsPerTopic=num_words)
    topics = topics_df.collect()

  
    topic_labels = {}
    for row in topics:
        top_terms = [vocabulary[i] for i in row.termIndices[:3]]  # this is giving the top 3 words
        label = f"Topic {row.topic} — ({', '.join(top_terms)})"
        topic_labels[row.topic] = label

    
    
    # Here we are creating our first visualzation: Bar Charts. 
    num_topics = len(topics)
    fig, axes = plt.subplots(1, num_topics, figsize=(5 * num_topics, 6))
    if num_topics == 1:
        axes = [axes]

    for idx, topic_row in enumerate(topics):
        term_indices = topic_row.termIndices
        term_weights = topic_row.termWeights
        words = [vocabulary[i] for i in term_indices]
        weights = [float(w) for w in term_weights]

        ax = axes[idx]
        ax.barh(words, weights)
        ax.set_title(topic_labels[topic_row.topic])
        ax.set_xlabel("Weight")
        ax.invert_yaxis() # we do this so that the highest weight is at the top.

    plt.tight_layout()
    plt.show()

    # Here we are creating our second  visualzation:Word Clouds. 
    if HAS_WORDCLOUD:
        fig, axes = plt.subplots(1, num_topics, figsize=(5 * num_topics, 4))
        if num_topics == 1:
            axes = [axes]

        for idx, topic_row in enumerate(topics):
            term_indices = topic_row.termIndices
            term_weights = topic_row.termWeights
            word_freq = {vocabulary[i]: float(w) for i, w in zip(term_indices, term_weights)}

            wordcloud = WordCloud(width=425, height=325, background_color="white")                 .generate_from_frequencies(word_freq)

            axes[idx].imshow(wordcloud, interpolation="bilinear")
            axes[idx].set_title(topic_labels[topic_row.topic])
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()

    # Here we are creating our third visualzation: Document Topic Distributions. 
    if df_transformed is not None:
        
     # we want to extract the topic distribution column from the df which is containing the vector of probabilites   
        rows = df_transformed.select("topicDistribution").collect()
        topic_probs = []
        for row in rows:
            dist = row.topicDistribution
            
            if isinstance(dist, (DenseVector, SparseVector)):
                dist = dist.toArray()
            topic_probs.append([float(x) for x in dist])

        df_topics = pd.DataFrame(
            topic_probs,
            columns=[topic_labels[i] for i in range(num_topics)]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        topic_means = df_topics.mean()
        ax.bar(topic_means.index, topic_means.values)
        ax.set_ylabel("Average Probability")
        ax.set_title("Average Topic Proportions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if HAS_SEABORN:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_topics.T, cmap="YlOrRd", cbar_kws={"label": "Probability"}, ax=ax)
            ax.set_xlabel("Document")
            ax.set_ylabel("Topic")
            ax.set_title("Document-Topic Distribution")
            plt.tight_layout()
            plt.show()



# Here we are calling the visualizations. Uncomment when you want to employ it.
# visualize_topics(lda_model, vocab, transformed, num_words=10)






