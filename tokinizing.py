from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, regexp_replace, size, monotonically_increasing_id
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA
import os


spark = (
    SparkSession.builder
    .appName("TXT_LDA_TEST")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


TXT_PATH = "/Users/steven/Documents/test_txt.txt"
OUTPUT_DIR = "/Users/steven/Documents/test_txt_output"

df = spark.read.text(TXT_PATH).withColumnRenamed("value", "raw_text")

df = df.filter(~col("raw_text").startswith("{\\rtf"))

print("Loaded TXT documents:")
df.show(truncate=False)


df = df.filter(length(col("raw_text")) > 20)

df = df.withColumn("doc_id", monotonically_increasing_id())

df = df.withColumn(
    "raw_text",
    regexp_replace(col("raw_text"), r"\\\\[a-zA-Z0-9]+", " ")
)

df_clean = df.withColumn(
    "clean_text",
    regexp_replace(col("raw_text"), r"[^\w\s]", " ")
)


tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
df_tokens = tokenizer.transform(df_clean)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokens)

df_filtered = df_filtered.filter(size(col("filtered_words")) >= 3)

print("\nTokenized Sample:")
df_filtered.select("filtered_words").show(truncate=False)


MAX_VOCAB_SIZE = 1000
MIN_DOC_FREQ = 1

cv = CountVectorizer(
    inputCol="filtered_words",
    outputCol="raw_features",
    vocabSize=MAX_VOCAB_SIZE,
    minDF=MIN_DOC_FREQ
)

cv_model = cv.fit(df_filtered)
df_vectorized = cv_model.transform(df_filtered)

vocab = cv_model.vocabulary
print(f"\nVocabulary Size: {len(vocab)}")



idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_vectorized)
df_tfidf = idf_model.transform(df_vectorized)

print("\nTF-IDF completed")



NUM_TOPICS = 3
MAX_ITER = 10

lda = LDA(
    k=NUM_TOPICS,
    maxIter=MAX_ITER,
    optimizer="online",
    featuresCol="features",
    seed=42
)

print("\nTraining LDA...")
lda_model = lda.fit(df_tfidf)

ll = lda_model.logLikelihood(df_tfidf)
lp = lda_model.logPerplexity(df_tfidf)

print(f"\nLog Likelihood: {ll}")
print(f"Log Perplexity: {lp}")



print("\n TOPICS \n")

topics_df = lda_model.describeTopics(maxTermsPerTopic=10)
topics = topics_df.collect()

for idx, topic in enumerate(topics):
    print(f"Topic {idx}:")
    terms = [vocab[i] for i in topic.termIndices]
    for term, weight in zip(terms, topic.termWeights):
        print(f"  {term:20s} {weight:.4f}")
    print()


os.makedirs(OUTPUT_DIR, exist_ok=True)

df_topics = lda_model.transform(df_tfidf)

model_path = os.path.join(OUTPUT_DIR, "lda_model")
topics_path = os.path.join(OUTPUT_DIR, "topics")
doc_topics_path = os.path.join(OUTPUT_DIR, "document_topics")
vocab_path = os.path.join(OUTPUT_DIR, "vocabulary.txt")

lda_model.write().overwrite().save(model_path)

topics_df.write.mode("overwrite").parquet(topics_path)

df_topics.select("doc_id", "topicDistribution") \
    .write.mode("overwrite").parquet(doc_topics_path)

with open(vocab_path, "w") as f:
    for word in vocab:
        f.write(word + "\n")

print("\nALL PROCESSING COMPLETE")
print("Results saved to:", OUTPUT_DIR)

spark.stop()