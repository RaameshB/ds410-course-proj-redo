from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, regexp_replace, size, monotonically_increasing_id
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
import os

# Spark Safe Initialization
import os

BASE_TMP = "/Users/steven/Documents/spark_safe_tmp"
os.makedirs(BASE_TMP, exist_ok=True)

os.environ["SPARK_LOCAL_DIRS"] = BASE_TMP
os.environ["TMPDIR"] = BASE_TMP
os.environ["TEMP"] = BASE_TMP
os.environ["TMP"] = BASE_TMP

spark = (
    SparkSession.builder
    .appName("TXT_TOKENIZE_TFIDF_ONLY")
    .config("spark.local.dir", BASE_TMP)
    .config("spark.sql.warehouse.dir", BASE_TMP)
    .config("spark.sql.artifact.dir", BASE_TMP)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Example Input DataFrame (NO DISK LOADING)
data = [
    ("Machine learning is used in data science for prediction and classification.",),
    ("Cybersecurity protects systems and networks from digital attacks.",),
    ("Large language models are trained on massive datasets of text.",),
    ("Deep learning uses neural networks to learn complex patterns.",),
]

df = spark.createDataFrame(data, ["raw_text"])

# Remove very short lines
df = df.filter(length(col("raw_text")) > 20)

# Add document ID
df = df.withColumn("doc_id", monotonically_increasing_id())

# TEXT CLEANING (FOR TOKENIZATION)

# Keep only letters, numbers, and spaces
# This removes punctuation and symbols before tokenization
df_clean = df.withColumn(
    "intermidiate",
    regexp_replace(col("raw_text"), r"[^\w\s]", " ")
)

# TOKENIZATION 

# Step 1: Tokenizer
# - Takes one full sentence from 'intermidiate'
# - Splits it into individual words based on whitespace
tokenizer = Tokenizer(
    inputCol="intermidiate",   # Cleaned text column
    outputCol="words"          # Output column containing word list
)

df_tokens = tokenizer.transform(df_clean)

# Step 2: StopWords Removal
# - Removes common useless words like: the, is, and, for, to, etc.
remover = StopWordsRemover(
    inputCol="words",          # Input is the token list
    outputCol="filtered_words" # Output is cleaned token list
)

df_filtered = remover.transform(df_tokens)

# Step 3: Remove rows with too few meaningful words
df_filtered = df_filtered.filter(size(col("filtered_words")) >= 3)

print("\nTokenized Output:")
df_filtered.select("raw_text", "filtered_words").show(truncate=False)

# VECTORIZATION

cv = CountVectorizer(
    inputCol="filtered_words",
    outputCol="raw_features",
    vocabSize=1000,
    minDF=1
)

cv_model = cv.fit(df_filtered)
df_vectorized = cv_model.transform(df_filtered)

vocab = cv_model.vocabulary
print(f"\nVocabulary Size: {len(vocab)}")


# TF-IDF

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_vectorized)
df_tfidf = idf_model.transform(df_vectorized)

print("\nTF-IDF Completed")

# Save Outputs (Optional)

OUTPUT_DIR = "/Users/steven/Documents/test_txt_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_tfidf.select("doc_id", "filtered_words", "features") \
    .write.mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "tfidf_features"))

with open(os.path.join(OUTPUT_DIR, "vocabulary.txt"), "w") as f:
    for word in vocab:
        f.write(word + "\n")

print("\nALL PROCESSING COMPLETE (NO DISK LOADING, TOKENIZATION COMMENTED)")

spark.stop()
