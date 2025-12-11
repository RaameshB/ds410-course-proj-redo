#!/usr/bin/env python
# coding: utf-8

# In[1]:


from util.sparkhandler import SparkHandler
from util.dataproctools import get_extracted_wet, save_rdd, load_rdd
from util.dataproctools import extracted_wet_to_df
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline
from util.updatedVIZ import visualize_topics
import optuna

# In[2]:


print("\n\n========================================")
print("                      STAGE ONE")
print("========================================\n\n")



handler = SparkHandler(available_cores=8, driver_mem = 35, executor_mem = 35, mem_overhead = 35, cluster=True) # look at the parameters for SparkHandler to increase max memory
ss = handler.get_spark_session()
sc = handler.get_spark_context()

# In[3]:


raw_data = get_extracted_wet(spark_context=sc, approx_sample_size=200000, wet_paths_file="data/wet.paths")
save_rdd(raw_data, "saved_intermediates/rawStrRDD", overwrite=True)

# In[4]:


# raw_data = load_rdd(spark_context=sc, path_to_load="../saved_intermediates/rawStrRDD")

# In[5]:


df = extracted_wet_to_df(spark_session=ss, extracted_wet_rdd=raw_data)
filtered = df.filter(((df.tld == '.gov') | (df.tld == '.edu')) & (df.languages == 'eng'))
filtered.write.mode("overwrite").format("json").save("saved_intermediates/filtered_dataframe")

# In[6]:


loaded_df = ss.read.json("saved_intermediates/filtered_dataframe")

# In[7]:


# loaded_df.head()

# In[9]:




print("\n\n========================================")
print("                      STAGE TWO")
print("========================================\n\n")


tuning_sample = loaded_df.sample(0.2)
train, val, test = tuning_sample.randomSplit([0.7, 0.2, 0.1], seed=1237)

# In[10]:


def objective(trial: optuna.trial.Trial):

    minTokenLength = trial.suggest_int("min word len", 1, 3)
    k = 12
    learningDecay = trial.suggest_float("lr decay", 0.5, 1.0)
    learningOffset = trial.suggest_float("offset", 0, 10)
    maxItr = trial.suggest_int("intrs", 10, 100)
    subsamplingRate = trial.suggest_float("subsampling rate", 0, 1)

    # 1) Tokenize text into tokens
    tokenizer = RegexTokenizer(minTokenLength=minTokenLength, gaps=False, pattern=r"\b[a-zA-Z]+[\d]*(?:[-'][a-zA-Z]+[\d]*)*\b", inputCol="raw_content", outputCol="tokenized")

    # 2) Remove stopwords
    remover = StopWordsRemover(inputCol="tokenized", outputCol="filtered")

    # 3) Convert tokens to term-frequency vectors
    cv = CountVectorizer(
        inputCol="filtered",
        outputCol="features",
        vocabSize=5000,
        minDF=1    # keep terms that appear in at least 1 document
    )

    # 4) LDA model (k = number of topics)
    lda = LDA(
        k=k,
        learningDecay=learningDecay,
        learningOffset=learningOffset,
        maxIter=maxItr,
        subsamplingRate=subsamplingRate,
        featuresCol="features"
    )

    # 5) Build pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, cv, lda])
    model = pipeline.fit(train)
    valed = model.transform(val)
    perplexity = model.stages[-1].logPerplexity(valed)
    log_likelyhood = model.stages[-1].logLikelihood(valed)
    return perplexity, log_likelyhood

# In[11]:


study = optuna.create_study(directions=["minimize", "maximize"])

# In[12]:


study.optimize(objective, n_trials=200, timeout=480, n_jobs=20)

# In[13]:


best_params = study.best_trials[-1].params
# best_params

# In[14]:


#def objective(trial: optuna.trial.Trial):

minTokenLength = best_params['min word len']
k = 12
learningDecay = best_params['lr decay']
learningOffset = best_params['offset']
maxItr = best_params['intrs']
subsamplingRate = best_params['subsampling rate']



print("\n\n========================================")
print("                      STAGE THREE")
print("========================================\n\n")

# 1) Tokenize text into tokens
tokenizer = RegexTokenizer(minTokenLength=minTokenLength, gaps=False, pattern=r"\b[a-zA-Z]+[\d]*(?:[-'][a-zA-Z]+[\d]*)*\b", inputCol="raw_content", outputCol="tokenized")

# 2) Remove stopwords
remover = StopWordsRemover(inputCol="tokenized", outputCol="filtered")

# 3) Convert tokens to term-frequency vectors
cv = CountVectorizer(
    inputCol="filtered",
    outputCol="features",
    vocabSize=5000,
    minDF=1    # keep terms that appear in at least 1 document
)

# 4) LDA model (k = number of topics)
lda = LDA(
    k=k,
    learningDecay=learningDecay,
    learningOffset=learningOffset,
    maxIter=maxItr,
    subsamplingRate=subsamplingRate,
    featuresCol="features"
)

# 5) Build pipeline
pipeline = Pipeline(stages=[tokenizer, remover, cv, lda])
model = pipeline.fit(loaded_df)
transformed = model.transform(loaded_df)

# In[15]:


# Extract the CV and LDA sub-models
cv_model = model.stages[2]
lda_model = model.stages[3]

vocab = cv_model.vocabulary

# In[19]:


visualize_topics(lda_model, vocab, transformed, num_words=10)

# In[23]:


optuna.visualization.plot_pareto_front(study, target_names=["perplexity", "log likelyhood"]).write_html("visualization_outputs/pareto_front.html")
optuna.visualization.plot_param_importances(study).write_html("visualization_outputs/param_importances.html")

# In[ ]:


ss.stop()
