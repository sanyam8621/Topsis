#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install required packages
pip install datasets
pip install sentence-transformers


# In[4]:


# Import necessary libraries
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd


# In[5]:


# Load the STS Benchmark dataset
dataset = load_dataset("mteb/stsbenchmark-sts")
train_data = dataset['train']


# In[6]:


# Convert dataset to DataFrame
train_df = pd.DataFrame(train_data)
train_df = train_df.iloc[:, 5:]


# In[7]:


# Define the new range for normalization
new_min_range = -1
new_max_range = 1 

# Calculate the original min and max values
original_min_value = train_df['score'].min()
original_max_value = train_df['score'].max()

# Normalize the 'score' column to the new range
train_df['score'] = ((train_df['score'] - original_min_value) * (new_max_range - new_min_range) / (original_max_value - original_min_value)) + new_min_range


# In[8]:


# Define models and model names
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/clip-ViT-B-32-multilingual-v1"
]

model_names = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "clip-ViT-B-32-multilingual-v1"
]


# In[9]:


# Function to evaluate models
def evaluate_models(actual_column, *predicted_columns, threshold=0.5):
    results = {}
    for predicted_column in predicted_columns:
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(train_df[actual_column], train_df[predicted_column]))

        # Convert the predicted values to binary based on the threshold
        binary_predictions = (train_df[predicted_column] >= threshold).astype(int)

        # Convert actual values to binary based on the threshold
        binary_actuals = (train_df[actual_column] >= threshold).astype(int)

        # Calculate precision, recall, and accuracy
        precision = precision_score(binary_actuals, binary_predictions)
        recall = recall_score(binary_actuals, binary_predictions)
        accuracy = accuracy_score(binary_actuals, binary_predictions)

        # Store results in the dictionary
        results[predicted_column] = {'RMSE': rmse, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

    return results


# In[10]:


# Initialize SentenceTransformer models and calculate similarity scores
for i, model_name in enumerate(model_names):
    print(model_name)
    model = SentenceTransformer(models[i])
    similarity_scores = []
    for index, row in train_df.iterrows():
        embeddings = model.encode([row['sentence1'], row['sentence2']], convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

        # Append similarity score to the list
        similarity_scores.append(cosine_similarity.item())

    train_df[f'model_{i + 1}'] = similarity_scores


# In[11]:


# Evaluate models
evaluation_results = evaluate_models('score', 'model_1', 'model_2', 'model_3', 'model_4')

# Print evaluation results
for model, metrics in evaluation_results.items():
    print(f"Metrics for {model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()


# In[12]:


# Save evaluation results to a CSV file
output_file = 'evaluation_results.csv'
df_results = pd.DataFrame.from_dict(evaluation_results, orient='index')
df_results.to_csv(output_file)
print(f"Results have been saved to '{output_file}'.")


# In[13]:


# Perform TOPSIS analysis
pip install topsis-taanisha-10210323
get_ipython().system('topsis evaluation_results.csv "1,1,1,1" "-,+,+,+" answer.csv')


# In[14]:


# Read TOPSIS result file
FinalAnswer = pd.read_csv('answer.csv')


# In[15]:


# Plot TOPSIS scores
model_names = FinalAnswer['Unnamed: 0']
topsis_scores = FinalAnswer['Topsis Score']

plt.figure(figsize=(5,5))
plt.bar(model_names, topsis_scores, color='blue')
plt.xlabel('Model Name')
plt.ylabel('Topsis Score')
plt.title('Topsis Score evaluation for different text classification models (Done by Taanisha)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()
