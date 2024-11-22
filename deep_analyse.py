import pandas as pd	
import matplotlib.pyplot as plt


df = pd.read_parquet('data/df.parquet')

#Agurpamento por similaridade com KMeans
from sklearn.cluster import KMeans
#Agrupamento por similaridade baseado e contexto 
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('tgsc/sentence-transformer-ult5-pt-small')
#model = SentenceTransformer('intfloat/multilingual-e5-large')
#model = SentenceTransformer('rufimelo/Legal-BERTimbau-large-TSDAE-v4-GPL-sts')
model = SentenceTransformer('rufimelo/Legal-BERTimbau-sts-large')
#model = SentenceTransformer('sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')


embeddings = model.encode(df['Pedido_limpo'].tolist())
df['embeddings'] = embeddings.tolist()
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)
print(df[['Pedido', 'cluster']])


from sklearn.decomposition import PCA
# Reduzir a dimensionalidade dos embeddings para 2D usando PCA
pca = PCA(n_components=2)

embeddings_2d = pca.fit_transform(embeddings)

# Plotar os clusters
plt.figure(figsize=(8, 5))
plt.scatter(embeddings_2d[:, 0][:10], embeddings_2d[:, 1][:10], c=df['cluster'][:10], cmap='viridis')
plt.colorbar()
plt.title('Clusters dos Pedidos usando PCA')
for i, texto in enumerate(df['Pedido'][:10]):
    plt.annotate(texto[:15], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)
plt.show()


k_values = range(2, 10)

#Analise de clusters shilouette
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

print (silhouette_score(embeddings, df['cluster']) )

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    print(f'Clusters: {k}, Silhouette: {silhouette_score(embeddings, df["cluster"])}')

#analise de clusters Elbow Method

metrics_kmeans = {
    'silhouette': [],
    'davies_bouldin': [],
    'elbow': [],
    'calinski_harabasz':[], 
    'bic_scores': [],
    'aic_scores': []
}
metrics_gausiano = metrics_kmeans.copy()

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    #inertia.append(kmeans.inertia_)
    metrics_kmeans['silhouette'].append(silhouette_score(embeddings, kmeans.labels_))
    metrics_kmeans['davies_bouldin'].append(davies_bouldin_score(embeddings, kmeans.labels_))
    metrics_kmeans['elbow'].append(kmeans.inertia_)
    metrics_kmeans['calinski_harabasz'].append(calinski_harabasz_score(embeddings, kmeans.labels_))

    


for metric in metrics_kmeans.keys():
    if len(metrics_kmeans[metric]) == len(k_values):
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, metrics_kmeans[metric], marker='o')
        plt.title(f'Metodo: {metric}, Kmeans ', fontsize=16)
        plt.xlabel('Número de Clusters (k)', fontsize=14)
        plt.ylabel(f'Score {metric}', fontsize=14)
        plt.xticks(k_values)
        plt.grid(True)
        plt.savefig(f'kmeans_{metric}.png')
        plt.show()


#Avaliação Intrínseca com Soft Clustering

from sklearn.mixture import GaussianMixture
for k in k_values:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    metrics_gausiano['silhouette'].append(silhouette_score(embeddings, labels))
    metrics_gausiano['davies_bouldin'].append(davies_bouldin_score(embeddings, labels))
    metrics_gausiano['calinski_harabasz'].append(calinski_harabasz_score(embeddings, labels))
    metrics_gausiano['bic_scores'].append(gmm.bic(embeddings))
    metrics_gausiano['aic_scores'].append(gmm.aic(embeddings))

for metric in metrics_gausiano.keys():
    if len(metrics_gausiano[metric])==len(k_values) :
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, metrics_gausiano[metric], marker='o')
        plt.title(f'Metodo: {metric}, Gaussian Mixture Models  ', fontsize=16)
        plt.xlabel('Número de Clusters (k)', fontsize=14)
        plt.ylabel(f'Score {metric}', fontsize=14)
        plt.xticks(k_values)
        plt.grid(True)
        plt.show()

#wordcloud de clusters
from wordcloud import WordCloud
import matplotlib.pyplot as plt
embeddings = model.encode(df['Pedido_limpo'].tolist())
df['embeddings'] = embeddings.tolist()
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)


for i in df['cluster'].unique():
    text = df[df['cluster']==i]['Pedido_limpo'].str.cat(sep=',')
    wc = WordCloud(width=800, height=600, max_words=100,background_color="white",
                margin=10,max_font_size=50, 
        random_state=1).generate(text)
    # store default colored image
    text_dictionary = wc.process_text(text)
    wc.to_file(f'cluster_{i}.png')
    list_words= sorted(text_dictionary.items(), key= lambda x: x[1], reverse=True)
    
df['cluster'].value_counts().plot(kind='bar')
df['cluster'].value_counts()


#Análise de similaridade


from transformers import pipeline

frase1 = "pessimismo"
frase2 = "otimismo"

from sklearn.metrics.pairwise import cosine_similarity

def dectectar_sentimento(x,model, frase1, frase2, enable_score=False):        
    embeddings = model.encode([x])
    embeddings_frase1 = model.encode([frase1])
    embeddings_frase2 = model.encode([frase2])
    similaridade_frase1 = cosine_similarity(embeddings, embeddings_frase1)[0, 0]
    similaridade_frase2 = cosine_similarity(embeddings, embeddings_frase2)[0, 0]
    if similaridade_frase1 > similaridade_frase2:        
        return 'Otimista'if not enable_score else f"{similaridade_frase1:.4f}"
    else:
        return 'Pessimista' if not enable_score else f"{similaridade_frase1:.4f}"   

df['sentimento'] = df['Pedido_limpo'].apply(lambda x: dectectar_sentimento(x,model,frase1,frase2))
df['score_sentimento'] = df['discurso_limpo'].apply(lambda x: dectectar_sentimento(x,model,frase1,frase2,True))



# Análise de sentimento


print(df[['Pedido', 'sentimento','cluster']])
