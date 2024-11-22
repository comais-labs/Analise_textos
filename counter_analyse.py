import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora

import spacy

# df = pd.DataFrame(data['opiniao'], columns=['pesquisador','discurso'])
# df.to_json('data_discurso.json', orient='records', indent=4)
# df = pd.read_json('data_discurso.json', encoding='utf-8')

df = pd.read_excel('data/Dataset_Atendimentos_Formatado.xlsx')
df['Pedido'] 

#spacy download pt_core_news_lg

nlp = spacy.load('pt_core_news_lg')

stop_words = nlp.Defaults.stop_words

#Carregar os dados
#df = pd.read_json('data_discurso.json')

df.dropna(subset=['Pedido'], inplace=True )
df['Pedido']


# Função para limpar os discursos
def limpar_texto(texto):
    texto = texto.lower().strip()  # Converter para minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # Remover pontuações
    texto = re.sub(r'\d+', '', texto)  # Remover números
    texto = texto.replace('\n', ' ')
    texto = re.sub(r'\s+', ' ', texto)  # Remover espaços extras
    
    #remover quebra de linha
    return texto

# Aplicar a limpeza aos discursos
df['Pedido_limpo'] = df['Pedido'].apply(limpar_texto)

# Ver os discursos limpos
print(df[['Pedido', 'Pedido_limpo']].head())
#tokenizar 

def tokenizar(texto):
    tokens = texto.split(' ') # Dividir o texto em palavras
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


# Tokenizar os discursos limpos
df['palavras'] = df['Pedido_limpo'].apply(tokenizar) 

# Criar uma lista de todas as palavras
todas_palavras = [palavra for discurso in df['palavras'] for palavra in discurso ]

# Contar a frequência das palavras
contagem_palavras = Counter(todas_palavras)

# Exibir as palavras mais comuns
print(contagem_palavras.most_common(10))

# gera uma lista de lemmas
def lemmatizar(palavras):
    doc = nlp(' '.join(palavras))
    lemmas = [token.lemma_ for token in doc if token.lemma_ not in stop_words]
    return lemmas
df['lemmas'] = df['palavras'].apply(lemmatizar)
# concatenar todas as listas de lemmas
todos_lemmas = [lemma for discurso in df['lemmas'] for lemma in discurso]

# Contar a frequência dos lemmas
contagem_lemmas = Counter(todos_lemmas)
print(contagem_lemmas.most_common(10))

#Extração de Palavras-Chave (TF-IDF)

tfidf_vectorizer_lemmas = TfidfVectorizer()
tfidf_vectorizer_pedidos = TfidfVectorizer()


tfidf_matrix = tfidf_vectorizer_lemmas.fit_transform(df['lemmas'].apply(lambda x: ' '.join(x)))
# lista de palavras-chave
feature_names = tfidf_vectorizer_lemmas.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()

tfidf_matrix_pedidos = tfidf_vectorizer_pedidos.fit_transform(df['Pedido_limpo'])
feature_names_pedidos = tfidf_vectorizer_pedidos.get_feature_names_out()
dense_pedidos = tfidf_matrix_pedidos.todense()
denselist_pedidos = dense_pedidos.tolist()

tfidf_data = pd.DataFrame(denselist_pedidos, columns=feature_names_pedidos)
for i in range(5):
    print('--'*40)
    print(df['Pedido'].iloc[i])
    print(tfidf_data.iloc[i].sort_values(ascending=False).head(10))
    print('**'*40)


df.to_parquet('data/df.parquet')
#Análise de Tópicos com LDA( Latent Dirichlet Allocation )

df = pd.read_parquet('data/df.parquet')
from gensim.models.ldamodel import LdaModel


corpus = df['Pedido_limpo'].apply(lambda x: [token.text for token in nlp(x) if token.text not in stop_words])
id2word = corpora.Dictionary(corpus)

corpus_doc2bow = [id2word.doc2bow(doc) for doc in corpus]



lda_model = LdaModel(corpus_doc2bow, num_topics=4, id2word=id2word, passes=15)

# Exibir os tópicos encontrados
for idx, tópico in lda_model.print_topics(-1):
    print(f"Tópico {idx + 1}: {tópico}")

id_doc = 0
documento_topicos = lda_model.get_document_topics(corpus_doc2bow[id_doc])
palavra = [id2word[x[0]] for x in documento_topicos]
print(f"Tópicos no Documento {id_doc}: {documento_topicos}, \n{palavra}")

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
# Preparar os dados para visualização
pyLDAvis.gensim.prepare(lda_model, corpus_doc2bow, id2word)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Concatenar todos os discursos limpos
todos_pedidos = ' '.join(df['lemmas'].apply(lambda x: ' '.join(x)))

# Gerar a nuvem de palavras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(todos_pedidos)

# Exibir a nuvem de palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('wordcloud.png')

