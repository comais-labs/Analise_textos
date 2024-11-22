
# Analisador de Texto

## Descrição
Este projeto é um **Analisador de Texto** desenvolvido em Python, que realiza diversas análises e pré-processamentos em um conjunto de textos, como limpeza, tokenização, lematização, extração de palavras-chave, análise de tópicos, e geração de visualizações como nuvem de palavras.

---

## Funcionalidades
1. **Carregamento e Pré-processamento de Dados**
   - Leitura de arquivos Excel (`.xlsx`) e JSON.
   - Limpeza de texto para remover números, pontuações e espaços desnecessários.

2. **Tokenização e Lematização**
   - Divisão de textos em palavras (tokens).
   - Redução das palavras à sua forma base (lematização) utilizando a biblioteca SpaCy.

3. **Contagem de Palavras e Lemmas**
   - Identificação das palavras mais frequentes.
   - Análise de frequência de palavras e lemmas.

4. **Extração de Palavras-Chave**
   - Utilização do TF-IDF para identificar palavras-chave de cada texto.

5. **Análise de Tópicos**
   - Aplicação do modelo LDA (Latent Dirichlet Allocation) para identificar tópicos em documentos.

6. **Visualizações**
   - Geração de nuvens de palavras para representar graficamente a frequência dos termos.

---

## Requisitos

### Instalar Python no Windows
1. Baixe o instalador mais recente do Python em [python.org](https://www.python.org/downloads/).
2. Execute o instalador e:
   - Marque a opção **"Add Python to PATH"**.
   - Clique em **"Customize Installation"** e certifique-se de incluir o módulo `pip`.
3. Após a instalação, abra o terminal (Prompt de Comando) e verifique se o Python foi instalado:
   ```bash
   python --version
   pip --version
   ```

### Instalar o Projeto via Git
1. Certifique-se de que o Git está instalado. Caso não esteja, faça o download e instale pelo site oficial: [git-scm.com](https://git-scm.com/).
2. No terminal, execute o comando para clonar o repositório:
   ```bash
   git clone https://github.com/seu-usuario/analisador-texto.git
   ```
3. Acesse o diretório do projeto:
   ```bash
   cd analisador-texto
   ```

### Criar um Ambiente Virtual (venv)
1. No terminal, crie o ambiente virtual:
   ```bash
   python -m venv venv
   ```
2. Ative o ambiente virtual:
   ```bash
   venv\Scripts\activate
   ```
3. Instale as dependências do projeto no ambiente virtual:
   ```bash
   pip install -r requirements.txt
   ```

---

## Estrutura do Projeto
```
data/
│   Dataset_Atendimentos_Formatado.xlsx
│
output/
│   df.parquet
│   wordcloud.png
│
requirements.txt     # Dependências do projeto
analisador_texto.py  # Código principal
README.md            # Documentação
```

---

## Uso

### 1. Carregar os Dados
Os dados são carregados de um arquivo Excel na pasta `data`. Certifique-se de que o arquivo `Dataset_Atendimentos_Formatado.xlsx` está presente no diretório.

### 2. Pré-processamento
O texto é limpo, convertido para minúsculas e processado para remoção de caracteres indesejados.

### 3. Análises e Resultados
- Palavras e lemmas mais comuns são identificados.
- Palavras-chave são extraídas usando TF-IDF.
- Tópicos nos textos são identificados com o modelo LDA.

### 4. Visualização
- Nuvens de palavras são geradas e salvas no diretório `output`.

### Executar o Código
Execute o script principal:
```bash
python analisador_texto.py
```

### Saída
- Arquivo parquet contendo os dados processados: `output/df.parquet`
- Imagem da nuvem de palavras: `output/wordcloud.png`

---

## Contribuição
Se desejar contribuir:
1. Faça um fork do repositório.
2. Crie uma branch para a sua funcionalidade:
   ```bash
   git checkout -b minha-nova-funcionalidade
   ```
3. Faça commit das suas alterações:
   ```bash
   git commit -m 'Minha nova funcionalidade'
   ```
4. Envie para o branch:
   ```bash
   git push origin minha-nova-funcionalidade
   ```
5. Abra um Pull Request.

---

## Licença
Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.

---

**Contato:** Se tiver dúvidas ou sugestões, entre em contato!
