import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import sys

print(f"Carregando dados...")

# ---------------------------
# carregar dados
# ---------------------------

user_df = pd.read_csv("kaggle/input/archive/series/series-enriched.csv")
series_db = pd.read_csv("/kaggle/input/archive/series/shows.csv")


print(f"User data loaded: {len(user_df)} rows")
print(f"Series data loaded: {len(series_db)} rows")
# ---------------------------
# pré-processamento e limpeza
# ---------------------------

if "popularity" in series_db.columns:
    series_db = series_db[series_db["popularity"] > 5.0]

series_db = series_db.drop_duplicates(subset=["name"], keep="first")
series_db["overview"] = series_db["overview"].fillna("")
series_db = series_db.reset_index(drop=True)


# ---------------------------
# criar textos semânticos
# ---------------------------

user_df["text"] = (
    user_df["series_name"].fillna("") + " " +
    user_df["genre"].fillna("") + " " +
    user_df["description"].fillna("")
)

series_db["text"] = (
    series_db["name"].fillna("") + " " +
    series_db["overview"].fillna("")
)


print("Criando Emebedding")
# ---------------------------
# carregar modelo embedding
# ---------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Modelos de embedding carregados.")

# ---------------------------
# gerar embeddings
# ---------------------------

watched_series = set(user_df["series_name"].str.lower())
db_embeddings = embed_model.encode(series_db["text"].tolist())

# ---------------------------
# criar vector database FAISS
# ---------------------------

dimension = db_embeddings.shape[1]


index = faiss.IndexFlatL2(dimension)
index.add(np.ascontiguousarray(db_embeddings))


# ---------------------------
# função de busca RAG
# ---------------------------

def retrieve_context(query, k=2, user_k=2):
    """
    Busca recomendações usando:
    1. Cross-Encoder para encontrar séries relevantes no histórico do usuário.
    2. Combina descrições do histórico com a query.
    3. Busca no banco de dados (FAISS) usando a query enriquecida.
    4. Refina os candidatos finais com Cross-Encoder.
    """

    # ----------------------------------
    # enriquecer query com histórico relevante do banco 
    # ----------------------------------

    augmented_query = query

    if not user_df.empty:
        history_pairs = []
        for text in user_df["text"]:
            history_pairs.append([query, str(text)])
        
        if history_pairs:
            history_scores = cross_encoder.predict(history_pairs)

            top_history_indices = np.argsort(history_scores)[-user_k:][::-1]
            relevant_history_desc = user_df.iloc[top_history_indices]["description"].fillna("").tolist()
            history_context_str = " ".join([str(d) for d in relevant_history_desc])
            
            augmented_query = f"{query}: {history_context_str}"
            print(f"\n[DEBUG] Augmented Query:\n{augmented_query}\n")


    # ----------------------------------
    #  busca no faiss por distancia euclidiana
    # ----------------------------------

    query_vector = embed_model.encode([augmented_query])

    search_k = k*10

    distances, indices = index.search(query_vector.astype("float32"), search_k)
    print("DEBUG: FAISS search completed. Distances shape:", distances, "Indices shape:", indices)
    if len(indices[0]) == 0:
        return "Nenhuma série encontrada."

    valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(series_db)]
    
    print(f"\n[DEBUG] Valid indices from FAISS: {valid_indices}\n")
    if not valid_indices:
        return "Nenhuma série válida encontrada."

    cross_input = []
    candidates_data = []
    seen_titles = set()
    watched_titles = {str(x).lower().strip() for x in user_df["series_name"].unique()}
    
    for idx in valid_indices:
        row = series_db.iloc[idx]
        title = str(row.get("name", "Unknown")).strip()
        desc = str(row.get("overview", "")).strip()
        genre = str(row.get("genre", "N/A")).strip()

        if len(desc) == 0:
            continue
            
        title_lower = title.lower()
        if title_lower in seen_titles or title_lower in watched_titles:
            continue
            
        seen_titles.add(title_lower)
        
        candidate_text = f"{title}. {desc}"
        cross_input.append([query, candidate_text])
        candidates_data.append({
            "title": title,
            "genre": genre,
            "desc": desc
        })

    if not cross_input:
        return "Nenhuma série válida encontrada após filtragem."

    # ----------------------------------
    #  reranking com Cross-Encoder
    # ----------------------------------
    scores = cross_encoder.predict(cross_input)


    scored_candidates = list(zip(scores, candidates_data))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:k]

    filtered_context = ""
    for score, data in top_candidates:
        filtered_context += f"""
            Title: {data['title']}
            Genre: {data['genre']}
            Description: {data['desc']}
            Relevance Score: {score:.4f}
            ---
            """

    return filtered_context

llm = ChatOllama(
    model="llama3",
    temperature=0.5
)


def generate_response(user_input):
    """
    Gera uma recomendação de série baseada na entrada do usuário.
    """
    context = retrieve_context(user_input)
    
    system_prompt = f"""
    [SYSTEM]
    You are a helpful series recommendation assistant.
    User preferences are based on their watch history.
    Don't show Relevance Score to the user.
    [CONTEXT]:
    {context}
    [INSTRUCTION]:
    Use only the context to recommend series that match the user's request: "{user_input}"
    Explain why you are recommending each series based on the context provided. If no good matches are found, say "No good recommendations found based on the user's preferences. Use only the context"
    """
    print("\n[DEBUG] System Prompt:\n" + "-"*40 + "\n" + system_prompt + "\n" + "-"*40)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = llm.invoke(messages)
    return response.content

def run_chat():
    print("Sistema de Recomendação de Séries (digite 'sair' para encerrar)")
    chat_history = []
    
    while True:
        try:
            user_input = input("\nVocê: ")
            if not user_input or user_input.strip() == "":
                continue
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break

            response_content = generate_response(user_input)
            print(f"\nBot: {response_content}")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response_content))
            
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\nErro: {e}")

if __name__ == "__main__":
    
    print("Iniciando sistema de recomendação...")
    run_chat()