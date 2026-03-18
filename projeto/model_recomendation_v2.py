import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import sys

# ---------------------------
# Configurações do Modelo Híbrido
# ---------------------------
K_FAISS_CANDIDATES = 50      
TOP_K_FINAL = 5               


print(f"Carregando dados...")

# ---------------------------
# Carregar dados
# ---------------------------
try:
    user_df = pd.read_csv("kaggle/input/archive/series/series-enriched.csv")
    series_db = pd.read_csv("kaggle/input/archive/series/shows.csv")
except FileNotFoundError:
    print("Arquivos CSV não encontrados. Verifique os caminhos.")
    sys.exit(1)

print(f"User data loaded: {len(user_df)} rows")
print(f"Series data loaded: {len(series_db)} rows")

# ---------------------------
# Pré-processamento e limpeza
# ---------------------------

if "popularity" in series_db.columns:
    series_db = series_db[series_db["popularity"] > 5.0]

series_db = series_db.drop_duplicates(subset=["name"], keep="first")
series_db["overview"] = series_db["overview"].fillna("")
series_db = series_db.reset_index(drop=True)

# ---------------------------
# Criar textos semânticos
# ---------------------------
# Texto para o histórico do usuário
user_df["text"] = (
    user_df["series_name"].fillna("") + " " +
    user_df["genre"].fillna("") + " " +
    user_df["description"].fillna("")
)

# Texto para o banco de dados (catálogo)
series_db["text"] = (
    series_db["name"].fillna("") + " " +
    series_db["overview"].fillna("")
)

print("Carregando e gerando Embeddings...")

# ---------------------------
# Carregar modelo embedding
# ---------------------------
# Usamos o mesmo modelo para tudo (Query, Docs, Histórico)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Gerar embeddings
# ---------------------------

# 1. Embeddings do catálogo completo (para o índice FAISS)
db_embeddings = embed_model.encode(series_db["text"].tolist())

# 2. Embeddings do histórico do usuário (para cálculo de âncora de gosto)
user_texts = user_df["text"].fillna("").tolist()
if user_texts:
    history_embeddings = embed_model.encode(user_texts)
else:
    history_embeddings = np.array([])  # Vazio se sem histórico

# ---------------------------
# Criar vector database FAISS
# ---------------------------
dimension = db_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.ascontiguousarray(db_embeddings))
print(f"Índice FAISS criado com {index.ntotal} itens.")


def retrieve_context(query, k_faiss=50, top_k_final=5):
    taste_temperature = 100
    stretch_scores = True
    use_hybrid_score = False

    print(f"\nEvaluating Query: '{query}'\n" + "="*50)
    
    # 1. EMBEDDING DA QUERY
    query_emb = embed_model.encode([query]).astype("float32")

    dist, idx = index.search(query_emb, k_faiss)
    
    watched_titles = set()
    if not user_df.empty and "series_name" in user_df.columns:
        watched_titles = {str(x).lower().strip() for x in user_df["series_name"].unique()}
    
    valid_idx = []
    seen_titles = set()
    for i in idx[0]:
        if 0 <= i < len(series_db):
            candidate_title = str(series_db.iloc[i].get("name", "")).strip().lower()
            if candidate_title in watched_titles:
                continue
            if candidate_title not in seen_titles:
                valid_idx.append(i)
                seen_titles.add(candidate_title)

    if not valid_idx:
        print("Not enough unseen candidates found.")
        return "Not enough unseen candidates found."

    candidates_names = [str(series_db.iloc[i].get("name", "")) for i in valid_idx]
    

    candidates_embs = db_embeddings[valid_idx]
    
    pure_query_sims = cosine_similarity(candidates_embs, query_emb).flatten()
    top_pure_idx = np.argsort(pure_query_sims)[-top_k_final:][::-1]


    # 3. BUSCA NAS ASSISTIDAS (FILTRO DE GOSTO ATUAL)
    user_texts = user_df["text"].fillna("").tolist()
    if not user_texts:
        print("Usuário sem histórico.")
        return "Usuário sem histórico."
        
    history_embs = embed_model.encode(user_texts)
    
    history_to_query_sims = cosine_similarity(history_embs, query_emb).flatten()
    
    top_history_for_query_idx = np.argsort(history_to_query_sims)[-3:][::-1]
    
    print("[ITENS DO HISTÓRICO QUE ANCORAM O GOSTO DA QUERY ATUAL]")
    anchor_embs = []
    for i in top_history_for_query_idx:
         name = user_df.iloc[i].get("series_name", "Unknown")
         print(f" -> {name} (Hit com a Query: {history_to_query_sims[i]:.2f})")
         anchor_embs.append(history_embs[i])
 

    ideal_gosto_emb = np.mean(anchor_embs, axis=0).reshape(1, -1)
    


    gosto_sims = cosine_similarity(candidates_embs, ideal_gosto_emb).flatten()
    
    if taste_temperature is not None:
        weight_taste = max(0, min(100, taste_temperature)) / 100.0
        weight_query = 1.0 - weight_taste
        base_sims = (gosto_sims * weight_taste) + (pure_query_sims * weight_query)
        print(f"\n[Temperatura de Gosto Aplicada: {taste_temperature}% Gosto | { weight_query * 100:.0f}% Query Original]")
    elif use_hybrid_score:
        base_sims = (gosto_sims + pure_query_sims) / 2
    else:
        base_sims = gosto_sims
        
    if stretch_scores:
        final_sims = np.power(base_sims, 4) * 1000
    else:
        final_sims = base_sims
    
    top_final_idx = np.argsort(final_sims)[-top_k_final:][::-1]
    
    print("\n[RANKING ORIGINAL (Sem filtro de Gosto - Pura similaridade com a Query)]")
    for rank, i in enumerate(top_pure_idx):
        print(f" {rank+1}. {candidates_names[i]} | Match com Query: {pure_query_sims[i]:.2f}")
        
    print("\n[Similaridade com o Embedding do Histórico Assistido)]")
    context_parts = []
    
    for rank, i in enumerate(top_final_idx):
        print(f" {rank+1}. {candidates_names[i]} | Match com o Seu Gosto: {final_sims[i]:.2f}")
        

        db_idx = valid_idx[i]
        
        item_title = series_db.iloc[db_idx].get("name", "Unknown")
        item_desc = series_db.iloc[db_idx].get("unencoded_overview", "") # Fallback ou usar overview
        item_desc = series_db.iloc[db_idx].get("overview", "")
        item_genre = series_db.iloc[db_idx].get("genre", "Unknown")
        
        context_parts.append(f"""
            Title: {item_title}
            Genre: {item_genre}
            Description: {item_desc}
            Relevance Score (Hybrid): {final_sims[i]:.4f}
            ---
        """)

    return "\n".join(context_parts)

# ---------------------------
# LLM Integration
# ---------------------------
llm = ChatOllama(
    model="llama3",
    temperature=0.5
)

def generate_response(user_input):
    context = retrieve_context(user_input)
    
    if "Nenhuma série válida" in context:
        return "Desculpe, não consegui encontrar recomendações com base na sua busca."

    system_prompt = f"""
    [SYSTEM]
    You are a helpful series recommendation assistant.
    [CONTEXT]:
    {context}

    [INSTRUCTION]:
    You MUST recommend the TOP 3 series from the context list based on the user's request: "{user_input}" and their Scores.
    For each of the 3 series, provide a short, distinct reason explaining why they would enjoy it, based on the description.
    Present them clearly, perhaps using bullet points or a numbered list.Don't show the Relevance Score:
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    print("\n[DEBUG] Gerando resposta com LLM...")
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Erro ao comunicar com o LLM: {e}"

def run_chat():
    print("="*60)
    print("Sistema de Recomendação v2 (Busca Híbrida: Query + Gosto)")
    print("Tecnologia de 'Evaluate FAISS Augment' aplicada.")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nO que você quer assistir hoje? (ou 'sair'): ")
            if not user_input or user_input.strip() == "":
                continue
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break

            response_content = generate_response(user_input)
            print(f"\nBot: {response_content}")
            
        except KeyboardInterrupt:
            print("\nOperação cancelada.")
            break
        except Exception as e:
            print(f"\nErro inesperado: {e}")

if __name__ == "__main__":
    run_chat()