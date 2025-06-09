import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load FAQ
faq_df = pd.read_csv("faq.csv")
faq_texts = [
    f"Q: {q.strip()} A: {a.strip()}" for q, a in zip(faq_df['question'], faq_df['answer'])
]

# Embed FAQ texts
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = embed_model.encode(faq_texts, convert_to_tensor=True)

# Build FAISS index
index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(faq_embeddings.cpu().detach().numpy())

# Load a small LLM
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# RAG-style helper functions
def get_relevant_context(query, k=2):
    query_emb = embed_model.encode([query])
    _, indices = index.search(query_emb, k)
    return "\n".join([faq_texts[i] for i in indices[0]])

def generate_response(context, query):
    prompt = f"Answer the question based on the following info:\n{context}\n\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = gen_model.generate(input_ids, max_length=200)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()

# Chatbot function for Gradio
def chatbot(user_input, history=[]):
    context = get_relevant_context(user_input)
    answer = generate_response(context, user_input)
    history.append((user_input, answer))
    return history, history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 RAG FAQ Chatbot")
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Your question")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chatbot, [msg, state], [chatbot_ui, state])
    clear.click(lambda: ([], []), None, [chatbot_ui, state])

# Run the app
if __name__ == "__main__":
    demo.launch()
