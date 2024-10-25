import streamlit as st
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import arxiv

# Cache the model and tokenizer to avoid reloading on every run
@st.cache_resource
def load_scibert_model():
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

# Cache the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load the model, tokenizer, and summarizer
scibert_model, tokenizer = load_scibert_model()
summarizer = load_summarizer()

# Function to get SciBERT embeddings for a given text
def get_scibert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = scibert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# Fetch ArXiv papers using their API
def fetch_arxiv_papers(query, max_results=5, categories=None):
    category_query = f" AND cat:{categories}" if categories else ""
    full_query = query + category_query
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(search.results())

# Function to extract excerpts related to the concept from papers
def extract_paper_excerpts(papers):
    excerpts = []
    for paper in papers:
        abstract = paper.summary
        # Use summarization model to create concise excerpts
        summary = summarizer(abstract, max_length=50, min_length=30, do_sample=False)
        excerpts.append((paper.title, summary[0]['summary_text'], paper.pdf_url))
    return excerpts

# List closely related papers by calculating similarity between SciBERT embeddings
def list_related_papers_with_excerpts(concept, max_results=5, similarity_threshold=0.4, categories="cs.LG"):
    papers = fetch_arxiv_papers(concept, max_results=max_results, categories=categories)
    
    if len(papers) == 0:
        return f"No papers found for concept '{concept}'"
    
    # Get SciBERT embedding for the concept
    concept_embedding = get_scibert_embedding(concept)
    
    related_papers = []
    for paper in papers:
        paper_text = paper.title + " " + paper.summary
        paper_embedding = get_scibert_embedding(paper_text)
        similarity_score = cosine_similarity([concept_embedding], [paper_embedding]).flatten()[0]
        if similarity_score >= similarity_threshold:
            related_papers.append((paper, similarity_score))
    
    # Sort papers by similarity score
    related_papers.sort(key=lambda x: x[1], reverse=True)
    
    # Extract excerpts from top papers
    excerpts = extract_paper_excerpts([paper[0] for paper in related_papers])
    
    return excerpts

# Streamlit app for listing related papers and excerpts
def main():
    st.title("📄 Find Related Research Papers")
    concept = st.text_input("Enter a research concept or topic")
    max_results = 7
    similarity_threshold = 0.4
    categories = "cs.LG"

    if st.button("Find Related Papers"):
        with st.spinner("Searching for related papers..."):
            related_papers = list_related_papers_with_excerpts(concept, max_results, similarity_threshold, categories)
            
            if isinstance(related_papers, str):
                st.error(related_papers)
            else:
                for i, (title, excerpt, url) in enumerate(related_papers):
                    st.subheader(f"{i+1}. {title}")
                    st.write(f"**Excerpt**: {excerpt}")
                    st.markdown(f"[Read full paper]({url})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
