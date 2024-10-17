import streamlit as st
import os
import requests
from langgraph.state_graph import StateGraph, TaskNode, Edge
from langchain_core.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import SimpleDocumentLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq model
groq_instance = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5, api_key=GROQ_API_KEY)

# Bing News API Key
API_KEY = "your_bing_news_api_key"

# LangChain Components for Summarization and Retrieval
prompt_template = PromptTemplate(input_variables=["summary"], template="Summarize the following article: {summary}")

# Define Groq-based summarization chain
def groq_summarize(text):
    # Use ChatGroq to summarize the text
    return groq_instance(text)

# Document Retrieval using LangChain and FAISS (optional)
def setup_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_loader = SimpleDocumentLoader("your_document_folder_path")
    documents = document_loader.load_and_split(text_splitter=text_splitter)
    faiss_index = FAISS.from_documents(documents, embeddings)
    return faiss_index

retriever = setup_retriever()

# Fetch News
def fetch_news(query, count=5):
    url = f"https://api.bing.microsoft.com/v7.0/news/search?q={query}&count={count}&sortBy=Date"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    response = requests.get(url, headers=headers)
    articles = response.json().get('value', [])
    
    news_data = []
    for article in articles:
        news = {
            "title": article["name"],
            "url": article["url"],
            "snippet": article["description"],
            "published_date": article["datePublished"],
            "source": article["provider"][0]["name"],
        }
        news_data.append(news)
    
    return news_data

# Define the agents as tasks for the StateGraph
class NewsFetcherAgent(TaskNode):
    def run(self, query):
        return fetch_news(query)

class GroqSummarizerAgent(TaskNode):
    def run(self, news_data):
        summaries = [groq_summarize(article['snippet']) for article in news_data]
        return summaries

# Cross-reference news articles (optional)
def cross_reference(news_data):
    # This is where you would implement logic to cross-reference articles.
    # For now, this is a placeholder for cross-checking facts between sources.
    return "Cross-reference completed"

class CrossCheckerAgent(TaskNode):
    def run(self, news_data):
        return cross_reference(news_data)

# Streamlit App Interface
st.title("Real-Time News Summarization and Correction Agent with Groq and LangChain Core")

# User Input
query = st.text_input("Enter a news topic or query", "")

if st.button("Fetch and Summarize News"):
    if query:
        # Initialize StateGraph
        graph = StateGraph()

        # Create the agents (nodes)
        fetcher_node = NewsFetcherAgent(name="News Fetcher")
        groq_summarizer_node = GroqSummarizerAgent(name="Groq Summarizer")
        cross_checker_node = CrossCheckerAgent(name="Cross-Checker")

        # Add the agents to the graph
        graph.add_node(fetcher_node)
        graph.add_node(groq_summarizer_node)
        graph.add_node(cross_checker_node)

        # Connect agents with dynamic transitions
        graph.add_edge(Edge(from_node=fetcher_node, to_node=groq_summarizer_node))
        graph.add_edge(Edge(from_node=groq_summarizer_node, to_node=cross_checker_node))

        # Run the graph starting from the fetcher agent
        st.write("Fetching news articles...")
        news_data = graph.run(fetcher_node, query=query)
        
        # Display the fetched news articles
        st.write(f"Fetched {len(news_data)} articles:")
        for i, article in enumerate(news_data):
            st.write(f"**{i+1}. {article['title']}** - {article['source']}")
            st.write(f"Published: {article['published_date']}")
            st.write(f"URL: {article['url']}")
            st.write(f"Snippet: {article['snippet']}")
            st.write("----")

        # Summarize each article with Groq
        st.write("Summarizing news articles using Groq...")
        groq_summaries = graph.run(groq_summarizer_node, news_data=news_data)
        for i, summary in enumerate(groq_summaries):
            st.write(f"**Groq Summary for Article {i + 1}:**")
            st.write(summary)
            st.write("----")
        
        # Perform cross-referencing between articles (optional)
        st.write("Cross-referencing articles for factual consistency...")
        cross_refs = graph.run(cross_checker_node, news_data=news_data)
        st.write(cross_refs)
    else:
        st.write("Please enter a query.")
