import streamlit as st
from dotenv import load_dotenv
import os
import requests
from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
news_api_key = os.getenv("NEWS_API_KEY")

# Set Groq API key in the environment
groq_instance = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5, api_key=GROQ_API_KEY)

# Define State Structure
class State(Dict):
    news_content: str
    summary: str
    correctness: str
    category: str
    sentiment: str
    bias: str
    reliability_score: str

# Node Functions
def summarize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following news content in a few sentences: {news_content}"
    )
    chain = prompt | ChatGroq(temperature=0.5)
    summary = chain.invoke({"news_content": state["news_content"]}).content
    return {"summary": summary}

def check_correctness(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Check the correctness of the following news content: {news_content}"
    )
    chain = prompt | ChatGroq(temperature=0.5)
    correctness = chain.invoke({"news_content": state["news_content"]}).content
    return {"correctness": correctness}

def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following news article into Politics, Sports, Technology, or General. Article: {news_content}"
    )
    chain = prompt | ChatGroq(temperature=0)
    category = chain.invoke({"news_content": state["news_content"]}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following news content: {news_content}. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'."
    )
    chain = prompt | ChatGroq(temperature=0)
    sentiment = chain.invoke({"news_content": state["news_content"]}).content
    return {"sentiment": sentiment}

def detect_bias(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Detect any potential bias in the following news article: {news_content}."
    )
    chain = prompt | ChatGroq(temperature=0)
    bias = chain.invoke({"news_content": state["news_content"]}).content
    return {"bias": bias}

def compute_reliability(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on cross-referencing sources, provide a reliability score (0-100) for the following article: {news_content}"
    )
    chain = prompt | ChatGroq(temperature=0)
    score = chain.invoke({"news_content": state["news_content"]}).content
    return {"reliability_score": str(score)}

# Create and Configure the Graph
news_workflow = StateGraph(State)
news_workflow.add_node("summarize", summarize)
news_workflow.add_node("check_correctness", check_correctness)
news_workflow.add_node("categorize", categorize)
news_workflow.add_node("analyze_sentiment", analyze_sentiment)
news_workflow.add_node("detect_bias", detect_bias)
news_workflow.add_node("compute_reliability", compute_reliability)

# Define the workflow connections
news_workflow.add_edge("summarize", "check_correctness")
news_workflow.add_edge("check_correctness", "categorize")
news_workflow.add_edge("categorize", "analyze_sentiment")
news_workflow.add_edge("analyze_sentiment", "detect_bias")
news_workflow.add_edge("detect_bias", "compute_reliability")
news_workflow.add_edge("compute_reliability", END)

# Set entry point
news_workflow.set_entry_point("summarize")

# Compile the workflow
news_app = news_workflow.compile()

# Function to run the agent
def run_news_agent(news_content: str) -> Dict[str, str]:
    """Process news content through the LangGraph workflow and return the results."""
    results = news_app.invoke({"news_content": news_content})
    return {
        "summary": results["summary"],
        "correctness": results["correctness"],
        "category": results["category"],
        "sentiment": results["sentiment"],
        "bias": results.get("bias", "N/A"),
        "reliability_score": results.get("reliability_score", "N/A")
    }

# Function to fetch news articles using News API
def fetch_news_articles(query: str):
    news_api_url = "https://newsapi.org/v2/everything"
    headers = {
        "Authorization": f"Bearer {news_api_key}"
    }
    params = {
        "q": query,
        "language": "en"
    }
    
    response = requests.get(news_api_url, headers=headers, params=params)
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        raise Exception(f"Failed to fetch news articles: {response.status_code}")

# Streamlit App
st.title("News Summarization and Correctness Agent")

# Get user input for the news query
query = st.text_input("Enter a news topic or keyword to analyze:")

if st.button("Analyze News"):
    if query:
        # Fetch and display news articles
        try:
            articles = fetch_news_articles(query)
            st.write(f"Found {len(articles)} articles on '{query}':")
            
            for article in articles:
                st.subheader(article["title"])
                st.write(f"URL: {article['url']}")
                
                # Run the news agent to process the article content
                result = run_news_agent(article["content"])

                # Function to colorize each result section
                def colorize_result(result_key, result_value, color):
                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 5px;">
                            <strong>{result_key}:</strong> {result_value}
                        </div>
                    """, unsafe_allow_html=True)

                # Display the results
                colorize_result("Summary", result['summary'], '#f0f9ff')
                colorize_result("Correctness", result['correctness'], '#e0ffe0')
                colorize_result("Category", result['category'], '#fff8e1')
                colorize_result("Sentiment", result['sentiment'], '#fbe7e9')
                colorize_result("Bias", result['bias'], '#fef9e7') 
                colorize_result("Reliability Score", result['reliability_score'], '#e7f9e7')
                st.write("\n---\n")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query to analyze.")
