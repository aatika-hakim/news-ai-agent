# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# import requests
# from typing import Dict, List, Tuple

# # Load environment variables
# load_dotenv()

# # Get API keys from environment
# GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# # Initialize Groq model
# groq_instance = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5, api_key=GROQ_API_KEY)

# # Function to retrieve news articles
# def fetch_news(topic: str, max_results: int = 5) -> List[Dict]:
#     url = (f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&pageSize={max_results}")
#     response = requests.get(url)
#     news_data = response.json()
#     if news_data["status"] == "ok":
#         return news_data["articles"]
#     else:
#         raise Exception(f"Error fetching news: {news_data['message']}")

# # Summarize News using Groq Model
# def summarize_news(articles: List[Dict]) -> List[str]:
#     summaries = []
#     for article in articles:
#         prompt = f"Summarize this news article: {article['title']} {article['description']}"
#         try:
#             summary = groq_instance(prompt)
#             summaries.append(summary)
#         except Exception as e:
#             st.error(f"Error during summarization: {e}")
#             summaries.append("Error in generating summary.")
#     return summaries

# # Check for inaccuracies in the news and provide corrections
# def cross_check_inaccuracies(articles: List[Dict], summaries: List[str]) -> List[Dict]:
#     corrected_summaries = []
#     for i, article in enumerate(articles):
#         prompt = (
#             f"Here is a summary of the article: {summaries[i]}.\n"
#             f"Check for any inaccuracies based on other reliable sources and provide corrections."
#         )
#         try:
#             correction = groq_instance(prompt)
#             corrected_summaries.append({"original": summaries[i], "corrected": correction})
#         except Exception as e:
#             st.error(f"Error during correction: {e}")
#             corrected_summaries.append({"original": summaries[i], "corrected": "Error in correction."})
#     return corrected_summaries

# # Summarize News using Groq Model with debugging
# def summarize_news(articles: List[Dict]) -> List[str]:
#     summaries = []
#     for article in articles:
#         prompt = f"Summarize this news article: {article['title']} {article['description']}"
#         try:
#             # Debug: print the prompt being sent to the Groq model
#             st.write(f"Prompt for summarization: {prompt}")
            
#             summary = groq_instance(prompt)
            
#             # Debug: print the type of response from the Groq model
#             st.write(f"Response from Groq summarization: {type(summary)}, {summary}")
            
#             summaries.append(summary)
#         except Exception as e:
#             st.error(f"Error during summarization: {e}")
#             summaries.append("Error in generating summary.")
#     return summaries

# # Check for inaccuracies in the news and provide corrections with debugging
# def cross_check_inaccuracies(articles: List[Dict], summaries: List[str]) -> List[Dict]:
#     corrected_summaries = []
#     for i, article in enumerate(articles):
#         prompt = (
#             f"Here is a summary of the article: {summaries[i]}.\n"
#             f"Check for any inaccuracies based on other reliable sources and provide corrections."
#         )
#         try:
#             # Debug: print the prompt being sent for correction
#             st.write(f"Prompt for correction: {prompt}")
            
#             correction = groq_instance(prompt)
            
#             # Debug: print the type of response from the Groq model
#             st.write(f"Response from Groq correction: {type(correction)}, {correction}")
            
#             corrected_summaries.append({"original": summaries[i], "corrected": correction})
#         except Exception as e:
#             st.error(f"Error during correction: {e}")
#             corrected_summaries.append({"original": summaries[i], "corrected": "Error in correction."})
#     return corrected_summaries

# # Grade the reliability of news sources with debugging
# def grade_reliability(articles: List[Dict]) -> List[Tuple[str, str]]:
#     graded_sources = []
#     for article in articles:
#         source = article['source']['name']
#         prompt = f"Grade the reliability of the source '{source}' on a scale of 1 to 10 and explain your rating."
#         try:
#             # Debug: print the prompt for reliability grading
#             st.write(f"Prompt for reliability grading: {prompt}")
            
#             reliability_rating = groq_instance(prompt)
            
#             # Debug: print the type of response from the Groq model
#             st.write(f"Response from Groq reliability grading: {type(reliability_rating)}, {reliability_rating}")
            
#             graded_sources.append((source, reliability_rating))
#         except Exception as e:
#             st.error(f"Error during reliability grading: {e}")
#             graded_sources.append((source, "Error in grading."))
#     return graded_sources


# # Streamlit App
# def app():
#     st.title("Real-Time News Summarization and Correction Agent")
    
#     # Input for topic
#     topic = st.text_input("Enter the topic for news summarization", "artificial intelligence")
    
#     # Input for number of articles to retrieve
#     max_results = st.slider("Number of articles to retrieve", 1, 10, 5)
    
#     if st.button("Fetch News and Summarize"):
#         try:
#             # Fetch news articles
#             with st.spinner("Fetching news articles..."):
#                 articles = fetch_news(topic, max_results)
            
#             # Display raw articles
#             st.subheader(f"Top {max_results} Articles for '{topic}'")
#             for i, article in enumerate(articles):
#                 st.write(f"**{i + 1}. {article['title']}**")
#                 st.write(f"Source: {article['source']['name']}")
#                 st.write(article['description'])
#                 st.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)  # Make the link clickable
#                 st.write("—" * 10)
            
#             # Summarize news articles
#             with st.spinner("Summarizing articles..."):
#                 summaries = summarize_news(articles)
            
#             # Cross-check inaccuracies and generate corrected summaries
#             with st.spinner("Checking for inaccuracies and providing corrections..."):
#                 corrected_summaries = cross_check_inaccuracies(articles, summaries)
            
#             # Grade source reliability
#             with st.spinner("Grading source reliability..."):
#                 reliability_grades = grade_reliability(articles)
            
#             # Display results
#             st.subheader("Summarized and Corrected Articles")
#             for i, article in enumerate(articles):
#                 st.write(f"**{i + 1}. {article['title']}**")
#                 st.write(f"Source: {article['source']['name']}")
#                 st.write(f"Original Summary: {corrected_summaries[i]['original']}")
#                 st.write(f"Corrected Summary: {corrected_summaries[i]['corrected']}")
#                 st.write(f"Source Reliability: {reliability_grades[i][1]}")
#                 st.write("—" * 10)
        
#         except Exception as e:
#             st.error(f"Error: {e}")

# # Run the app
# if __name__ == "__main__":
#     app()


import os
import requests
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import streamlit as st

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key="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")


# Define a list of reliable sources
reliable_sources = ['BBC', 'Reuters', 'The New York Times', 'The Guardian']

# Step 1: Retrieve News Articles
def get_news_articles(topic: str, num_articles: int):
    """Fetch latest news articles based on a topic."""
    api_key = os.getenv("NEWS_API_KEY")  # Get the API key from environment variable
    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&apiKey={api_key}&pageSize={num_articles}"
    
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to fetch articles. Please check the API key and topic.")
        return []
    
    articles = response.json().get('articles', [])
    
    # Debug: Log the retrieved articles
    for article in articles:
        print(f"Title: {article['title']}, Source: {article['source']['name']}, Published At: {article['publishedAt']}")
    
    return [
        {
            'title': article['title'],
            'description': article['description'],
            'content': article['content'],
            'url': article['url'],
            'source': article['source']['name'],
            'publishedAt': article['publishedAt']
        }
        for article in articles
    ]

# Step 2: Filter Recent News
def filter_recent_news(articles, days=1):
    """Filter news articles to retrieve only recent ones."""
    recent_articles = []
    cutoff_time = datetime.now() - timedelta(days=days)
    
    for article in articles:
        article_time = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        if article_time > cutoff_time:
            recent_articles.append(article)
    
    # Debug: Log the filtered recent articles
    for article in recent_articles:
        print(f"Filtered Title: {article['title']}, Source: {article['source']['name']}")
            
    return recent_articles

# Step 3: Summarize Articles
def summarize_article(article):
    """Summarize a given news article."""
    prompt = f"Summarize the following article:\n{article['content']}\n"
    summary = llm.predict(prompt)
    return summary

# Step 4: Cross-Reference Articles
def cross_reference_articles(articles):
    """Compare and cross-reference key points between articles."""
    summaries = [summarize_article(article) for article in articles]
    
    # Collect unique key points from all summaries
    combined_summary = " ".join(summaries)
    return combined_summary

# Step 5: Grade Source Reliability
def grade_source_reliability(source_name):
    """Grade the reliability of the news source."""
    if source_name in reliable_sources:
        return 9  # High reliability
    else:
        return 1  # Lower reliability for less-known sources

# Step 6: Correct and Finalize the Summary
def correct_summary(articles):
    """Correct summary based on cross-referenced articles and source reliability."""
    reliable_summaries = []
    
    for article in articles:
        reliability_grade = grade_source_reliability(article['source'])
        summary = summarize_article(article)
        # Append the summary along with its reliability grade
        reliable_summaries.append((summary, reliability_grade))
    
    # Sort summaries by reliability grade
    reliable_summaries.sort(key=lambda x: x[1], reverse=True)
    
    # If there are no reliable summaries, return the best available summary
    if reliable_summaries:
        return reliable_summaries[0][0]  # Return the highest reliability summary
    return "No reliable summaries found."

# Step 7: News Summarization Agent
def news_summary_agent(topic: str, num_articles: int):
    """Main function for retrieving, summarizing, and correcting news on a given topic."""
    # Step 1: Get news articles
    articles = get_news_articles(topic, num_articles)
    
    # Step 2: Filter for recent news
    recent_articles = filter_recent_news(articles)
    
    # Step 3: Cross-reference articles (optional but can enhance summary)
    cross_reference_articles(recent_articles)
    
    # Step 4: Correct and finalize summary
    final_summary = correct_summary(recent_articles)
    
    return final_summary

# Streamlit app interface
def main():
    st.title("Real-Time News Summarization and Correction Agent")
    topic = st.text_input("Enter a topic for news summarization:", "artificial intelligence")
    num_articles = st.slider("Number of articles to retrieve:", 1, 20, 5)

    if st.button("Get Summary"):
        with st.spinner("Fetching news..."):
            summary = news_summary_agent(topic, num_articles)
            if summary:
                st.success("Summary fetched successfully!")
                st.write(summary)
            else:
                st.error("No summaries available for the provided topic.")

if __name__ == "__main__":
    main()