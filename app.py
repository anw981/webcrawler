import streamlit as st
import requests
import time
import re
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2.service_account import Credentials
import gspread
from playwright.sync_api import sync_playwright
import os

# ==================== CONFIGURATION ====================
RELEVANCE_THRESHOLD = 0.01  # Lowered for debugging
MAX_CRAWL_DEPTH = 2
CRAWL_TIMEOUT = 30 * 60  # 30 minutes
CUSTOM_DOMAINS = ["https://example.com"]  # Use a simple, known site for testing
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ==================== GOOGLE SHEETS ====================
def get_gsheet_client(credentials_json):
    credentials = Credentials.from_service_account_info(credentials_json, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client

def load_credentials():
    uploaded_file = st.file_uploader("Upload Google Sheets credentials JSON", type="json")
    if uploaded_file:
        return json.load(uploaded_file)
    return None

def update_sheet(sheet, links):
    existing = sheet.col_values(1)
    new_links = [link for link in links if link not in existing]
    if new_links:
        sheet.append_rows([[link] for link in new_links])
    return new_links

# ==================== TF-IDF RELEVANCE ====================
def is_relevant(content, keywords):
    docs = [content, " ".join(keywords)]
    try:
        vectorizer = TfidfVectorizer().fit_transform(docs)
        score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    except Exception as e:
        st.write(f"Error in TF-IDF for content: {str(e)}")
        return False, 0
    return score >= RELEVANCE_THRESHOLD, score

# ==================== LINK CLASSIFICATION ====================
def classify_link(html):
    if re.search(r'<input|<form|@|\.com', html, re.IGNORECASE):
        return 'form'
    return 'open'

# ==================== CRAWLING (Playwright only) ====================
def install_playwright_browsers():
    """Ensure Playwright browsers are installed"""
    os.system("playwright install")

def get_links_from_page(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            time.sleep(2)  # Ensure page is fully loaded
            html = page.content()
            st.write(f"Fetched {len(html)} characters from {url}")
            soup = BeautifulSoup(html, 'html.parser')
            links = set()
            for tag in soup.find_all('a', href=True):
                href = tag['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).scheme in ['http', 'https']:
                    links.add(full_url)
            browser.close()
            return html, list(links)
    except Exception as e:
        st.write(f"Error fetching page {url}: {str(e)}")
        return '', []

def crawl_site(start_urls, keywords, visited, depth=0):
    results = {'open': [], 'form': []}
    if depth > MAX_CRAWL_DEPTH:
        return results

    for url in start_urls:
        if url in visited:
            continue
        visited.add(url)
        html, links = get_links_from_page(url)
        if not html:
            st.write(f"Skipped (no HTML): {url}")
            continue
        is_rel, score = is_relevant(html, keywords)
        st.write(f"Visited: {url} | HTML length: {len(html)} | Relevance score: {score:.3f}")
        if is_rel:
            category = classify_link(html)
            results[category].append(url)
        sub_results = crawl_site(links, keywords, visited, depth + 1)
        results['open'].extend(sub_results['open'])
        results['form'].extend(sub_results['form'])

    return results

# ==================== GOOGLE SEARCH API ====================
def google_search(query, api_key, cse_id):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
    st.write(f"Querying Google Search API with: {query}")  # Helpful for debugging
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
            return []
        results = response.json().get("items", [])
        return [item["link"] for item in results if "link" in item]
    except Exception as e:
        st.error(f"Exception while fetching Google results: {e}")
        return []


# ==================== STREAMLIT APP ====================
st.set_page_config(page_title="Smart Web Crawler", layout="wide")
st.title("Smart Web Crawler with Google Sheets Integration")

# Ensure Playwright browsers are installed before running any logic
install_playwright_browsers()

keywords_input = st.text_input("Enter keywords (comma separated):")
option = st.selectbox("Choose search option", ["Google Search", "Internal Site Search"])
api_key = st.text_input("Google API Key (for Google Search option only):", type="password")
cse_id = st.text_input("Custom Search Engine ID (for Google Search option only):", type="password")
sheet_url_open = st.text_input("Enter Google Sheet URL for Open Access Links")
sheet_url_form = st.text_input("Enter Google Sheet URL for Form-Based Links")
credentials = load_credentials()

def generate_queries(keywords):
    keywords = [k.strip() for k in keywords]
    and_query = " AND ".join([f'"{k}"' for k in keywords])
    or_query = " OR ".join([f'"{k}"' for k in keywords])
    return [and_query, or_query]


if st.button("Start Crawling") and keywords_input and credentials and sheet_url_open and sheet_url_form:
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    
    # Check for missing Google API credentials
    if not api_key or not cse_id:
        st.error("Please provide valid Google API Key and Custom Search Engine ID")
        st.stop()

    client = get_gsheet_client(credentials)
    sheet_open = client.open_by_url(sheet_url_open).sheet1
    sheet_form = client.open_by_url(sheet_url_form).sheet1

    # Collecting initial URLs
    initial_urls = []

    # 1. Get URLs from Google Search (Option 1)
    if option == "Google Search":
        queries = generate_queries(keywords)
        for query in queries:
            google_urls = google_search(query, api_key, cse_id)
            initial_urls.extend(google_urls)
    
    # 2. Get URLs from Internal Domain List (Option 2)
    if option == "Internal Site Search":
        initial_urls.extend(CUSTOM_DOMAINS)

    # Deduplicate and make sure URLs are unique
    initial_urls = list(set(initial_urls))
    
    st.write("Initial URLs to crawl:", initial_urls)
    
    if not initial_urls:
        st.warning("No valid starting URLs found.")
        st.stop()

    # Start crawling
    visited = set()
    start_time = time.time()
    
    with st.spinner('Crawling in progress...'):
        final_results = crawl_site(initial_urls, keywords, visited)

    # Check if crawl timed out
    if time.time() - start_time > CRAWL_TIMEOUT:
        st.warning("Crawl timed out.")

    # Update Google Sheets with the results
    open_added = update_sheet(sheet_open, list(set(final_results['open'])))
    form_added = update_sheet(sheet_form, list(set(final_results['form'])))

    # Display the results
    st.subheader("Crawling Complete")
    st.write(f"Added {len(open_added)} open access links.")
    st.write(f"Added {len(form_added)} form-based links.")

