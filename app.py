import streamlit as st
import requests
import time
import re
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2.service_account import Credentials
import gspread
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# ==================== CONFIGURATION ====================
RELEVANCE_THRESHOLD = 0.2
MAX_CRAWL_DEPTH = 2
CRAWL_TIMEOUT = 30 * 60  # 30 minutes
CUSTOM_DOMAINS = ["https://example1.com", "https://example2.com"]  # Replace with your actual domain list
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
    vectorizer = TfidfVectorizer().fit_transform(docs)
    score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    return score >= RELEVANCE_THRESHOLD

# ==================== LINK CLASSIFICATION ====================
def classify_link(html):
    if re.search(r'<input|<form|@|\.com', html, re.IGNORECASE):
        return 'form'
    return 'open'

# ==================== CRAWLING ====================
def get_links_from_page(driver, url):
    try:
        driver.get(url)
        time.sleep(2)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            full_url = urljoin(url, href)
            if urlparse(full_url).scheme in ['http', 'https']:
                links.add(full_url)
        return html, list(links)
    except Exception:
        return '', []

def crawl_site(start_urls, keywords, visited, depth=0):
    results = {'open': [], 'form': []}
    if depth > MAX_CRAWL_DEPTH:
        return results

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

    for url in start_urls:
        if url in visited:
            continue
        visited.add(url)
        html, links = get_links_from_page(driver, url)
        if is_relevant(html, keywords):
            category = classify_link(html)
            results[category].append(url)
        sub_results = crawl_site(links, keywords, visited, depth + 1)
        results['open'].extend(sub_results['open'])
        results['form'].extend(sub_results['form'])

    driver.quit()
    return results

# ==================== GOOGLE SEARCH API ====================
def google_search(query, api_key, cse_id):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
    try:
        response = requests.get(url)
        results = response.json().get("items", [])
        return [item["link"] for item in results if "link" in item]
    except:
        return []

# ==================== STREAMLIT APP ====================
st.set_page_config(page_title="Smart Web Crawler", layout="wide")
st.title("Smart Web Crawler with Google Sheets Integration")

keywords_input = st.text_input("Enter keywords (comma separated):")
option = st.selectbox("Choose search option", ["Google Search", "Internal Site Search", "Selenium + Scrapy"])
api_key = st.text_input("Google API Key (for Google Search option only):", type="password")
cse_id = st.text_input("Custom Search Engine ID (for Google Search option only):", type="password")
sheet_url_open = st.text_input("Enter Google Sheet URL for Open Access Links")
sheet_url_form = st.text_input("Enter Google Sheet URL for Form-Based Links")
credentials = load_credentials()

# Function to generate queries for Google Search (AND/OR combinations)
def generate_queries(keywords):
    queries = []
    keywords = [keyword.strip() for keyword in keywords]
    # Generate all combinations of AND/OR logic for keywords
    and_query = " AND ".join(keywords)
    or_query = " OR ".join(keywords)
    queries.append(and_query)
    queries.append(or_query)
    return queries

# Main logic
if st.button("Start Crawling") and keywords_input and credentials and sheet_url_open and sheet_url_form:
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    client = get_gsheet_client(credentials)
    sheet_open = client.open_by_url(sheet_url_open).sheet1
    sheet_form = client.open_by_url(sheet_url_form).sheet1

    # Collecting initial URLs
    initial_urls = []

    # 1. Get URLs from Google Search (Option 1)
    if option == "Google Search" or option == "Selenium + Scrapy":
        queries = generate_queries(keywords)
        for query in queries:
            google_urls = google_search(query, api_key, cse_id)
            initial_urls.extend(google_urls)
    
    # 2. Get URLs from Internal Domain List (Option 2)
    if option == "Internal Site Search" or option == "Selenium + Scrapy":
        initial_urls.extend(CUSTOM_DOMAINS)

    # Deduplicate and make sure URLs are unique
    initial_urls = list(set(initial_urls))
    
    # Check if there are starting URLs
    if not initial_urls:
        st.warning("No valid starting URLs found.")
        st.stop()

    # Start crawling with Selenium + Scrapy
    visited = set()
    start_time = time.time()
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
