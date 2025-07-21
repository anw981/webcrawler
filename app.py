# streamlit_app.py
import streamlit as st
import requests
import time
import re
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from google.oauth2.service_account import Credentials
import gspread
import os

# ==================== CONFIGURATION ====================
RELEVANCE_THRESHOLD = 0.3
MAX_CRAWL_DEPTH = 2
CRAWL_TIMEOUT = 5 * 60
CUSTOM_DOMAINS = [
    "https://rbi.org.in",
    "https://npci.org.in"
]  # Editable list
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Use secrets.toml instead of hardcoding
SERVICE_ACCOUNT_JSON = st.secrets["google"]

# Hardcoded sheet URLs
OPEN_SHEET_URL = st.secrets["sheets"]["open"]
FORM_SHEET_URL = st.secrets["sheets"]["form"]

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_gsheet_client():
    credentials = Credentials.from_service_account_info(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    return gspread.authorize(credentials)

def update_sheet(sheet, rows):
    existing = set(sheet.col_values(2))
    new_rows = [row for row in rows if row[1] not in existing]
    if new_rows:
        sheet.append_rows(new_rows)
    return new_rows

def is_relevant_bert(text, keywords):
    try:
        content_emb = model.encode(text, convert_to_tensor=True)
        keyword_emb = model.encode(" ".join(keywords), convert_to_tensor=True)
        score = util.pytorch_cos_sim(content_emb, keyword_emb).item()
        return score >= RELEVANCE_THRESHOLD, score
    except Exception as e:
        st.write(f"BERT error: {e}")
        return False, 0

def classify_link(html):
    return 'form' if re.search(r'<input|<form|@|\.com', html, re.IGNORECASE) else 'open'

def get_links_requests(url):
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        links = list(set(urljoin(url, tag['href']) for tag in soup.find_all('a', href=True)))
        return html, links
    except Exception as e:
        st.write(f"Requests error: {e}")
        return '', []

def crawl_site(start_urls, keywords, visited, depth=0, start_time=None):
    if start_time is None:
        start_time = time.time()
    results = {'open': [], 'form': []}

    if depth > MAX_CRAWL_DEPTH or (time.time() - start_time > CRAWL_TIMEOUT):
        return results

    for url in start_urls:
        if url in visited or (time.time() - start_time > CRAWL_TIMEOUT):
            continue
        visited.add(url)

        html, links = get_links_requests(url)
        if not html:
            continue

        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string.strip() if soup.title else "(No Title)"
        summary = soup.find('p').get_text().strip() if soup.find('p') else ""
        main_text = " ".join([p.get_text() for p in soup.find_all('p')])
        is_rel, score = is_relevant_bert(main_text, keywords)

        st.write(f"Visited: {url} | Relevance Score: {score:.3f}")

        if is_rel:
            category = classify_link(html)
            results[category].append([title, url, category, summary, round(score, 3)])

        sub_results = crawl_site(links, keywords, visited, depth + 1, start_time=start_time)
        results['open'].extend(sub_results['open'])
        results['form'].extend(sub_results['form'])

    return results

def google_search(query, api_key, cse_id):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
        r = requests.get(url)
        return [item['link'] for item in r.json().get('items', []) if 'link' in item]
    except:
        return []

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Smart Web Crawler", layout="wide")
st.title("BERT-Powered Smart Web Crawler")

keywords_input = st.text_input("Enter keywords (comma separated):")
search_mode = st.selectbox("Select crawling method", ["Google Search", "Internal Domains Only"])
api_key = st.text_input("Google API Key:", type="password")
cse_id = st.text_input("Custom Search Engine ID:", type="password")

if st.button("Start Crawling") and keywords_input:
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    client = get_gsheet_client()
    sheet_open = client.open_by_url(OPEN_SHEET_URL).sheet1
    sheet_form = client.open_by_url(FORM_SHEET_URL).sheet1

    initial_urls = []
    if search_mode == "Google Search":
        if not api_key or not cse_id:
            st.error("API key and CSE ID required for Google Search")
            st.stop()
        queries = [" AND ".join([f'\"{k}\"' for k in keywords]), " OR ".join([f'\"{k}\"' for k in keywords])]
        for q in queries:
            initial_urls.extend(google_search(q, api_key, cse_id))
    else:
        initial_urls = CUSTOM_DOMAINS

    st.write("Initial URLs:", initial_urls)
    visited = set()
    start_time = time.time()
    with st.spinner("Crawling..."):
        final_results = crawl_site(initial_urls, keywords, visited, start_time=start_time)

    st.success("Crawling Done!")
    st.write("Preview:", final_results)
    open_written = update_sheet(sheet_open, final_results['open'])
    form_written = update_sheet(sheet_form, final_results['form'])
    st.write(f"Added {len(open_written)} open links, {len(form_written)} form links.")
