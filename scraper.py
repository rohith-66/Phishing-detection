import pymongo
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import os

# ------------------------------
# MongoDB setup
# ------------------------------
client = pymongo.MongoClient("mongodb://host.docker.internal:27017/")
db = client["phishing_db"]
collection = db["websites"]

# ------------------------------
# Selenium setup for headless Chrome in Docker
# ------------------------------
options = Options()
options.add_argument("--headless=new")  # use modern headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-software-rasterizer")
options.add_argument("--remote-debugging-port=9222")

service = Service(executable_path="/usr/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=options)

# ------------------------------
# Helper functions
# ------------------------------
def clean_url(url):
    """Fix obfuscated protocols and validate URL string."""
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.replace("hXXps://", "https://").replace("hXXp://", "http://")
    return url if url.startswith(("http://", "https://")) else None

def scrape_url(url, label):
    """Scrape a URL and store the content in MongoDB."""
    cleaned_url = clean_url(url)
    if cleaned_url is None:
        print(f"[SKIP] Invalid URL: {url}")
        return

    try:
        print(f"[SCRAPE] {cleaned_url} ({label})")
        driver.get(cleaned_url)

        # wait for <body> to be present (up to 10s)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        page_content = driver.find_element(By.TAG_NAME, "body").text[:5000]  # limit size

        data = {
            "url": cleaned_url,
            "content": page_content,
            "label": label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        collection.insert_one(data)
        print(f"[OK] Stored {cleaned_url}")
    except Exception as e:
        print(f"[FAIL] {cleaned_url}: {e}")

def get_urls_from_csv(file_path):
    """Reads phishing URLs from a CSV (second column)."""
    urls = []
    if not os.path.exists(file_path):
        print(f"[WARN] CSV file '{file_path}' not found.")
        return urls

    with open(file_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) > 1:
                urls.append(row[1].strip())
    return urls

# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    csv_file_path = "phishing_urls.csv"
    phishing_urls = get_urls_from_csv(csv_file_path)

    legitimate_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://www.bbc.com",
        "https://www.cnn.com",
    ]

    for url in phishing_urls:
        scrape_url(url, "phishing")

    for url in legitimate_urls:
        scrape_url(url, "legitimate")

    driver.quit()
    print("[DONE] Scraping finished.")
