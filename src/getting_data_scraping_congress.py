import requests
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import Dict, Set

def paragraph_mentions(text: str, keyword: str) -> bool:
    """
    Returns True if a <p> inside the text mentions {keyword}
    """
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]
    return any(keyword.lower() in paragraph.lower()
               for paragraph in paragraphs)


text = """<body><h1>Facebook</h1><p>Twitter</p>"""
assert paragraph_mentions(text, "twitter")      # is inside a <p>
assert not paragraph_mentions(text, "facebook") # not inside a <p>

# I put the relevant HTML file on GitHub. In order to fit
# the URL in the book I had to split it across two lines.
# Recall that whitespace-separated strings get concatenated.
url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, 'html5lib')

# regex to match only representative's web sites
regex = r"^https?://.*\.house\.gov/?$"

# Let's write some tests!
assert re.match(regex, "http://joel.house.gov")
assert re.match(regex, "https://joel.house.gov")
assert re.match(regex, "http://joel.house.gov/")
assert re.match(regex, "https://joel.house.gov/")
assert not re.match(regex, "joel.house.gov")
assert not re.match(regex, "http://joel.house.com")
assert not re.match(regex, "https://joel.house.gov/biography")

all_urls = [a["href"] 
            for a in soup("a")
            if a.has_attr("href")]

good_urls = {url for url in all_urls if re.match(regex, url)}

press_releases: Dict[str, Set[str]] = {}

for house_url in tqdm(good_urls):
    html = requests.get(house_url).text
    soup = BeautifulSoup(html, 'html5lib')
    pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

    press_releases[house_url] = pr_links


for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = ""
        if pr_link.startswith("/"):
            url = f"{house_url}/{pr_link}"
        else:
            url = pr_link
            
        text = requests.get(url).text
        
        if paragraph_mentions(text, 'data'):
            print(f"{house_url}")
            break # done with this house_url