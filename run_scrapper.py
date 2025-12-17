from scrapper.scrapper_engine import AssessmentScrapperEngine
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor , as_completed

ASE = AssessmentScrapperEngine()

# get possible assasment links
page_base_type_1_url = "https://www.shl.com/products/product-catalog/?start=12&type=1"
type_1_max_pages = 32
page_base_type_2_url = "https://www.shl.com/products/product-catalog/?start=12&type=2"
type_2_max_pages = 12

major_assessment_listing_pages_links_urls = [
    f"https://www.shl.com/products/product-catalog/?start={i}&type=1" for i in range(0, type_1_max_pages * 12)
] + [
    f"https://www.shl.com/products/product-catalog/?start={i}&type=2" for i in range(0, type_2_max_pages * 12)
]
print(f"Total major assessment listing pages to scrape: {len(major_assessment_listing_pages_links_urls)}")

def fetch_and_parse(page_url):
    soup = ASE.hit_page_and_get_soup(page_url)
    return ASE.parse_assessment_listings_links(soup)

def fetch_assessment_detail(link):
    try:
        soup = ASE.hit_page_and_get_soup(link)
        details = ASE.parse_assessment_details(soup)

        return {
            "link": link,
            "heading": details.get("heading"),
            "desc": details.get("desc"),
            "job_levels": details.get("job_levels"),
            "languages": details.get("languages"),
            "assessment_length": details.get("assessment_length"),
            "test_type": details.get("test_type"),
            "remote_testing": details.get("remote_testing"),
        }

    except Exception as e:
        return {
            "link": link,
            "error": str(e)
        }

all_assessment_links = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(fetch_and_parse, url)
        for url in major_assessment_listing_pages_links_urls[:250]
    ]

    for future in tqdm(as_completed(futures), total=len(futures),
                       desc="Fetching assessment links =>"):
        all_assessment_links.extend(future.result())

print(f"Total assessment links found: {len(all_assessment_links)}")

assessments_detail_dict = {
    "link":[],
    "heading":[],
    "desc":[],
    "job_levels":[],
    "languages":[],
    "assessment_length":[],
    "test_type":[],
    "remote_testing":[]
}

results = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [
        executor.submit(fetch_assessment_detail, link)
        for link in all_assessment_links
    ]

    for future in tqdm(as_completed(futures),
                       total=len(futures),
                       desc="Fetching assessment details =>"):
        results.append(future.result())

for r in results:
    if "error" in r:
        print(f"Failed: {r['link']} → {r['error']}")
        continue

    for k in assessments_detail_dict:
        assessments_detail_dict[k].append(r.get(k))

assessments_detail_df = pd.DataFrame(assessments_detail_dict)
assessments_detail_df.to_csv("data/assessments_details.csv", index=False)