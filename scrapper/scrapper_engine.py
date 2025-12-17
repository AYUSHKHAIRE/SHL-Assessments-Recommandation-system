from .logger_config import logger
import requests
from bs4 import BeautifulSoup

class AssessmentScrapperEngine:
    def __init__(self):
        self.data_dir = "../data"

    def hit_page_and_get_soup(self , page_url):
        # logger.debug(f"Hitting page: {page_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(page_url, headers=headers)
        if response.status_code == 200:
            # logger.debug(f"Successfully fetched page: {page_url}")
            return BeautifulSoup(response.content, "html.parser")
        else:
            logger.error(f"Failed to fetch page: {page_url} with status code {response.status_code}")
            return None

    def parse_assessment_listings_links(self, soup):
        # logger.debug("Parsing assessment listings from the page.")
        assessment_links = []
        listing_elements = soup.find_all("td", class_="custom__table-heading__title")
        for element in listing_elements:
            link = element.find("a")["href"]
            # logger.debug(f"Found assessment: {link}")
            assessment_links.append(f"https://www.shl.com{link}")
        return assessment_links
    
    def parse_assessment_details(self, soup):
        # logger.debug("Parsing assessment details from the page.")
        assessment_details = {}
        heading = soup.find("div", class_="row content__container typ").get_text(strip=True)
        desc = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")[0].find("p").get_text(strip=True)
        job_levels = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")[1].find("p").get_text(strip=True)
        languages = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")[2].find("p").get_text(strip=True)
        assessment_length = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")[3].find("p").get_text(strip=True) 
        test_type = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")[3].find("div",class_ = "d-flex").find_all("p")[0].get_text(strip=True)
        remote_testing = soup.find_all("div", class_="product-catalogue-training-calendar__row typ")[3].find("div",class_ = "d-flex").find_all("p")[1].get_text(strip=True)
        assessment_details["heading"] = heading
        assessment_details["desc"] = desc
        assessment_details["job_levels"] = job_levels
        assessment_details["languages"] = languages
        assessment_details["assessment_length"] = assessment_length
        assessment_details["test_type"] = test_type
        assessment_details["remote_testing"] = remote_testing
        return assessment_details