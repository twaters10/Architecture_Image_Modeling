import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from utils.google_image_scrapper_utils import *

# Ask the user for the search query
search_query = input("Enter your Google Images search query: ")

# Create the 'imgs/' directory if it doesn't exist
download_path = f"../architectural_style_images/{search_query}"
os.makedirs(download_path, exist_ok=True)

# Create a Chrome driver
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Open the Google Images search page with the provided search query
search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
driver.get(search_url)

# Perform image scraping and downloading
urls = get_images_from_google(driver, 2, 10)

for i, url in enumerate(urls):
    download_image(download_path, url, generate_random_string() + ".jpg")

# Close the driver instance
driver.quit()