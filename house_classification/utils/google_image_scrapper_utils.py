import os
import time
import requests  # Import the requests module
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from PIL import Image
import io

import random
import string

def generate_random_string(length=10):
    """
    Generates a random alphanumeric string of a specified length.

    Args:
        length (int): The desired length of the string. Defaults to 10.

    Returns:
        str: A randomly generated alphanumeric string.
    """
    # Define the set of characters to choose from (letters and digits)
    characters = string.ascii_letters + string.digits
    
    # Use a list comprehension and random.choice to build the string
    random_string = ''.join(random.choice(characters) for _ in range(length))
    
    return random_string

def get_images_from_google(driver, delay, max_images):
    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    image_urls = set()
    skips = 0

    while len(image_urls) + skips < max_images:
        scroll_down(driver)

        thumbnails = driver.find_elements(By.CSS_SELECTOR, '.rg_i, .Q4LuWd')
        for img in thumbnails[len(image_urls) + skips:max_images]:
            try:
                img.click()
                time.sleep(delay)
            except:
                continue

            images = driver.find_elements(By.CSS_SELECTOR, '.r48jcc, .pT0Scc, .iPVvYb')
            for image in images:
                if image.get_attribute('src') in image_urls:
                    max_images += 1
                    skips += 1
                    break

                if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))
                    print(f"Found {len(image_urls)}")

    return image_urls

def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)

        # Check if the image can be identified and is in a compatible format
        if image.format not in ["JPEG", "PNG"]:
            print(f"Skipping image with unsupported format: {url}")
            return

        file_path = os.path.join(download_path, file_name)  # Use os.path.join to ensure correct path

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print("Success")
    except Exception as e:
        print('FAILED -', e)


