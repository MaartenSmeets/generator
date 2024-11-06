import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_chromedriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Remove this if you want to see the browser
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    try:
        logger.debug("Initializing undetected_chromedriver.")
        driver = uc.Chrome(options=chrome_options, use_subprocess=True)  # use_subprocess=True can help isolate processes
        logger.debug("Launching Chrome.")
        driver.get("https://www.google.com")
        logger.debug(f"Page title: {driver.title}")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        try:
            driver.quit()
            logger.debug("Driver quit successfully.")
        except:
            pass

if __name__ == "__main__":
    test_chromedriver()
