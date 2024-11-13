import chromedriver_autoinstaller
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--host-resolver-rules='MAP * ~NOTFOUND , EXCLUDE localhost'")
chrome_options.add_argument("--dns-prefetch-disable")
chrome_options.add_argument("--disable-async-dns")

# Install ChromeDriver matching the installed Chrome version
chromedriver_path = chromedriver_autoinstaller.install()  # Automatically detects version
logger.debug(f"Using ChromeDriver at {chromedriver_path}")

# Initialize WebDriver
try:
    driver = uc.Chrome(executable_path=chromedriver_path, options=chrome_options)
    logger.info("WebDriver initialized successfully.")

    # Navigate to a test page
    test_url = "https://www.example.com"
    driver.get(test_url)
    logger.info(f"Page title: {driver.title}")
    print(f"Successfully accessed {test_url}. Page title: {driver.title}")

except Exception as e:
    logger.exception("Error during WebDriver initialization or page load.")
finally:
    # Clean up
    if 'driver' in locals():
        driver.quit()
