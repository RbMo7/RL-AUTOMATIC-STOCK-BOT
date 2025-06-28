from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import csv
import time

def scrape_price_history_table(driver):
    try:
        table = driver.find_element(By.CSS_SELECTOR, "table.table.table-bordered.table-striped.table-hover")
        print("✅ Found price history table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        data = []
        for row in rows:
            cells = row.find_elements(By.XPATH, "./th|./td")
            row_data = [cell.text.strip() for cell in cells]
            data.append(row_data)
        return data
    except NoSuchElementException as e:
        print(f"❌ Table not found: {e}")
        return []

def save_to_csv(data, filename):
    headers = ["Index","Date", "LTP", "% Change", "High", "Low", "Open", "Qty.", "Turnover"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in data:
            if len(row) == 9:
                writer.writerow(row)
            else:
                print(f"⚠️ Skipping invalid row (len={len(row)}): {row}")

def click_next_page(driver):
    try:
        next_link = driver.find_element(By.XPATH, "//a[@title='Next Page']")
        # Scroll element into view
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_link)
        time.sleep(0.5)
        # Click via JavaScript (bypasses interception)
        driver.execute_script("arguments[0].click();", next_link)
        time.sleep(2)  # wait for page load after clicking
        return True
    except NoSuchElementException:
        print("No 'Next Page' link found.")
        return False
    except Exception as e:
        print(f"Error clicking next page: {e}")
        return False

if __name__ == "__main__":
    url = "https://merolagani.com/CompanyDetail.aspx?symbol=PHCL#0"
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    print("⏳ Waiting 10 seconds for you to load the table manually...")
    time.sleep(10)

    all_data = []
    page_count = 1

    while True:
        print(f"📄 Scraping Page {page_count}...")
        data = scrape_price_history_table(driver)

        if not data or len(data) < 2:
            print("⚠️ No data found on this page. Stopping.")
            break

        if page_count == 1:
            all_data.extend(data[1:])  # skip header
        else:
            all_data.extend(data[1:])  # skip repeated headers

        print(f"✅ Scraped {len(data) - 1} rows")

        if not click_next_page(driver):
            print("🚫 No more pages. Finished scraping.")
            break

        page_count += 1

    save_to_csv(all_data, "price_history_phcl.csv")
    print("📁 All data saved to price_history_ghl.csv")

    driver.quit()
