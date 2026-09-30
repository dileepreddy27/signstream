from playwright.sync_api import sync_playwright
import json
from pathlib import Path

Path("work").mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1120}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8010")
    page.wait_for_function("document.querySelector('#evaluation').textContent.includes('200')")
    for label in ["wave", "sweep", "raise", "circle"]:
        page.select_option("#motion", label)
        page.click("#play")
        page.wait_for_function(
            "label => document.querySelector('#prediction').textContent === label", arg=label
        )
    assert page.locator("#tokens").inner_text() == "wave · sweep · raise · circle"
    page.screenshot(path="docs/demo.png", full_page=True)
    page.click("#clear")
    page.wait_for_function("document.querySelector('#tokens').textContent.includes('Your motion')")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path="work/mobile.png", full_page=True)
    assert not errors, errors
    print(
        json.dumps(
            {
                "browser": "Chromium",
                "fixtures": 4,
                "clear": "passed",
                "mobile_overflow": "none",
                "page_errors": errors,
            }
        )
    )
    browser.close()
