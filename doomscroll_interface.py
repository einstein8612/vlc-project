import time
import webbrowser

from selenium.webdriver.common.action_chains import ActionChains

_driver = None
_mode = None


def start_doomscroll(
    mode: str = "selenium",
    browser: str = "chrome",
    headless: bool = False,
    wait: float = 2.0,
):
    global _driver, _mode
    _mode = mode
    url = "https://www.youtube.com/shorts/R2mwCdVb7lE"

    if mode == "native":
        webbrowser.open(url)
        _driver = None
        time.sleep(wait)
        return None

    # Selenium backend: support Chrome or Firefox
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
    except Exception as e:
        raise RuntimeError(
            "Selenium backend requested but selenium is not installed. Install with: pip install selenium webdriver-manager"
        ) from e

    if browser.lower() == "chrome":
        try:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from webdriver_manager.chrome import ChromeDriverManager
        except Exception:
            raise RuntimeError(
                "Chrome webdriver manager not available. Install webdriver-manager."
            )

        chrome_options = ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        svc = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=svc, options=chrome_options)

    elif browser.lower() == "firefox":
        try:
            from selenium.webdriver.firefox.service import Service as FirefoxService
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            from webdriver_manager.firefox import GeckoDriverManager
        except Exception:
            raise RuntimeError(
                "Firefox webdriver manager not available. Install webdriver-manager."
            )

        firefox_options = FirefoxOptions()
        if headless:
            try:
                firefox_options.headless = True
            except Exception:
                firefox_options.add_argument("--headless")
        
        svc = FirefoxService(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=svc, options=firefox_options)

    else:
        raise ValueError("Unsupported browser: choose 'chrome' or 'firefox'")

    print("Visiting brainrot")
    driver.get(url)
    try:
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import (
            NoSuchElementException,
            ElementClickInterceptedException,
        )

        consent_xpaths = [
            "//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'i agree')]",
            "//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept all')]",
            "//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept')]",
            "//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'agree')]",
            "//button[contains(translate(@aria-label,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept')]",
            "//button[contains(translate(@title,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept')]",
            "//input[@type='submit' and (contains(translate(@value,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept') or contains(translate(@value,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'agree'))]",
        ]

        for xp in consent_xpaths:
            try:
                el = driver.find_element(By.XPATH, xp)
                try:
                    el.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", el)
                time.sleep(1)
                print("Clicked consent button")
                break
            except NoSuchElementException:
                continue
            except Exception:
                continue
    except Exception:
        pass
    time.sleep(wait)
    _driver = driver
    return driver


def _ensure_driver():
    if _mode != "selenium":
        raise RuntimeError(
            "This action requires selenium backend. Call start_doomscroll(mode='selenium')"
        )
    if _driver is None:
        raise RuntimeError(
            "Selenium driver not started. Call start_doomscroll() first."
        )


def scroll_up():
    """Move to the next short (scroll down the feed).

    For selenium backend this sends PAGE_DOWN to the page body. For native mode it sends a PageDown keypress.
    """
    global _driver, _mode
    if _mode == "native":
        try:
            import pyautogui

            pyautogui.press("pagedown")
            return
        except Exception as e:
            raise RuntimeError(
                "pyautogui is required for native mode: pip install pyautogui"
            ) from e

    _ensure_driver()
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By

    body = _driver.find_element(By.TAG_NAME, "body")
    body.send_keys(Keys.PAGE_DOWN)


def scroll_down():
    """Move to the previous short (scroll up the feed).

    Sends PAGE_UP.
    """
    global _driver, _mode
    if _mode == "native":
        try:
            import pyautogui

            pyautogui.press("pageup")
            return
        except Exception as e:
            raise RuntimeError(
                "pyautogui is required for native mode: pip install pyautogui"
            ) from e

    _ensure_driver()
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By

    body = _driver.find_element(By.TAG_NAME, "body")
    body.send_keys(Keys.PAGE_UP)


def toggle_play():
    """Toggle play/pause for the current short by sending 'k' (works in YouTube web player)."""
    global _driver, _mode
    if _mode == "native":
        try:
            import pyautogui

            pyautogui.press("k")
            return
        except Exception as e:
            raise RuntimeError(
                "pyautogui is required for native mode: pip install pyautogui"
            ) from e

    _ensure_driver()
    from selenium.webdriver.common.by import By

    body = _driver.find_element(By.TAG_NAME, "body")
    body.send_keys("k")


def zoom():
    """Toggle fullscreen with 'f'."""
    global _driver, _mode
    if _mode == "native":
        try:
            import pyautogui

            pyautogui.press("f")
            return
        except Exception as e:
            raise RuntimeError(
                "pyautogui is required for native mode: pip install pyautogui"
            ) from e

    _ensure_driver()
    from selenium.webdriver.common.by import By

    body = _driver.find_element(By.TAG_NAME, "body")
    body.send_keys("f")


def open_comments():
    """Open the comment section for the current short.

    Selenium backend: attempts to locate a comments button and click it.
    Native backend: sends 'c' (best-effort) or suggests using selenium.
    """
    global _driver, _mode
    if _mode == "native":
        try:
            import pyautogui

            pyautogui.press("c")
            return
        except Exception:
            raise RuntimeError(
                "Open comments not supported reliably in native mode. Use selenium mode."
            )

    _ensure_driver()
    # Try a few selectors that commonly match the comments button on Shorts
    from selenium.common.exceptions import (
        NoSuchElementException,
        ElementClickInterceptedException,
    )

    possible_xpaths = [
        # explicit comments button on some layouts
        "//yt-icon-button[@id='comments-button']",
        # button with aria-label containing comment(s)
        "//button[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'comment')]",
        # links/anchors that open comment panels
        "//a[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'comment')]",
        # data-tooltip or title attributes mentioning comments
        "//button[contains(translate(@data-tooltip, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'comment')]",
        "//button[contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'comment')]",
        # generic comments region trigger
        "//*[@id='comments']",
        "//ytd-comments",
    ]

    from selenium.webdriver.common.by import By

    for xp in possible_xpaths:
        try:
            el = _driver.find_element(By.XPATH, xp)
            _driver.execute_script("arguments[0].scrollIntoView(true);", el)
            time.sleep(0.2)
            try:
                el.click()
            except ElementClickInterceptedException:
                # fallback to JS click
                _driver.execute_script("arguments[0].click();", el)
            return
        except NoSuchElementException:
            continue
        except Exception:
            # try next selector
            continue

    # Fallback: try opening the comments panel by focusing the player and pressing 'c' then wait
    try:
        from selenium.webdriver.common.by import By

        body = _driver.find_element(By.TAG_NAME, "body")
        body.send_keys("c")
        return
    except Exception:
        raise RuntimeError("Could not open comments - selector heuristics failed.")


def close_comments():
    """Close comments. Works by pressing Escape or clicking a close area."""
    global _driver, _mode
    if _mode == "native":
        try:
            import pyautogui

            pyautogui.press("esc")
            return
        except Exception:
            raise RuntimeError(
                "Close comments not supported reliably in native mode. Use selenium mode."
            )

    _ensure_driver()
    try:
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import (
            NoSuchElementException,
            ElementClickInterceptedException,
        )

        # Try close button(s) first - common labels: 'Close', 'X', aria-label 'Close'
        close_xpaths = [
            # Specific Shorts comments close button (absolute xpath supplied by user)
            "/html/body/ytd-app/div[1]/ytd-page-manager/ytd-shorts/div[4]/div[2]/ytd-engagement-panel-section-list-renderer[1]/div[1]/ytd-engagement-panel-title-header-renderer/div[3]/div[6]/ytd-button-renderer/yt-button-shape/button",
            "//button[contains(translate(@aria-label,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'close')][1]",
            "//button[contains(translate(@title,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'close')][1]",
            "//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'close')][1]",
            "//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'x')][1]",
            "//yt-icon-button[contains(translate(@aria-label,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'close')]",
            "//div[@id='comments']//button[contains(translate(@aria-label,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'close')][1]",
        ]

        for xp in close_xpaths:
            try:
                el = _driver.find_element(By.XPATH, xp)
                try:
                    el.click()
                except ElementClickInterceptedException:
                    _driver.execute_script("arguments[0].click();", el)
                return
            except NoSuchElementException:
                continue
            except Exception:
                continue

        # If no close button found, press Escape as a last resort
        body = _driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ESCAPE)
        return
    except Exception:
        # Try clicking an area outside comments
        try:
            _driver.execute_script("window.scrollTo(0, 0);")
            return
        except Exception:
            raise RuntimeError("Could not close comments programmatically.")


def _click_like_dislike(like=True):
    """Internal helper to click like or dislike button."""
    global _driver
    _ensure_driver()
    from selenium.common.exceptions import NoSuchElementException
    from selenium.webdriver.common.by import By

    # Build robust xpath list for like/dislike. We search aria-label, title, data-tooltip, and toggle renderers.
    target_text = "like" if like else "dislike"
    tt = target_text
    xpaths = [
        f"//button[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tt}')][1]",
        f"//button[contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tt}')][1]",
        f"//button[contains(translate(@data-tooltip, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tt}')][1]",
        # ytd-toggle-button-renderer wrapper commonly used by YouTube for like/dislike
        f"//ytd-toggle-button-renderer//button[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tt}')][1]",
        f"//ytd-toggle-button-renderer//button[contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tt}')][1]",
        # fallback: look for elements containing the text (less strict)
        f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{tt}') and (self::button or self::yt-icon-button or name()='div')][1]",
    ]

    for xp in xpaths:
        try:
            el = _driver.find_element(By.XPATH, xp)
            _driver.execute_script("arguments[0].scrollIntoView(true);", el)
            time.sleep(0.12)
            try:
                el.click()
            except Exception:
                # fallback to JS click
                _driver.execute_script("arguments[0].click();", el)
            
            actions = ActionChains(_driver)
            actions.move_to_element_with_offset(el, 100, 0).click().perform()

            return True
        except NoSuchElementException:
            continue
        except Exception:
            continue

    # If not found, return False
    return False


def like():
    """Like the current short.

    Returns True if action was performed, False if button couldn't be found.
    """
    global _mode
    if _mode == "native":
        raise RuntimeError(
            "Like action is not supported in native mode reliably. Use selenium mode."
        )
    return _click_like_dislike(like=True)


def dislike():
    """Dislike the current short.

    Returns True if action was performed, False otherwise.
    """
    global _mode
    if _mode == "native":
        raise RuntimeError(
            "Dislike action is not supported in native mode reliably. Use selenium mode."
        )
    return _click_like_dislike(like=False)


def close_driver():
    """Close and quit the selenium driver if running."""
    global _driver, _mode
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
    _driver = None
    _mode = None


if __name__ == "__main__":
    # Simple interactive demo
    print("doomscroll_interface demo. Launching selenium-backed YouTube Shorts...")
    try:
        start_doomscroll(mode="selenium", headless=False, wait=3)
    except RuntimeError as e:
        print("Failed to start selenium backend:", e)
        print("Try: pip install selenium webdriver-manager")
        raise

    print(
        "Commands available: scroll_up(), scroll_down(), toggle_play(), like(), dislike(), open_comments(), close_comments(), zoom(), close_driver()"
    )
