"""Demo script for doomscroll_interface using Firefox.

Run with the workspace venv python:
    /Users/samuelbruin/vlc-project/.venv/bin/python demo_doomscroll_firefox.py
"""

import time
import traceback

from doomscroll_interface import (
    start_doomscroll,
    scroll_up,
    scroll_down,
    toggle_play,
    like,
    dislike,
    open_comments,
    close_comments,
    zoom,
    close_driver,
)


def run_demo():
    print("Starting demo: launching browser and opening YouTube Shorts (Firefox)...")
    try:
        start_doomscroll(mode="selenium", browser="firefox", headless=False, wait=6)
        print(
            "Opened Shorts. Pausing longer to let UI settle and for consent handlers to run..."
        )
        time.sleep(4)

        print("Toggling play/pause")
        toggle_play()
        time.sleep(2)

        print("Scrolling to next short")
        scroll_up()
        time.sleep(2.5)

        print("Scrolling back to previous short")
        scroll_down()
        time.sleep(2.5)

        print(
            "Attempting to like the current short (may require signed-in account in the browser)"
        )
        ok = like()
        print("Like clicked?", ok)
        time.sleep(2)

        print("Opening comments")
        try:
            open_comments()
        except Exception as e:
            print("open_comments failed:", e)
        time.sleep(3)

        print("Closing comments")
        try:
            close_comments()
        except Exception as e:
            print("close_comments failed:", e)
        time.sleep(2)

        print("Toggling fullscreen")
        zoom()
        time.sleep(2)
        print("Exiting fullscreen")
        zoom()
        time.sleep(2)

    except Exception:
        print("Demo encountered an error:")
        traceback.print_exc()
    finally:
        print("Cleaning up: closing driver")
        try:
            close_driver()
        except Exception:
            pass


if __name__ == "__main__":
    run_demo()
