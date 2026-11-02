"""Load test of the running server: several browsers read sentences at the same time (NF-L-xx).
Starts `streamlit run` on a free port with the stand-in encoder, drives it with Playwright, and
measures the time from pressing Predict to the result for every session. Skipped when Playwright
or its browser is not installed."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

SENTENCE = "Shares jumped after the company beat earnings forecasts, but the outlook stayed cautious."


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(app_with_fake_encoder):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(app_with_fake_encoder), "--server.headless", "true", "--server.port", str(port),
                             "--browser.gatherUsageStats", "false"], cwd=str(Path(app_with_fake_encoder).parent),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED="1", SENTIMENT_ENCODER=os.environ["SENTIMENT_ENCODER"]))
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(1)
    time.sleep(2)
    yield url
    proc.terminate()
    proc.wait(timeout=30)


def read_once(browser, url, model_label, text):
    """One user: open the page, pick a model, type, predict, wait for the result. Returns seconds."""
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    try:
        page.goto(url, wait_until="networkidle", timeout=90000)
        page.wait_for_selector("text=Sentiment prediction", timeout=90000)
        page.get_by_role("combobox").click()
        page.keyboard.type(model_label)  # the dropdown draws only the rows in view, so filter down to this one
        page.get_by_role("option", name=model_label).click()
        page.wait_for_timeout(500)
        page.locator("textarea").fill(text)
        t0 = time.time()
        page.get_by_role("button", name="Predict").click()
        page.wait_for_selector("text=Result", timeout=120000)
        page.wait_for_selector('[data-testid="stMetric"]', timeout=120000)
        return time.time() - t0
    finally:
        page.close()


def test_nf_l_01_one_user_gets_a_word_reading_quickly(server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        t = read_once(browser, server, "Word Logit, SEntFiN [MNLM]", SENTENCE)
        browser.close()
    print(f"single user, word model: {t:.2f} s to the result")
    assert t < 10.0


@pytest.mark.parametrize("n_users", [4, 8])
def test_nf_l_02_concurrent_users_are_all_served(server, n_users):
    from concurrent.futures import ThreadPoolExecutor
    labels = ["Word Logit, SEntFiN [MNLM]", "Word Logit, FinGPT [MNLM]", "FinGPT Network [deep learning]", "Dialogue Network [deep learning]"]

    def one(i):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                return read_once(browser, server, labels[i % len(labels)], f"{SENTENCE} Session {i}.")
            finally:
                browser.close()

    with ThreadPoolExecutor(max_workers=n_users) as pool:
        times = list(pool.map(one, range(n_users)))
    print(f"{n_users} concurrent users: median {np.median(times):.2f} s, slowest {max(times):.2f} s")
    assert len(times) == n_users and max(times) < 60.0
