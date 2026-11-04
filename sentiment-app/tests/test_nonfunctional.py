"""Non-functional requirements of the hosted app: response time, model loading, memory, size of
the store, security of the code and the store, maintainability and portability (NF-xx). Numbers
are printed so a run leaves a record; the asserts hold the budgets of the test plan."""
import importlib.util
import re
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import pytest

from conftest import APP_DIR, model_keys

pytestmark = pytest.mark.slow


def have(module):
    return importlib.util.find_spec(module) is not None


class TestPerformance:
    def test_nf_01_a_rerun_with_a_cached_encoding_is_fast(self, apptest):
        at = apptest
        at.selectbox[0].set_value("dialogue_network_fingpt_sentfin_grpo").run()
        at.text_area[0].input("Shares jumped after the company beat earnings forecasts.")
        at.button[0].click().run()
        for t in at.toggle:
            t.set_value(True)
        at.run()
        times = []
        for _ in range(5):
            t0 = time.time()
            at.run()
            times.append(time.time() - t0)
        print(f"rerun with every panel open: median {np.median(times):.2f} s, worst {max(times):.2f} s")
        assert np.median(times) < 1.0 and max(times) < 3.0

    def test_nf_02_every_model_loads_quickly_from_a_cold_cache(self, app):
        app["load_model"].clear()
        times = {}
        for key in model_keys():
            t0 = time.time()
            app["load_model"](key)
            times[key] = time.time() - t0
        print("load seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in times.items()))
        assert max(times.values()) < 1.0 and sum(times.values()) < 5.0

    def test_nf_03_a_reading_without_the_encoder_takes_milliseconds(self, app, catalog, golden_readings):
        x = golden_readings["X"][0]
        model, meta = app["load_model"]("dialogue_network"), catalog["dialogue_network"]
        t0 = time.time()
        for _ in range(100):
            app["run_head"](model, meta, x)
        per = (time.time() - t0) / 100
        wmodel, wmeta = app["load_model"]("words_fingpt"), catalog["words_fingpt"]
        t0 = time.time()
        for _ in range(100):
            app["run_words"](wmodel, wmeta, "Shares jumped after the company beat earnings forecasts, but the outlook stayed cautious.")
        wper = (time.time() - t0) / 100
        print(f"network forward pass {per * 1000:.2f} ms, word reading {wper * 1000:.2f} ms")
        assert per < 0.02 and wper < 0.02

    def test_nf_04_memory_for_all_fourteen_models_stays_small(self, app):
        app["load_model"].clear()
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        for key in model_keys():
            app["load_model"](key)
        peak = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        print(f"peak allocation while loading every model: {peak / 1e6:.0f} MB")
        assert peak < 400e6


class TestStoreAndCode:
    SOURCES = [APP_DIR / "app.py", APP_DIR / "build_models.py"]

    def test_nf_05_store_fits_in_a_repository(self):
        total = sum(f.stat().st_size for f in (APP_DIR / "models").iterdir())
        print(f"models/ is {total / 1e6:.1f} MB")
        assert total < 30e6

    def test_nf_06_weights_are_loaded_without_unpickling_anything(self):
        src = (APP_DIR / "app.py").read_text(encoding="utf-8")
        assert "allow_pickle=False" in src
        risky = re.compile(r"import pickle|pickle\.loads?\(|allow_pickle=True|(?<![\w.])eval\(|(?<![\w.])exec\(|os\.system|subprocess|yaml\.load\(|torch\.load\(")
        for f in self.SOURCES:
            hits = [m.group(0) for m in risky.finditer(f.read_text(encoding="utf-8"))]
            assert not hits, (f.name, hits)

    def test_nf_07_no_credentials_hosts_or_project_paths_in_what_is_shipped(self):
        files = list(self.SOURCES) + [APP_DIR / "README.md", APP_DIR / "requirements.txt", APP_DIR / "models" / "catalog.json"]
        bad = re.compile(r"ssh |/home/\w+|(?<![\w.:])/(?:[a-z_][\w.-]*/){3,}|\b\w+@\w+\.(edu|com)\b|password\s*=|api_key|secret\s*=|token\s*=\s*[\"']", re.I)
        for f in files:
            hits = [m.group(0) for m in bad.finditer(f.read_text(encoding="utf-8"))]
            assert not hits, (f.name, hits)

    def test_nf_08_the_only_network_access_is_the_encoder_download(self):
        src = (APP_DIR / "app.py").read_text(encoding="utf-8")
        # besides the encoder, the only address is the Power BI one the tab checks links against; the viewer's browser loads that report, not the app
        assert "http" not in src.lower().replace("huggingface", "").replace("https://app\\.powerbi\\.com/", "").replace("https://app.powerbi.com/", "")
        assert "requests" not in src and "urllib" not in src

    def test_nf_09_house_style_no_dashes_and_ascii_sources(self):
        for f in self.SOURCES + [APP_DIR / "README.md", APP_DIR / "models" / "catalog.json"]:
            text = f.read_text(encoding="utf-8")
            assert chr(0x2014) not in text and chr(0x2013) not in text, f.name
        catalog = (APP_DIR / "models" / "catalog.json").read_text(encoding="utf-8")
        assert all(ord(c) < 128 for c in catalog)

    def test_nf_10_theme_is_the_project_purple(self):
        cfg = (APP_DIR / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        assert 'primaryColor = "#7c4df5"' in cfg

    def test_nf_11_requirements_set_lower_bounds_for_every_dependency(self):
        lines = [l.strip() for l in (APP_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("--")]
        assert sorted(l.split(">=")[0] for l in lines) == ["numpy", "plotly", "streamlit", "torch", "transformers"]
        assert all(">=" in l for l in lines)


class TestMaintainabilityAndPortability:
    def test_nf_12_sources_compile_and_declare_a_minimum_python(self):
        for f in (APP_DIR / "app.py", APP_DIR / "build_models.py"):
            subprocess.run([sys.executable, "-m", "py_compile", str(f)], check=True)
        if not have("vermin"):
            pytest.skip("vermin not installed")
        run = subprocess.run([sys.executable, "-m", "vermin", "--no-tips", str(APP_DIR / "app.py"), str(APP_DIR / "build_models.py")], capture_output=True, text=True)
        out = run.stdout + run.stderr
        m = re.search(r"Minimum required versions:\s*([\d.]+)", out)
        if m is None:
            run = subprocess.run(["vermin", "--no-tips", str(APP_DIR / "app.py"), str(APP_DIR / "build_models.py")], capture_output=True, text=True)
            m = re.search(r"Minimum required versions:\s*([\d.]+)", run.stdout + run.stderr)
        assert m, out[-500:]
        print(f"minimum python by syntax: {m.group(1)} (Streamlit itself needs 3.9 or later)")
        assert tuple(int(x) for x in m.group(1).split(".")) <= (3, 10)

    def test_nf_13_cyclomatic_complexity_is_recorded_and_bounded(self):
        if not have("radon"):
            pytest.skip("radon not installed")
        from radon.complexity import cc_visit
        worst = []
        for f in (APP_DIR / "app.py", APP_DIR / "build_models.py"):
            for block in cc_visit(f.read_text(encoding="utf-8")):
                worst.append((block.complexity, f.name, block.name))
        worst.sort(reverse=True)
        print("most complex functions: " + ", ".join(f"{n} {c}" for c, _, n in worst[:5]))
        assert worst[0][0] <= 45  # network_figure, the drawing, is the known outlier (recorded in the defect log)
        assert sum(1 for c, _, _ in worst if c > 20) <= 1

    def test_nf_14_static_security_scan_has_no_high_findings(self):
        if not have("bandit"):
            pytest.skip("bandit not installed")
        run = subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", str(APP_DIR / "app.py"), str(APP_DIR / "build_models.py")], capture_output=True, text=True)
        import json
        report = json.loads(run.stdout or "{}")
        issues = [(i["test_id"], i["issue_severity"], i["line_number"]) for i in report.get("results", [])]
        print("bandit findings: " + (", ".join(f"{t} {s} line {l}" for t, s, l in issues) or "none"))
        assert not any(s == "HIGH" for _, s, _ in issues)
        assert {t for t, _, _ in issues} <= {"B615"}  # the unpinned encoder download, an open item in the defect log

    def test_nf_15_lint_is_clean(self):
        if not have("ruff"):
            pytest.skip("ruff not installed")
        run = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "E,F,W", "--ignore", "E501,E741", str(APP_DIR / "app.py"), str(APP_DIR / "build_models.py")],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout
