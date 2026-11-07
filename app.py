"""
Twitter Sentiment Analysis - Root Launcher
Enables running `streamlit run app.py` directly from workspace root.
"""
import os
import sys
from pathlib import Path

target_dir = Path(__file__).resolve().parent / "Twitter-Sentiment-Analysis"
if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))

os.chdir(target_dir)

app_path = target_dir / "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_path), "exec")
    exec(code)
