"""
Central configuration for Agent Incentive Scenario Simulator.
"""
import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

# --- Synthetic Data ---
N_AGENTS = 5000
RANDOM_SEED = 42
SEGMENT_DIST = {
    "Star": 0.10,
    "Stable": 0.50,
    "Developing": 0.25,
    "At-Risk": 0.15,
}

# --- PostgreSQL (local dev) ---
USE_CSV_FALLBACK = os.getenv("USE_CSV_FALLBACK", "false").lower() == "true"
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5434"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "incentive_sim")
POSTGRES_USER = os.getenv("POSTGRES_USER", "sim_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sim_secret")

# --- Monte Carlo ---
MC_ITERATIONS = 10_000
MC_SEED = 42

# --- LLM Backend ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" | "anthropic"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# --- Streamlit ---
DASHBOARD_TITLE = "Agent Incentive Scenario Simulator"
DASHBOARD_ICON = "📊"
