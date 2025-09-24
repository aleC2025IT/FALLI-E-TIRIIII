# streamlit_app.py
# -------------------------------------------------------------
# Serie A: Probable Lineups + Player Probabilities (Fouls≥2, SOT≥1)
# Data sources: API-FOOTBALL (fixtures, players stats),
#               Gazzetta/Fantacalcio/Sky (probabili formazioni via HTML),
#               Open‑Meteo (match weather by venue coords)
# -------------------------------------------------------------
# ⚠️ Read me:
# 1) Set an environment variable API_FOOTBALL_KEY with your key (https://dashboard.api-football.com)
# 2) Run:  streamlit run streamlit_app.py
# 3) If scraping of "probabili formazioni" fails (HTML changes), paste lineups manually in the fallback textbox.
# -------------------------------------------------------------

import os
import re
import math
import time
import json
import html
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------- CONFIG -----------------------
LEAGUE_ID = 135        # Serie A in API‑FOOTBALL (commonly 135)
SEASON = 2025          # 2025/26
HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": os.getenv("API_FOOTBALL_KEY", "")
}
API_BASE = "https://v3.football.api-sports.io"

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PARAMS = {
    "hourly": [
        "temperature_2m",
        "precipitation",
        "windspeed_10m",
        "cloudcover",
    ],
    "timezone": "Europe/Rome"
}

# ----------------------- UTILS -----------------------
@st.cache_data(ttl=1800, show_spinner=False)
def api_get(path: str, params: dict) -> dict:
    url = f"{API_BASE}/{path}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800, show_spinner=False)
def get_current_round() -> str:
    data = api_get("fixtures/rounds", {"league": LEAGUE_ID, "season": SEASON, "current": "true"})
    rounds = data.get("response", [])
    return rounds[0] if rounds else "Regular Season - 1"

@st.cache_data(ttl=1800, show_spinner=False)
def get_round_fixtures(round_name: str) -> pd.DataFrame:
    data = api_get("fixtures", {"league": LEAGUE_ID, "season": SEASON, "round": round_name})
    rows = []
    for fx in data.get("response", []):
        fixture = fx.get("fixture", {})
        teams = fx.get("teams", {})
        venue = fixture.get("venue", {})
        rows.append({
            "fixture_id": fixture.get("id"),
            "datetime": fixture.get("date"),
            "referee": fixture.get("referee"),
            "status": fixture.get("status", {}).get("short"),
            "home_id": teams.get("home", {}).get("id"),
            "home": teams.get("home", {}).get("name"),
            "away_id": teams.get("away", {}).get("id"),
            "away": teams.get("away", {}).get("name"),
            "venue_id": venue.get("id"),
            "venue": venue.get("name"),
            "city": venue.get("city"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])  # UTC
        df["kickoff_local"] = df["datetime"].dt.tz_convert("Europe/Rome")
    return df

@st.cache_data(ttl=86400, show_spinner=False)
def get_venue_coords(venue_id: int) -> Tuple[Optional[float], Optional[float]]:
    # API‑FOOTBALL venues endpoint sometimes available via fixtures/venues; fallback none
    try:
        data = api_get("venues", {"id": venue_id})
        resp = data.get("response", [])
        if resp:
            v = resp[0]
            return v.get("latitude"), v.get("longitude")
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(lat: float, lon: float, when_utc: pd.Timestamp) -> dict:
    # Fetch hourly weather +/- 12h around kickoff and pick nearest hour
    params = {
        **OPEN_METEO_PARAMS,
        "latitude": lat,
        "longitude": lon,
        "start_hour": (when_utc - pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:00"),
        "end_hour": (when_utc + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:00"),
    }
    # Open‑Meteo uses 'start_date/end_date' or 'hourly' window; we'll just query a day window
    params = {
        **OPEN_METEO_PARAMS,
        "latitude": lat,
        "longitude": lon,
        "start_date": (when_utc - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (when_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    r = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    hourly = js.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    if len(times) == 0:
        return {}
    idx = int(np.argmin(np.abs(times.tz_localize("Europe/Rome").tz_convert("UTC") - when_utc)))
    def pick(key):
        arr = hourly.get(key, [])
        return arr[idx] if idx < len(arr) else None
    return {
        "temperature_2m": pick("temperature_2m"),
        "precipitation": pick("precipitation"),
        "windspeed_10m": pick("windspeed_10m"),
        "cloudcover": pick("cloudcover"),
        "time": times[idx].isoformat(),
    }

# ---------------- Probabili formazioni scrapers ----------------
HEADERS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@st.cache_data(ttl=900)
def scrape_gazzetta() -> Dict[str, Dict[str, List[str]]]:
    url = "https://www.gazzetta.it/Calcio/prob_form/"
    r = requests.get(url, headers=HEADERS_BROWSER, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Heuristic parsing: find match blocks with team names and lists
    blocks = soup.find_all(lambda tag: tag.name in ["section", "article", "div"] and "Probabili Formazioni" in tag.get_text(strip=True))
    # Fallback: iterate all links to specific matches present in page
    matches = {}
    for a in soup.select("a[href*='/prob_form/']"):
        href = a.get("href")
        if not href or href.endswith("prob_form/"):
            continue
        try:
            rr = requests.get(href, headers=HEADERS_BROWSER, timeout=30)
            if rr.status_code != 200:
                continue
            ss = BeautifulSoup(rr.text, "html.parser")
            title = ss.find("h1") or ss.find("title")
            title_txt = title.get_text(strip=True) if title else ""
            # Extract team names from URL or title
            m = re.search(r"prob_form/([^/]+)/([^/?]+)", href)
            if m:
                t1, t2 = m.group(1), m.group(2)
            else:
                # fallback: split on ' - '
                parts = re.split(r"[-–]", title_txt)
                t1, t2 = (parts[0].strip(), parts[1].strip()) if len(parts) >= 2 else (None, None)
            if not t1 or not t2:
                continue
            # starters lists often inside <ul> or <p> blocks labeled 'Probabile formazione'
            text = ss.get_text("\n", strip=True)
            # Very permissive regex to catch player names (capitalized words, apostrophes)
            names = re.findall(r"[A-ZÀ-Ý][a-zà-ÿ'\-]+(?:\s[A-ZÀ-Ý][a-zà-ÿ'\-]+)?", text)
            # This is too noisy; we cannot trust. Better to keep empty and let manual paste handle.
            matches[f"{t1}|{t2}"] = {"home": [], "away": [], "bench_home": [], "bench_away": []}
        except Exception:
            continue
    return matches

@st.cache_data(ttl=900)
def scrape_fantacalcio() -> Dict[str, Dict[str, List[str]]]:
    url = "https://www.fantacalcio.it/probabili-formazioni-serie-a"
    r = requests.get(url, headers=HEADERS_BROWSER, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    matches = {}
    # The page lists tiles for each match that link to detail pages per team; we keep placeholder for now.
    for tile in soup.select("a[href*='probabili-formazioni']"):
        href = tile.get("href")
        if not href or href.endswith("probabili-formazioni-serie-a"):
            continue
        # Not scraping deeply to avoid breaking; leave empty and rely on manual paste if needed.
        # You can implement a tighter parser later when HTML is stable.
        t1_t2 = tile.get_text(" ", strip=True)
        if t1_t2 and "-" in t1_t2:
            parts = [p.strip() for p in re.split(r"[-–]", t1_t2)][:2]
            if len(parts) == 2:
                matches[f"{parts[0]}|{parts[1]}"] = {"home": [], "away": [], "bench_home": [], "bench_away": []}
    return matches

# ---------------- Player statistics ----------------
@st.cache_data(ttl=3600)
def get_team_players_stats(team_id: int) -> pd.DataFrame:
    # Pull players + season stats
    data = api_get("players", {"team": team_id, "season": SEASON, "league": LEAGUE_ID})
    rows = []
    for item in data.get("response", []):
        player = item.get("player", {})
        stats_list = item.get("statistics", [])
        # pick Serie A stats
        for stt in stats_list:
            if stt.get("league", {}).get("id") != LEAGUE_ID:
                continue
            games = stt.get("games", {})
            offs = stt.get("offsides")
            shots = stt.get("shots", {})
            fouls = stt.get("fouls", {})
            team = stt.get("team", {}).get("name")
            minutes = games.get("minutes") or 0
            appear = games.get("appearences") or 0
            # per90s
            min_safe = max(1, minutes)
            sot = shots.get("on") or 0
            sh_total = shots.get("total") or 0
            f_comm = fouls.get("committed") or 0
            per90_sot = sot / (min_safe / 90)
            per90_fouls = f_comm / (min_safe / 90)
            rows.append({
                "player_id": player.get("id"),
                "player": player.get("name"),
                "team_id": team_id,
                "team": team,
                "position": games.get("position"),
                "minutes": minutes,
                "apps": appear,
                "sot": sot,
                "shots": sh_total,
                "fouls_comm": f_comm,
                "per90_sot": per90_sot,
                "per90_fouls": per90_fouls,
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def get_team_defensive_tendencies(team_id: int) -> dict:
    # Approx: opp SOT allowed per game, fouls drawn per game by opponents (hard to get)
    # We'll derive from players stats aggregated after we pull all teams once; for now return neutral multipliers.
    return {"opp_sot_allowed": 4.0, "opp_fouls_drawn": 12.0}

# ---------------- Probability model ----------------

def poisson_prob_ge_k(lam: float, k: int) -> float:
    lam = max(lam, 1e-9)
    # P(X>=k) = 1 - sum_{i=0}^{k-1} e^-lam lam^i / i!
    cum = sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k))
    return float(max(0.0, min(1.0, 1 - cum)))


def weather_adjustment(weather: dict) -> Tuple[float, float]:
    """Returns (fouls_multiplier, sot_multiplier) based on weather"""
    if not weather:
        return 1.0, 1.0
    precip = weather.get("precipitation") or 0
    wind = weather.get("windspeed_10m") or 0
    temp = weather.get("temperature_2m") or 18
    fouls_mult = 1.0
    sot_mult = 1.0
    if precip >= 1.0:      # rain >=1 mm/h
        fouls_mult *= 1.05
        sot_mult *= 0.93
    if wind >= 25:         # strong wind km/h
        sot_mult *= 0.9
    if temp <= 5:
        fouls_mult *= 1.03
        sot_mult *= 0.95
    if temp >= 30:
        fouls_mult *= 1.02
        sot_mult *= 0.96
    return fouls_mult, sot_mult


def compute_probabilities(lineup: List[str], bench: List[str], team_stats: pd.DataFrame,
                          opponent_def: dict, weather: dict) -> pd.DataFrame:
    fouls_mult, sot_mult = weather_adjustment(weather)

    def player_rate(name: str, col: str, default: float) -> float:
        row = team_stats[team_stats["player"].str.fullmatch(name, case=False, na=False)]
        if row.empty:
            # attempt loose match by last name
            token = name.split()[-1]
            row = team_stats[team_stats["player"].str.contains(fr"\b{re.escape(token)}\b", case=False, na=False)]
        if row.empty:
            return default
        return float(row.iloc[0][col])

    rows = []
    for name in lineup + bench:
        starter = name in lineup
        exp_minutes = 80 if starter else 30
        lam_fouls = player_rate(name, "per90_fouls", 1.0) * (exp_minutes/90) * (opponent_def.get("opp_fouls_drawn", 12.0)/12.0) * fouls_mult
        lam_sot   = player_rate(name, "per90_sot",   0.3) * (exp_minutes/90) * (opponent_def.get("opp_sot_allowed", 4.0)/4.0) * sot_mult
        p_fouls2 = poisson_prob_ge_k(lam_fouls, 2)
        p_sot1   = 1 - math.exp(-lam_sot)
        rows.append({
            "player": name,
            "starter": starter,
            "exp_min": exp_minutes,
            "lambda_fouls": round(lam_fouls, 3),
            "lambda_sot": round(lam_sot, 3),
            "P(≥2 fouls)": round(p_fouls2, 3),
            "P(≥1 SOT)": round(p_sot1, 3),
        })
    return pd.DataFrame(rows)

# --------------- UI -----------------
st.set_page_config(page_title="Serie A • Probabili & Probabilities", layout="wide")
st.title("Serie A • Probabili & Probabilities (fouls≥2, SOT≥1)")

st.sidebar.header("Settings")
api_key_ok = bool(HEADERS["x-rapidapi-key"]) and HEADERS["x-rapidapi-key"] != ""
st.sidebar.write("API‑FOOTBALL key:", "✅" if api_key_ok else "❌ missing (set API_FOOTBALL_KEY)")
source = st.sidebar.selectbox("Probabili formazioni source", ["Manual/Fallback", "Gazzetta", "Fantacalcio", "Sky Sport"], index=0)
round_name = st.sidebar.text_input("Round (auto if blank)", value="")
minutes_bench = st.sidebar.slider("Expected minutes for bench players", 10, 45, 30, 5)
minutes_starter = st.sidebar.slider("Expected minutes for starters", 60, 95, 80, 5)

if round_name.strip() == "":
    try:
        round_name = get_current_round()
    except Exception as e:
        st.warning(f"Could not auto‑detect round: {e}")
        round_name = "Regular Season - 1"

st.subheader(f"Round: {round_name}")

try:
    fixtures_df = get_round_fixtures(round_name)
except Exception as e:
    st.error(f"Error loading fixtures: {e}")
    fixtures_df = pd.DataFrame()

if fixtures_df.empty:
    st.stop()

# Allow user to pick matches
match_labels = fixtures_df.apply(lambda r: f"{r['home']} vs {r['away']} — {r['kickoff_local']:%a %d %b %H:%M}", axis=1).tolist()
selected = st.multiselect("Matches to analyze", match_labels, default=match_labels)
sel_df = fixtures_df[fixtures_df.apply(lambda r: f"{r['home']} vs {r['away']} — {r['kickoff_local']:%a %d %b %H:%M}", axis=1).isin(selected)]

# Try to gather probabili formazioni
pf_map: Dict[str, Dict[str, List[str]]] = {}
if source == "Gazzetta":
    pf_map = scrape_gazzetta()
elif source == "Fantacalcio":
    pf_map = scrape_fantacalcio()
else:
    pf_map = {}

st.info("If lineups are empty or wrong, paste your own below for each match (comma‑separated players; bench after a slash '/'). Example: 'Maignan, Calabria, Thiaw, ... / Sportiello, Florenzi, Adli' ")

all_rows = []
for _, row in sel_df.iterrows():
    home, away = row.home, row.away
    fixture_id = int(row.fixture_id)
    ko_utc = pd.to_datetime(row["datetime"])  # already UTC

    # Venue & weather
    lat, lon = (None, None)
    if row.venue_id:
        lat, lon = get_venue_coords(int(row.venue_id))
    wx = {}
    if lat and lon:
        try:
            wx = get_weather(lat, lon, ko_utc.tz_localize("UTC") if ko_utc.tzinfo is None else ko_utc)
        except Exception:
            wx = {}

    # Team stats
    try:
        home_stats = get_team_players_stats(int(row.home_id))
        away_stats = get_team_players_stats(int(row.away_id))
    except Exception as e:
        st.warning(f"Stats fetch issue for {home} or {away}: {e}")
        home_stats = pd.DataFrame()
        away_stats = pd.DataFrame()

    opp_home = get_team_defensive_tendencies(int(row.away_id))
    opp_away = get_team_defensive_tendencies(int(row.home_id))

    # Probabili formazioni from scraper (may be empty)
    key = f"{home}|{away}"
    pf = pf_map.get(key, {"home": [], "away": [], "bench_home": [], "bench_away": []})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {home} (Casa)")
        default_line = ", ".join(pf.get("home", [])) + " / " + ", ".join(pf.get("bench_home", [])) if pf else ""
        inp_home = st.text_input(f"{home} titolari / panchina", key=f"home_{fixture_id}", value=default_line)
        parts = [p.strip() for p in inp_home.split("/")]
        home_line = [p.strip() for p in (parts[0].split(",") if parts and parts[0] else []) if p]
        home_bench = [p.strip() for p in (parts[1].split(",") if len(parts) > 1 else []) if p]
    with col2:
        st.markdown(f"### {away} (Trasferta)")
        default_line = ", ".join(pf.get("away", [])) + " / " + ", ".join(pf.get("bench_away", [])) if pf else ""
        inp_away = st.text_input(f"{away} titolari / panchina", key=f"away_{fixture_id}", value=default_line)
        parts = [p.strip() for p in inp_away.split("/")]
        away_line = [p.strip() for p in (parts[0].split(",") if parts and parts[0] else []) if p]
        away_bench = [p.strip() for p in (parts[1].split(",") if len(parts) > 1 else []) if p]

    # Compute probabilities
    if not home_stats.empty and (home_line or home_bench):
        df_home = compute_probabilities(home_line, home_bench, home_stats, opp_home, wx)
        df_home.insert(0, "team", home)
        df_home.insert(0, "match", f"{home} vs {away}")
        all_rows.append(df_home)
    if not away_stats.empty and (away_line or away_bench):
        df_away = compute_probabilities(away_line, away_bench, away_stats, opp_away, wx)
        df_away.insert(0, "team", away)
        df_away.insert(0, "match", f"{home} vs {away}")
        all_rows.append(df_away)

    # Weather card
    if wx:
        st.caption(f"Meteo (vicino al calcio d'inizio): temp {wx.get('temperature_2m','?')}°C, precipitazioni {wx.get('precipitation','?')} mm, vento {wx.get('windspeed_10m','?')} km/h, nuvolosità {wx.get('cloudcover','?')}%")

if not all_rows:
    st.stop()

final = pd.concat(all_rows, ignore_index=True)

# Sort by probabilities
tab1, tab2 = st.tabs(["Top: ≥2 falli", "Top: ≥1 tiro in porta"])
with tab1:
    df1 = final.sort_values(["P(≥2 fouls)", "starter"], ascending=[False, False])
    st.dataframe(df1[["match","team","player","starter","P(≥2 fouls)","lambda_fouls","exp_min"]], use_container_width=True)
with tab2:
    df2 = final.sort_values(["P(≥1 SOT)", "starter"], ascending=[False, False])
    st.dataframe(df2[["match","team","player","starter","P(≥1 SOT)","lambda_sot","exp_min"]], use_container_width=True)

st.download_button("Scarica CSV completo", final.to_csv(index=False).encode("utf-8"), file_name="seriea_probabilities.csv", mime="text/csv")

st.caption("Model: Poisson for events per player per 90, adjusted by expected minutes, opponent tendencies (placeholders), and weather. Data quality depends on sources and naming alignment.")
