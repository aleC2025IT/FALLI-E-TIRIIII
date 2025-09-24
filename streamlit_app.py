# ================================
# Serie A Probabilities – 1-file app
# Fouls ≥2  |  Shots on Target ≥1
# ================================
# Niente librerie extra: solo Python + Streamlit!
# - Dati partite/giocatori: API-FOOTBALL (serve API_FOOTBALL_KEY)
# - Meteo stadio: Open-Meteo (gratis, no key)
#
# Come si usa: vedi istruzioni a fine file (sezione "ISTRUZIONI").
# ================================

import os, json, math, time
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import ssl
import streamlit as st

# --------- CONFIG BASE ----------
LEAGUE_ID = 135   # Serie A
SEASON    = 2025  # Stagione 2025/26
API_BASE  = "https://v3.football.api-sports.io"
# Usa prima la variabile d'ambiente/Secrets; se manca, usa la tua chiave fornita
API_KEY   = os.getenv("API_FOOTBALL_KEY", "605521ceda7756cc4cdb65f41e369e0d")

# Disabilita verifiche SSL su alcuni ambienti (più tollerante)
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# ------------- UTILS ------------
def http_get_json(url: str, headers: dict | None = None, params: dict | None = None) -> dict:
    if params:
        qs = urlencode({k: v for k, v in params.items() if v is not None})
        if "?" in url:
            url = url + "&" + qs
        else:
            url = url + "?" + qs
    req = Request(url, headers=headers or {})
    with urlopen(req, context=ssl_ctx, timeout=30) as resp:
        data = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(data)
    except:
        return {}

def api_get(path: str, params: dict) -> dict:
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_KEY
    }
    return http_get_json(f"{API_BASE}/{path}", headers=headers, params=params)

def parse_iso_utc(s: str) -> datetime:
    # API restituisce ISO con Z oppure offset; normalizziamo
    # Esempi: "2025-09-27T18:45:00+00:00" oppure "2025-09-27T18:45:00Z"
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def round_name_current() -> str:
    js = api_get("fixtures/rounds", {"league": LEAGUE_ID, "season": SEASON, "current": "true"})
    arr = js.get("response", [])
    return arr[0] if arr else "Regular Season - 1"

def fixtures_for_round(round_name: str) -> list[dict]:
    js = api_get("fixtures", {"league": LEAGUE_ID, "season": SEASON, "round": round_name})
    out = []
    for fx in js.get("response", []):
        fixture = fx.get("fixture", {})
        teams   = fx.get("teams", {})
        venue   = fixture.get("venue", {})
        out.append({
            "fixture_id": fixture.get("id"),
            "date_utc": parse_iso_utc(fixture.get("date")),
            "status": fixture.get("status", {}).get("short"),
            "referee": fixture.get("referee"),
            "home_id": teams.get("home", {}).get("id"),
            "home":    teams.get("home", {}).get("name"),
            "away_id": teams.get("away", {}).get("id"),
            "away":    teams.get("away", {}).get("name"),
            "venue_id": venue.get("id"),
            "venue":   venue.get("name"),
            "city":    venue.get("city"),
        })
    return out

def venue_coords(venue_id: int) -> tuple[float|None, float|None]:
    js = api_get("venues", {"id": venue_id})
    resp = js.get("response", [])
    if not resp: return (None, None)
    v = resp[0]
    return (v.get("latitude"), v.get("longitude"))

def open_meteo_at(lat: float, lon: float, ko_utc: datetime) -> dict:
    # Prendiamo 48h attorno al kickoff e scegliamo l'ora più vicina
    start = (ko_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    end   = (ko_utc + timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m,cloudcover",
        "timezone": "Europe/Rome",
        "start_date": start,
        "end_date": end,
    }
    js = http_get_json("https://api.open-meteo.com/v1/forecast", params=params)
    hourly = js.get("hourly", {})
    times = hourly.get("time", [])
    if not times: return {}
    # trova indice con tempo più vicino a ko_utc
    best_i, best_diff = 0, 10**9
    for i, t in enumerate(times):
        try:
            dt_local = datetime.fromisoformat(t)  # naive Europe/Rome
            dt_utc = dt_local - timedelta(hours=2)  # approx offset
            diff = abs((dt_utc - ko_utc).total_seconds())
            if diff < best_diff:
                best_diff, best_i = diff, i
        except:
            pass
    def pick(key):
        arr = hourly.get(key, [])
        return arr[best_i] if best_i < len(arr) else None
    return {
        "temperature_2m": pick("temperature_2m"),
        "precipitation":  pick("precipitation"),
        "windspeed_10m":  pick("windspeed_10m"),
        "cloudcover":     pick("cloudcover"),
        "time_local":     times[best_i] if best_i < len(times) else None
    }

def team_players_stats(team_id: int) -> list[dict]:
    # Stats per giocatore per 90: falli commessi, tiri in porta, minuti
    js = api_get("players", {"team": team_id, "season": SEASON, "league": LEAGUE_ID})
    out = []
    for item in js.get("response", []):
        player = item.get("player", {})
        for stt in item.get("statistics", []):
            if stt.get("league", {}).get("id") != LEAGUE_ID: 
                continue
            games = stt.get("games", {}) or {}
            shots = stt.get("shots", {}) or {}
            fouls = stt.get("fouls", {}) or {}
            minutes = games.get("minutes") or 0
            min_safe = max(1, minutes)
            sot = shots.get("on") or 0
            fcomm = fouls.get("committed") or 0
            out.append({
                "player_id": player.get("id"),
                "player":    player.get("name") or "",
                "position":  games.get("position"),
                "minutes":   minutes,
                "per90_sot": (sot * 90.0) / min_safe,
                "per90_fouls": (fcomm * 90.0) / min_safe
            })
    return out

def opponent_baseline_def(team_id: int) -> dict:
    # placeholder medio (migliorabile in futuro)
    return {"opp_sot_allowed": 4.0, "opp_fouls_drawn": 12.0}

def poisson_prob_ge_k(lam: float, k: int) -> float:
    lam = max(lam, 1e-9)
    s = 0.0
    for i in range(k):
        s += math.exp(-lam) * (lam**i) / math.factorial(i)
    return max(0.0, min(1.0, 1.0 - s))

def weather_adjust(weather: dict) -> tuple[float, float]:
    if not weather: return (1.0, 1.0)
    precip = weather.get("precipitation") or 0.0
    wind   = weather.get("windspeed_10m") or 0.0
    temp   = weather.get("temperature_2m") or 18.0
    fouls_mult, sot_mult = 1.0, 1.0
    if precip >= 1.0:    # pioggia ↑ falli, ↓ SOT
        fouls_mult *= 1.05
        sot_mult   *= 0.93
    if wind >= 25:       # vento forte ↓ SOT
        sot_mult   *= 0.90
    if temp <= 5:
        fouls_mult *= 1.03
        sot_mult   *= 0.95
    if temp >= 30:
        fouls_mult *= 1.02
        sot_mult   *= 0.96
    return (fouls_mult, sot_mult)

def best_name_match(name: str, stats: list[dict]) -> dict | None:
    name_low = name.strip().lower()
    for r in stats:
        if (r.get("player") or "").strip().lower() == name_low:
            return r
    token = name_low.split()[-1] if name_low else ""
    if token:
        for r in stats:
            pl = (r.get("player") or "").strip().lower()
            if (" " + token) in (" " + pl) or pl.endswith(" " + token) or pl == token:
                return r
    return None

def compute_for_squad(lineup: list[str], bench: list[str], stats: list[dict],
                      opp_def: dict, weather: dict, min_starter: int, min_bench: int) -> list[dict]:
    fmul, smul = weather_adjust(weather)
    out = []
    for name, is_starter in [(n, True) for n in lineup] + [(n, False) for n in bench]:
        minutes = min_starter if is_starter else min_bench
        rec = best_name_match(name, stats)
        per90_fouls = (rec.get("per90_fouls") if rec else 1.0)
        per90_sot   = (rec.get("per90_sot")   if rec else 0.30)
        lam_f = per90_fouls * (minutes/90.0) * (opp_def.get("opp_fouls_drawn",12.0)/12.0) * fmul
        lam_s = per90_sot   * (minutes/90.0) * (opp_def.get("opp_sot_allowed",4.0)/4.0) * smul
        p_f2  = poisson_prob_ge_k(lam_f, 2)
        p_s1  = 1.0 - math.exp(-max(lam_s, 1e-9))
        out.append({
            "player": name.strip(),
            "starter": is_starter,
            "exp_min": minutes,
            "lambda_fouls": round(lam_f, 3),
            "lambda_sot":   round(lam_s, 3),
            "P(≥2 fouls)":  round(p_f2, 3),
            "P(≥1 SOT)":    round(p_s1, 3),
        })
    return out

# ------------- UI -------------
st.set_page_config(page_title="Serie A • Probabili & Probabilities", layout="wide")
st.title("Serie A • Probabili & Probabilities (fouls≥2, SOT≥1)")

st.sidebar.header("Impostazioni")
key_ok = bool(API_KEY)
st.sidebar.write("API-FOOTBALL key:", "✅" if key_ok else "❌ manca (vedi istruzioni sotto)")

min_starter = st.sidebar.slider("Minuti attesi TITOLARI", 60, 95, 80, 5)
min_bench   = st.sidebar.slider("Minuti attesi PANCHINA", 10, 45, 30, 5)

round_in = st.sidebar.text_input("Giornata (lascia vuoto per auto)", value="")
if not key_ok:
    st.warning("Manca la chiave API_FOOTBALL_KEY. Vai in 'Manage app' → 'Secrets' e aggiungila (istruzioni sotto).")

# Scopri la giornata
if not round_in:
    try:
        round_in = round_name_current()
    except Exception as e:
        round_in = "Regular Season - 1"
st.subheader(f"Giornata: {round_in}")

# Carica partite
fixtures = []
try:
    fixtures = fixtures_for_round(round_in)
except Exception as e:
    st.error("Errore nel caricare le partite. Controlla la chiave API e riprova.")

if not fixtures:
    st.stop()

# Elenco selezionabile
labels = [f"{fx['home']} vs {fx['away']} — {fx['date_utc'].astimezone().strftime('%a %d %b %H:%M')}" for fx in fixtures]
selected = st.multiselect("Scegli le partite da analizzare", labels, default=labels)
selected_ids = {labels[i]: fixtures[i] for i in range(len(labels)) if labels[i] in selected}

st.info("Inserisci i nominativi: 'Titolari separati da virgole / Panchina separata da virgole'. Esempio: "
        "Maignan, Calabria, Thiaw, ... / Sportiello, Florenzi, Adli")

all_rows = []

for label, fx in selected_ids.items():
    home = fx["home"]; away = fx["away"]
    ko_utc = fx["date_utc"]
    st.markdown(f"### {home} (Casa)  vs  {away} (Trasferta)")
    col1, col2 = st.columns(2)

    # Meteo
    wx = {}
    if fx.get("venue_id"):
        try:
            lat, lon = venue_coords(int(fx["venue_id"]))
            if lat and lon:
                wx = open_meteo_at(float(lat), float(lon), ko_utc)
        except:
            wx = {}
    if wx:
        st.caption(f"Meteo vicino al kickoff ({wx.get('time_local')}): "
                   f"temp {wx.get('temperature_2m','?')}°C, pioggia {wx.get('precipitation','?')} mm, "
                   f"vento {wx.get('windspeed_10m','?')} km/h, nuvole {wx.get('cloudcover','?')}%")

    with col1:
        inp_home = st.text_input(f"{home} — 'Titolari / Panchina'", key=f"home_{fx['fixture_id']}", value="")
    with col2:
        inp_away = st.text_input(f"{away} — 'Titolari / Panchina'", key=f"away_{fx['fixture_id']}", value="")

    def parse_line(s: str) -> tuple[list[str], list[str]]:
        if not s.strip():
            return ([], [])
        parts = s.split("/")
        starters = [x.strip() for x in parts[0].split(",") if x.strip()]
        bench = []
        if len(parts) > 1:
            bench = [x.strip() for x in parts[1].split(",") if x.strip()]
        return (starters, bench)

    home_line, home_bench = parse_line(inp_home)
    away_line, away_bench = parse_line(inp_away)

    # Stats squadre
    try:
        home_stats = team_players_stats(fx["home_id"])
    except Exception as e:
        home_stats = []
    try:
        away_stats = team_players_stats(fx["away_id"])
    except Exception as e:
        away_stats = []

    opp_home = opponent_baseline_def(fx["away_id"])
    opp_away = opponent_baseline_def(fx["home_id"])

    if (home_line or home_bench) and home_stats:
        rows_h = compute_for_squad(home_line, home_bench, home_stats, opp_home, wx, min_starter, min_bench)
        for r in rows_h:
            r["match"] = f"{home} vs {away}"
            r["team"] = home
        all_rows.extend(rows_h)

    if (away_line or away_bench) and away_stats:
        rows_a = compute_for_squad(away_line, away_bench, away_stats, opp_away, wx, min_starter, min_bench)
        for r in rows_a:
            r["match"] = f"{home} vs {away}"
            r["team"] = away
        all_rows.extend(rows_a)

if not all_rows:
    st.stop()

# Ordinamenti
rows_fouls = sorted(all_rows, key=lambda x: (x["P(≥2 fouls)"], x["starter"]), reverse=True)
rows_sot   = sorted(all_rows, key=lambda x: (x["P(≥1 SOT)"],   x["starter"]), reverse=True)

tab1, tab2 = st.tabs(["Top: ≥2 falli", "Top: ≥1 tiro in porta"])

def render_table(rows: list[dict], cols: list[str]):
    show = [{c: r.get(c) for c in cols} for r in rows]
    st.table(show)

with tab1:
    render_table(rows_fouls, ["match","team","player","starter","P(≥2 fouls)","lambda_fouls","exp_min"])

with tab2:
    render_table(rows_sot, ["match","team","player","starter","P(≥1 SOT)","lambda_sot","exp_min"])

# CSV download minimal

def to_csv(rows: list[dict]) -> str:
    all_cols = ["match","team","player","starter","exp_min","lambda_fouls","lambda_sot","P(≥2 fouls)","P(≥1 SOT)"]
    out = [",".join(all_cols)]
    def esc(x): 
        s = str(x)
        return '"' + s.replace('"','""') + '"' if ("," in s or '"' in s or "
" in s) else s
    for r in rows:
        out.append(",".join(esc(r.get(c,"")) for c in all_cols))
    return "
".join(out)

csv_data = to_csv(all_rows).encode("utf-8")
st.download_button("Scarica CSV completo", data=csv_data, file_name="seriea_probabilities.csv", mime="text/csv")

st.caption("Modello: Poisson sugli eventi per 90', aggiustato per minuti attesi, tendenza avversario (placeholder) e meteo. "
           "Inserisci con cura i nomi per migliorare l'aggancio alle statistiche dell'API.")

# ================================
# ISTRUZIONI (A PROVA DI BAMBINO)
# ================================
# 1) Hai già la chiave API dentro il file. Meglio ancora: puoi metterla nei Secrets come
#      API_FOOTBALL_KEY = "605521ceda7756cc4cdb65f41e369e0d"
#    e l'app userà quella al posto del fallback.
# 2) Crea un repo su GitHub con questo unico file chiamato: streamlit_app.py
# 3) Fai Deploy su Streamlit Cloud scegliendo:
#      - Repository: il tuo
#      - Branch: main
#      - Main file path: streamlit_app.py
# 4) Nell'app:
#    - Lascia vuoto "Giornata" per prendere quella corrente.
#    - Per ogni partita incolla i nomi: "Titolare1, Titolare2, ... / Panchina1, Panchina2, ..."
#    - Guarda le 2 tabelle (≥2 falli, ≥1 tiro in porta) e, se vuoi, scarica il CSV.
