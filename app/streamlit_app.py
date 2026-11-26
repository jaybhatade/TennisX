"""
Module 6: Streamlit Web UI (redesigned)
=======================================
Interactive win probability predictor for ATP tennis matches.

Launch from the project root (so .streamlit/config.toml is picked up):
    streamlit run app/streamlit_app.py

Page flow (unchanged from the original):
  1. Title and short description
  2. Players: two dropdowns plus last known rank cards
  3. Match context: Surface, Court, Series, Round, Best of
  4. Predict button and the prediction result
  5. How the model works
  6. Disclaimer

Player ranks and points are looked up from the most recent entry in
atp_tennis.csv where the player appears as Player_1 or Player_2.
Player names are NEVER passed to the model. Only the numeric and encoded
difference features (rank_diff, pts_diff, etc.) are used.

Design system notes (landing-page-design skill, Part B):
  - One typeface (Geist), weights 400 to 600, no italics.
  - Type sizes snap to Tailwind steps and use named text styles.
  - Spacing only from the 2/4/8/12/16/24/32/40/48/64/80/96 scale.
  - Flat dark backgrounds from the approved list, no background gradients.
  - Phosphor icons, base easing curve, transform and opacity animation only.
  - Loading skeletons, empty, error and stale states for every dynamic area.
"""

import html
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "atp_tennis.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")

# ---------------------------------------------------------------------------
# Ordinal mapping for Round (must match Module 2)
# ---------------------------------------------------------------------------
ROUND_ORDINAL = {
    "1st Round":     1,
    "2nd Round":     2,
    "3rd Round":     3,
    "4th Round":     4,
    "Quarterfinals": 5,
    "Round Robin":   5,
    "Semifinals":    6,
    "The Final":     7,
}

# OHE baseline categories (the ones dropped by drop_first=True in Module 2)
# Surface baseline = Carpet; Court baseline = Indoor; Series baseline = ATP250
# A row with all OHE cols = 0 means: Carpet / Indoor / ATP250

STALE_AFTER_DAYS = 365   # last match older than this (vs newest match in data) is flagged

# ---------------------------------------------------------------------------
# Design tokens: text styles (size, line height, weight) from the type scale
# ---------------------------------------------------------------------------
TEXT_STYLES = {
    "heading-1": (60, 60, 600),
    "heading-3": (36, 40, 600),
    "heading-5": (24, 32, 600),
    "title-2":   (18, 28, 500),
    "title-3":   (16, 24, 500),
    "title-4":   (14, 20, 500),
    "body-1":    (16, 24, 600),
    "body-3":    (16, 24, 400),
    "body-4":    (14, 20, 600),
    "body-6":    (14, 20, 400),
    "caption-3": (12, 16, 600),
    "caption-4": (12, 16, 500),
}

_TEXT_STYLE_CSS = "\n".join(
    f".stApp .{name}{{font-size:{size}px;line-height:{lh}px;font-weight:{wt};}}"
    for name, (size, lh, wt) in TEXT_STYLES.items()
)

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap");
@import url("https://unpkg.com/@phosphor-icons/web@2.1.1/src/regular/style.css");

:root{
  --tx-bg:#000000; --tx-surface:#181818; --tx-surface-2:#1F1F1F;
  --tx-surface-3:#272727; --tx-surface-4:#313131;
  --tx-line:#313131; --tx-text:#FFFFFF; --tx-muted:#9B9B9B;
  --tx-accent:#D4F53C; --tx-accent-hover:#E4FF73; --tx-warn:#F5C451;
  --tx-ease:cubic-bezier(0.22,1,0.36,1);
  --tx-s-25:2px; --tx-s-50:4px; --tx-s-75:8px; --tx-s-100:12px; --tx-s-200:16px;
  --tx-s-300:24px; --tx-s-400:32px; --tx-s-500:40px; --tx-s-600:48px;
  --tx-s-700:64px; --tx-s-800:80px; --tx-s-900:96px;
  --tx-font:'Geist', ui-sans-serif, system-ui, sans-serif;
  --tx-mono:'Geist Mono', ui-monospace, monospace;
}

/* ---------- Base ---------- */
.stApp{background:var(--tx-bg);color:var(--tx-text);font-family:var(--tx-font);font-style:normal;}
.stApp p,.stApp li,.stApp label,.stApp input,.stApp button,.stApp table,.stApp summary,
.stApp [data-baseweb="select"] *,
div[data-baseweb="popover"] li,div[data-baseweb="popover"] li *{font-family:var(--tx-font);font-style:normal;}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stMainBlockContainer"],.block-container{
  max-width:736px;padding:var(--tx-s-600) var(--tx-s-200) var(--tx-s-900);
}
[data-testid="stVerticalBlock"]{gap:var(--tx-s-200);}
[data-testid="stHorizontalBlock"]{gap:var(--tx-s-200);}
[data-testid="stMarkdownContainer"] p{margin:0;}
.stApp hr{margin:var(--tx-s-200) 0;border:0;border-top:1px solid var(--tx-line);}
.stApp :focus-visible{outline:2px solid var(--tx-accent);outline-offset:2px;}
.ph{display:inline-block;font-style:normal;line-height:1;}

/* ---------- Named text styles ---------- */
__TEXT_STYLES__
.stApp .tx-muted{color:var(--tx-muted);}
.stApp .tx-num{font-variant-numeric:tabular-nums;}
.stApp .tx-balance{text-wrap:balance;}
.stApp .tx-pretty{text-wrap:pretty;}

/* ---------- Hero ---------- */
.tx-hero{display:flex;flex-direction:column;gap:var(--tx-s-200);align-items:flex-start;}
.tx-brand{display:inline-flex;align-items:center;gap:var(--tx-s-75);color:var(--tx-accent);}
.tx-brand .ph{font-size:24px;}
.stApp .tx-grad{
  max-width:680px;width:fit-content;
  background:linear-gradient(90deg,#FFFFFF,#9B9B9B);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;color:transparent;text-wrap:balance;
}
.stApp .tx-sub{max-width:680px;color:var(--tx-muted);text-wrap:pretty;}
.tx-proof{display:flex;flex-wrap:wrap;gap:var(--tx-s-75);margin-top:var(--tx-s-75);}
.tx-chip{
  display:inline-flex;align-items:center;gap:var(--tx-s-75);
  padding:var(--tx-s-50) var(--tx-s-100);border:1px solid var(--tx-line);
  border-radius:9999px;background:var(--tx-surface);color:var(--tx-text);
}
.tx-chip .ph{color:var(--tx-accent);font-size:16px;}

/* ---------- Sections ---------- */
.tx-section{display:flex;align-items:center;gap:var(--tx-s-100);margin-top:var(--tx-s-200);}
.tx-section .ph{font-size:24px;color:var(--tx-accent);}

/* ---------- Cards ---------- */
.tx-card{
  background:var(--tx-surface);border:1px solid var(--tx-line);border-radius:16px;
  transition:border-color 300ms var(--tx-ease);
}
.tx-stat{padding:var(--tx-s-300);display:flex;flex-direction:column;gap:var(--tx-s-50);height:100%;}
.tx-stat:hover{border-color:var(--tx-muted);}
.tx-statfoot{display:flex;flex-wrap:wrap;align-items:center;gap:var(--tx-s-75);margin-top:var(--tx-s-50);}
.tx-badge{
  display:inline-flex;align-items:center;gap:var(--tx-s-50);
  padding:var(--tx-s-25) var(--tx-s-75);border-radius:8px;
  background:var(--tx-surface-3);color:var(--tx-text);
}
.tx-badge .ph{font-size:14px;}
.tx-badge.is-warn{color:var(--tx-warn);}
.tx-badge.is-good{color:var(--tx-accent);}
.tx-note{margin-top:var(--tx-s-75);}

/* ---------- Alerts and states ---------- */
.tx-alert{
  display:flex;gap:var(--tx-s-100);align-items:flex-start;padding:var(--tx-s-200);
  border:1px solid var(--tx-warn);border-radius:12px;background:var(--tx-surface);
}
.tx-alert .ph{font-size:24px;color:var(--tx-warn);flex:none;}
.tx-empty{
  display:flex;flex-direction:column;align-items:center;text-align:center;
  gap:var(--tx-s-100);padding:var(--tx-s-500) var(--tx-s-400);
}
.tx-empty .tx-icon{
  display:flex;align-items:center;justify-content:center;width:48px;height:48px;
  border-radius:9999px;background:var(--tx-surface-2);color:var(--tx-accent);font-size:24px;
}
.tx-empty .tx-copy{max-width:420px;}

/* ---------- Result ---------- */
.tx-result{padding:var(--tx-s-75);}
.tx-tiles{display:grid;grid-template-columns:1fr 1fr;gap:var(--tx-s-75);}
.tx-tile{
  display:flex;flex-direction:column;gap:var(--tx-s-50);padding:var(--tx-s-200);
  background:var(--tx-surface-2);border:1px solid var(--tx-line);border-radius:8px;min-width:0;
}
.tx-tile.is-winner{border-color:var(--tx-accent);}
.tx-tile .tx-name{display:flex;align-items:center;gap:var(--tx-s-75);overflow-wrap:anywhere;}
.tx-tile .tx-name .ph{color:var(--tx-accent);font-size:18px;flex:none;}
.tx-barblock{padding:var(--tx-s-200) var(--tx-s-200) 0;}
.tx-bar{display:flex;gap:var(--tx-s-25);height:8px;border-radius:9999px;overflow:hidden;background:var(--tx-surface-3);}
.tx-seg{height:100%;min-width:0;}
.tx-seg-l{transform-origin:left center;}
.tx-seg-r{transform-origin:right center;}
.tx-barlabels{display:flex;justify-content:space-between;gap:var(--tx-s-200);margin-top:var(--tx-s-75);}
.tx-barlabels span{overflow-wrap:anywhere;}
.tx-verdict{display:flex;flex-wrap:wrap;align-items:center;gap:var(--tx-s-100);padding:var(--tx-s-200);}
.tx-verdict .ph{font-size:24px;color:var(--tx-accent);}
.tx-verdict .tx-badge{margin-left:auto;}

/* ---------- Details (feature inputs) ---------- */
.tx-details{border:1px solid var(--tx-line);border-radius:12px;background:var(--tx-surface);}
.tx-details summary{
  display:flex;align-items:center;gap:var(--tx-s-75);padding:var(--tx-s-100) var(--tx-s-200);
  cursor:pointer;list-style:none;border-radius:12px;
  transition:background-color 300ms var(--tx-ease);
}
.tx-details summary::-webkit-details-marker{display:none;}
.tx-details summary:hover{background:var(--tx-surface-2);}
.tx-details .tx-caret{margin-left:auto;transition:transform 500ms var(--tx-ease);}
.tx-details[open] .tx-caret{transform:rotate(180deg);}
.tx-details-body{padding:0 var(--tx-s-200) var(--tx-s-200);max-height:320px;overflow:auto;}

/* ---------- Tables ---------- */
.tx-table-wrap{overflow-x:auto;border:1px solid var(--tx-line);border-radius:12px;background:var(--tx-surface);}
.stApp .tx-table{width:100%;min-width:480px;border-collapse:collapse;}
.stApp .tx-table th,.stApp .tx-table td{
  text-align:left;vertical-align:top;padding:var(--tx-s-100) var(--tx-s-200);
  border:0;border-bottom:1px solid var(--tx-line);background:transparent;
}
.stApp .tx-table th{color:var(--tx-muted);font-size:12px;line-height:16px;font-weight:500;}
.stApp .tx-table td{font-size:14px;line-height:20px;font-weight:400;}
.stApp .tx-table tr:last-child td{border-bottom:0;}
.stApp .tx-code{font-family:var(--tx-mono);font-weight:500;}
.tx-facts{display:grid;grid-template-columns:1fr 1fr;gap:var(--tx-s-200);}
.tx-fact{padding:var(--tx-s-300);display:flex;flex-direction:column;gap:var(--tx-s-50);}
.tx-disclaimer{
  display:flex;gap:var(--tx-s-100);align-items:flex-start;padding:var(--tx-s-200);
  border:1px solid var(--tx-line);border-radius:12px;
}
.tx-disclaimer .ph{font-size:24px;color:var(--tx-warn);flex:none;}

/* ---------- Skeletons ---------- */
.tx-skel{background:var(--tx-surface-3);border-radius:8px;animation:tx-pulse 900ms var(--tx-ease) infinite alternate;}
.tx-skel-grid{display:grid;grid-template-columns:1fr 1fr;gap:var(--tx-s-200);}

/* ---------- Widgets: labels, selects ---------- */
.stApp [data-testid="stWidgetLabel"] p{
  font-size:14px;line-height:20px;font-weight:500;color:var(--tx-muted);
}
.stApp div[data-baseweb="select"] > div{
  background:var(--tx-surface-2);border:1px solid var(--tx-line);border-radius:8px;min-height:40px;
  color:var(--tx-text);font-size:16px;line-height:24px;font-weight:400;
  transition:background-color 300ms var(--tx-ease),border-color 300ms var(--tx-ease);
}
.stApp div[data-baseweb="select"] > div:hover{background:var(--tx-surface-3);}
.stApp div[data-baseweb="select"] > div:focus-within{border-color:var(--tx-accent);outline:2px solid var(--tx-accent);outline-offset:2px;}
.stApp div[data-baseweb="select"] input{color:var(--tx-text);}
div[data-baseweb="popover"] > div{background:var(--tx-surface-2);border:1px solid var(--tx-line);border-radius:8px;}
div[data-baseweb="popover"] ul{background:var(--tx-surface-2);}
div[data-baseweb="popover"] li{background:transparent;color:var(--tx-text);transition:background-color 300ms var(--tx-ease);}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li[aria-selected="true"]{background:var(--tx-surface-4);}

/* ---------- Predict button ---------- */
.stApp .stButton{width:100%;}
.stApp .stButton > button{
  width:100%;min-height:40px;padding:var(--tx-s-75) var(--tx-s-100);border-radius:8px;
  border:1px solid var(--tx-accent);background:var(--tx-accent);color:#000000;
  transition:background-color 300ms var(--tx-ease),transform 500ms var(--tx-ease);
}
.stApp .stButton > button p{color:inherit;font-size:16px;line-height:24px;font-weight:600;}
.stApp .stButton > button:hover:not(:disabled){background:var(--tx-accent-hover);border-color:var(--tx-accent-hover);transform:translateY(-1px);}
.stApp .stButton > button:active:not(:disabled){transform:scale(0.98);}
.stApp .stButton > button:focus-visible{outline:2px solid var(--tx-text);outline-offset:2px;}
.stApp .stButton > button:disabled{background:var(--tx-surface-3);border-color:var(--tx-surface-3);color:var(--tx-muted);cursor:not-allowed;}

/* ---------- Motion ---------- */
@keyframes tx-rise{from{opacity:0;transform:translateY(32px);}to{opacity:1;transform:none;}}
@keyframes tx-fade{from{opacity:0;}to{opacity:1;}}
@keyframes tx-grow{from{transform:scaleX(0);}to{transform:scaleX(1);}}
@keyframes tx-pulse{from{opacity:0.4;}to{opacity:1;}}
.tx-animate{animation:tx-rise 700ms var(--tx-ease) both;}
.tx-animate .tx-seg{animation:tx-grow 800ms var(--tx-ease) both;}

@media (prefers-reduced-motion: reduce){
  .tx-animate{animation:tx-fade 300ms linear both;}
  .tx-animate .tx-seg{animation:none;}
  .tx-skel{animation:none;opacity:0.7;}
  .tx-details .tx-caret,.stApp .stButton > button{transition:none;}
}

/* ---------- Mobile (390px and up) ---------- */
@media (max-width:640px){
  .stApp .heading-1{font-size:36px;line-height:40px;}
  .stApp .tx-result .heading-3{font-size:30px;line-height:36px;}
  .tx-empty{padding:var(--tx-s-400) var(--tx-s-200);}
  .tx-verdict .tx-badge{margin-left:0;}
}
@media (max-width:480px){
  .tx-facts{grid-template-columns:1fr;}
}
""".replace("__TEXT_STYLES__", _TEXT_STYLE_CSS)


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def md(markup: str) -> None:
    """Render raw HTML. Lines are stripped so Markdown never reads them as code."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def icon(name: str) -> str:
    return f'<span class="ph ph-{name}" aria-hidden="true"></span>'


def section_title(icon_name: str, title: str) -> None:
    md(
        f"""
        <div class="tx-section">
          {icon(icon_name)}
          <div class="heading-5" role="heading" aria-level="2">{esc(title)}</div>
        </div>
        """
    )


def fmt_value(value) -> str:
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved RandomForest model and feature column list."""
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_cols"]


@st.cache_data(show_spinner=False)
def load_player_lookup():
    """
    Build a dict: player_name -> {"rank": float, "pts": float, "date": Timestamp}
    using the most recent row in atp_tennis.csv where the player appears.

    Both Player_1 and Player_2 columns are considered. All appearances are
    stacked into one frame, sorted by Date, and the last row per player wins.
    Also returns the newest match date in the file, used to flag stale ranks.
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])

    # Replace sentinel -1 with NaN (matches Module 1 logic)
    numeric_cols = ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]
    df[numeric_cols] = df[numeric_cols].replace(-1, np.nan)

    p1 = df[["Date", "Player_1", "Rank_1", "Pts_1"]].rename(
        columns={"Player_1": "player", "Rank_1": "rank", "Pts_1": "pts"}
    )
    p2 = df[["Date", "Player_2", "Rank_2", "Pts_2"]].rename(
        columns={"Player_2": "player", "Rank_2": "rank", "Pts_2": "pts"}
    )

    combined = (
        pd.concat([p1, p2], ignore_index=True)
        .dropna(subset=["player", "rank"])
        .sort_values("Date", kind="stable")
        .drop_duplicates(subset="player", keep="last")
    )

    lookup = {
        row["player"]: {"rank": row["rank"], "pts": row["pts"], "date": row["Date"]}
        for row in combined.to_dict("records")
    }
    return lookup, df["Date"].max()


def build_feature_row(
    rank1: float,
    pts1: float,
    rank2: float,
    pts2: float,
    surface: str,
    court: str,
    series: str,
    round_name: str,
    best_of: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Assemble a single-row feature DataFrame aligned to the model's feature_cols.

    Encodes match context using the same scheme as Module 2:
      - rank_diff = rank1 - rank2
      - pts_diff  = pts1  - pts2
      - Round_ord = ROUND_ORDINAL[round_name]
      - Best of   = best_of (3 or 5)
      - OHE columns for Surface, Court, Series (baseline categories = 0)
    """
    row = {col: 0 for col in feature_cols}
    row["rank_diff"] = rank1 - rank2
    row["pts_diff"]  = pts1 - pts2
    row["Round_ord"] = ROUND_ORDINAL[round_name]
    row["Best of"]   = best_of

    # OHE: Surface (baseline = Carpet), Court (baseline = Indoor), Series (baseline = ATP250)
    for column in (f"Surface_{surface}", f"Court_{court}", f"Series_{series}"):
        if column in row:
            row[column] = 1

    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------------------------
# Markup builders
# ---------------------------------------------------------------------------
HERO = """
<div class="tx-hero">
  <div class="tx-brand">
    <span class="ph ph-tennis-ball" aria-hidden="true"></span>
    <span class="title-3">TennisX</span>
  </div>
  <div class="heading-1 tx-grad" role="heading" aria-level="1">ATP win probability predictor</div>
  <p class="body-3 tx-sub">Select two players and the match context, then press Predict to see each player’s modelled chance of winning.</p>
  <div class="tx-proof">
    <span class="tx-chip caption-4"><span class="ph ph-tree-structure" aria-hidden="true"></span>Random Forest classifier</span>
    <span class="tx-chip caption-4"><span class="ph ph-calendar-blank" aria-hidden="true"></span>Trained on matches from 2000 to 2021</span>
    <span class="tx-chip caption-4"><span class="ph ph-target" aria-hidden="true"></span>60.95% test accuracy</span>
  </div>
</div>
"""

BOOT_SKELETON = """
<div class="tx-skel-grid" role="status" aria-busy="true" aria-label="Loading players and model">
  <div class="tx-skel" style="height:40px"></div>
  <div class="tx-skel" style="height:40px"></div>
  <div class="tx-skel" style="height:128px"></div>
  <div class="tx-skel" style="height:128px"></div>
</div>
"""

RESULT_SKELETON = """
<div class="tx-card tx-result" role="status" aria-busy="true" aria-label="Calculating prediction">
  <div class="tx-tiles">
    <div class="tx-tile">
      <div class="tx-skel" style="height:24px;width:60%"></div>
      <div class="tx-skel" style="height:40px;width:50%;margin-top:8px"></div>
      <div class="tx-skel" style="height:16px;width:70%;margin-top:8px"></div>
    </div>
    <div class="tx-tile">
      <div class="tx-skel" style="height:24px;width:60%"></div>
      <div class="tx-skel" style="height:40px;width:50%;margin-top:8px"></div>
      <div class="tx-skel" style="height:16px;width:70%;margin-top:8px"></div>
    </div>
  </div>
  <div class="tx-barblock"><div class="tx-skel" style="height:8px;border-radius:9999px"></div></div>
  <div class="tx-verdict"><div class="tx-skel" style="height:24px;width:55%"></div></div>
</div>
"""

EMPTY_STATE = """
<div class="tx-card tx-empty">
  <div class="tx-icon">__ICON__</div>
  <div class="tx-copy">
    <div class="title-2">__TITLE__</div>
    <p class="body-6 tx-muted tx-pretty" style="margin-top:4px">__BODY__</p>
  </div>
</div>
"""


def empty_state(icon_name: str, title: str, body: str) -> str:
    return (
        EMPTY_STATE.replace("__ICON__", icon(icon_name))
        .replace("__TITLE__", esc(title))
        .replace("__BODY__", esc(body))
    )


def rank_card(name: str, info: dict, latest_date: pd.Timestamp) -> str:
    pts_missing = pd.isna(info["pts"])
    pts_text = (
        "Points unavailable, counted as 0"
        if pts_missing
        else f"{int(info['pts']):,} points"
    )
    last_date = info["date"]
    stale = (latest_date - last_date) > pd.Timedelta(days=STALE_AFTER_DAYS)
    if stale:
        badge = (
            f'<span class="tx-badge is-warn caption-4">{icon("warning")}'
            f'Last match {last_date.strftime("%b %Y")}, may be outdated</span>'
        )
    else:
        badge = (
            f'<span class="tx-badge caption-4">{icon("calendar-check")}'
            f'Last match {last_date.strftime("%b %Y")}</span>'
        )
    return f"""
    <div class="tx-card tx-stat" title="ATP ranking from the player’s most recent match in the dataset.">
      <div class="caption-4 tx-muted">{esc(name)} · last known rank</div>
      <div class="heading-3 tx-num">#{int(info["rank"])}</div>
      <div class="tx-statfoot">
        <span class="body-6 tx-muted tx-num">{esc(pts_text)}</span>
        {badge}
      </div>
    </div>
    """


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.75:
        label, icon_name, tone = "High confidence", "seal-check", "is-good"
    elif confidence >= 0.60:
        label, icon_name, tone = "Moderate confidence", "gauge", "is-warn"
    else:
        label, icon_name, tone = "Low confidence, near even", "scales", ""
    return (
        f'<span class="tx-badge {tone} caption-3">{icon(icon_name)}'
        f"{label} · {confidence * 100:.1f}%</span>"
    )


def result_markup(player1: str, player2: str, p1: float, p2: float, animate: bool) -> str:
    p1_leads = p1 >= p2
    winner = player1 if p1_leads else player2
    confidence = max(p1, p2)

    def tile(name: str, prob: float, is_winner: bool) -> str:
        delta = (prob - 0.5) * 100
        direction = "above" if delta >= 0 else "below"
        trophy = icon("trophy") if is_winner else ""
        return f"""
        <div class="tx-tile{' is-winner' if is_winner else ''}">
          <div class="tx-name title-3">{trophy}<span>{esc(name)}</span></div>
          <div class="heading-3 tx-num" aria-label="{prob * 100:.1f} percent chance to win">{prob * 100:.1f}%</div>
          <div class="caption-4 tx-muted tx-num">{abs(delta):.1f} pp {direction} an even matchup</div>
        </div>
        """

    win_color = "var(--tx-accent)"
    other_color = "var(--tx-muted)"
    left_color = win_color if p1_leads else other_color
    right_color = other_color if p1_leads else win_color

    return f"""
    <div class="tx-card tx-result{' tx-animate' if animate else ''}" role="region" aria-label="Prediction result">
      <div class="tx-tiles">
        {tile(player1, p1, p1_leads)}
        {tile(player2, p2, not p1_leads)}
      </div>
      <div class="tx-barblock">
        <div class="tx-bar" role="img" aria-label="{esc(player1)} {p1 * 100:.1f} percent, {esc(player2)} {p2 * 100:.1f} percent">
          <div class="tx-seg tx-seg-l" style="flex:{p1 * 100:.3f};background:{left_color}"></div>
          <div class="tx-seg tx-seg-r" style="flex:{p2 * 100:.3f};background:{right_color}"></div>
        </div>
        <div class="tx-barlabels caption-4 tx-muted">
          <span>{esc(player1)}</span>
          <span style="text-align:right">{esc(player2)}</span>
        </div>
      </div>
      <div class="tx-verdict">
        {icon("trophy")}
        <span class="body-1">Predicted winner: {esc(winner)}</span>
        {confidence_badge(confidence)}
      </div>
    </div>
    """


def feature_details_markup(features: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<tr><td class="tx-code">{esc(name)}</td><td class="tx-num">{esc(value)}</td></tr>'
        for name, value in features
    )
    return f"""
    <details class="tx-details">
      <summary class="body-4">
        {icon("sliders-horizontal")}Show feature inputs sent to the model
        <span class="ph ph-caret-down tx-caret" aria-hidden="true"></span>
      </summary>
      <div class="tx-details-body">
        <table class="tx-table">
          <thead><tr><th>Feature</th><th>Value</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>
    """


HOW_IT_WORKS = """
<p class="body-3 tx-muted tx-pretty">
  This app uses a <strong style="color:var(--tx-text);font-weight:600">Random Forest classifier</strong>
  trained on ATP match data from 2000 to 2021. Two later periods were held back for evaluation:
  2022 to 2023 as the test set and 2024 onward as a challenge set. The model predicts win
  probability from information available before the match only. It uses no match scores and no betting odds.
</p>
<div class="title-3" style="margin-top:8px">Key features the model uses</div>
<div class="tx-table-wrap">
  <table class="tx-table">
    <thead><tr><th>Feature</th><th>Description</th></tr></thead>
    <tbody>
      <tr><td class="tx-code">rank_diff</td><td>Player 1 ATP ranking minus Player 2 ATP ranking</td></tr>
      <tr><td class="tx-code">pts_diff</td><td>Player 1 ranking points minus Player 2 ranking points</td></tr>
      <tr><td class="tx-code">Round_ord</td><td>Match round encoded from 1 (1st Round) to 7 (Final)</td></tr>
      <tr><td class="tx-code">Best of</td><td>3 or 5 sets</td></tr>
      <tr><td class="tx-code">Surface, Court, Series</td><td>Match context, one hot encoded</td></tr>
    </tbody>
  </table>
</div>
<div class="tx-facts">
  <div class="tx-card tx-fact">
    <div class="heading-3 tx-num">60.95%</div>
    <div class="body-6 tx-muted">Accuracy on the 2022 to 2023 test set</div>
  </div>
  <div class="tx-card tx-fact">
    <div class="heading-3 tx-num">62.0%</div>
    <div class="body-6 tx-muted">Accuracy on the 50 most recent matches from 2024 onward. A small sample, so treat it as indicative.</div>
  </div>
</div>
"""

DISCLAIMER = """
<div class="tx-disclaimer" role="note">
  <span class="ph ph-warning" aria-hidden="true"></span>
  <p class="body-6 tx-muted tx-pretty">
    <strong style="color:var(--tx-text);font-weight:600">Disclaimer.</strong>
    This model is trained on historical ATP match records only. Betting odds were excluded from all
    features per project constraints. Player rankings shown are from each player’s most recent
    recorded match in the dataset and may be outdated for retired or inactive players. This tool is
    intended for educational purposes in a workshop setting. Do not use it for gambling or financial decisions.
  </p>
</div>
"""

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TennisX | ATP win probability predictor",
    page_icon="🎾",
    layout="centered",
)

md(f"<style>{CSS}</style>")
md(HERO)

# Load resources behind a skeleton so the page never shows a blank gap
boot = st.empty()
with boot.container():
    md(BOOT_SKELETON)

try:
    model, feature_cols = load_model()
    player_lookup, latest_date = load_player_lookup()
except FileNotFoundError as err:
    boot.empty()
    missing = os.path.relpath(err.filename or "", BASE_DIR) if err.filename else "a required file"
    md(
        f"""
        <div class="tx-alert" role="alert">
          {icon("warning-circle")}
          <div>
            <div class="body-4">Required file not found</div>
            <div class="body-6 tx-muted">Could not open {esc(missing)}. Run the earlier modules to create it, then reload this page.</div>
          </div>
        </div>
        """
    )
    st.stop()

boot.empty()
all_players = sorted(player_lookup.keys())

# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
section_title("users", "Players")

col_p1, col_p2 = st.columns(2)

with col_p1:
    default_p1 = all_players.index("Djokovic N.") if "Djokovic N." in all_players else 0
    player1 = st.selectbox("Player 1", all_players, index=default_p1)

with col_p2:
    default_p2 = all_players.index("Alcaraz C.") if "Alcaraz C." in all_players else min(1, len(all_players) - 1)
    player2 = st.selectbox("Player 2", all_players, index=default_p2)

same_player = player1 == player2
if same_player:
    md(
        f"""
        <div class="tx-alert" role="alert">
          {icon("warning-circle")}
          <div>
            <div class="body-4">Choose two different players</div>
            <div class="body-6 tx-muted">Player 1 and Player 2 are the same. Pick another player to turn on predictions.</div>
          </div>
        </div>
        """
    )

info1 = player_lookup[player1]
info2 = player_lookup[player2]

col_r1, col_r2 = st.columns(2)
with col_r1:
    md(rank_card(player1, info1, latest_date))
with col_r2:
    md(rank_card(player2, info2, latest_date))

md(
    '<p class="caption-4 tx-muted tx-note">Rank and points come from each player’s most recent match in the dataset.</p>'
)

# ---------------------------------------------------------------------------
# Match context inputs
# ---------------------------------------------------------------------------
section_title("sliders-horizontal", "Match context")

ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

with ctx_col1:
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"], index=0)
    court   = st.selectbox("Court", ["Outdoor", "Indoor"], index=0)

with ctx_col2:
    series_options = [
        "International", "International Gold", "ATP500",
        "Masters", "Masters 1000", "Masters Cup",
        "Grand Slam", "ATP250",
    ]
    series = st.selectbox("Series or tournament tier", series_options, index=0)

with ctx_col3:
    round_name = st.selectbox("Round", list(ROUND_ORDINAL.keys()), index=0)
    best_of    = st.selectbox("Best of", [3, 5], index=0)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
st.divider()

predict_clicked = st.button(
    "Predict win probability",
    type="primary",
    disabled=same_player,
)

section_title("chart-line-up", "Prediction")
result_slot = st.empty()

signature = (player1, player2, surface, court, series, round_name, best_of)
fresh = False

if predict_clicked and not same_player:
    with result_slot.container():
        md(RESULT_SKELETON)

    feature_row = build_feature_row(
        rank1=info1["rank"],
        pts1=info1["pts"] if pd.notna(info1["pts"]) else 0.0,
        rank2=info2["rank"],
        pts2=info2["pts"] if pd.notna(info2["pts"]) else 0.0,
        surface=surface,
        court=court,
        series=series,
        round_name=round_name,
        best_of=best_of,
        feature_cols=feature_cols,
    )

    proba = model.predict_proba(feature_row)[0]
    classes = list(model.classes_)
    st.session_state["tx_last"] = {
        "signature": signature,
        "p1": float(proba[classes.index(1)]),   # class 1 = Player_1 wins
        "p2": float(proba[classes.index(0)]),   # class 0 = Player_2 wins
        "features": [(col, fmt_value(feature_row.iloc[0][col])) for col in feature_cols],
    }
    fresh = True

last = st.session_state.get("tx_last")

with result_slot.container():
    if last and last["signature"] == signature:
        md(result_markup(player1, player2, last["p1"], last["p2"], animate=fresh))
        md(feature_details_markup(last["features"]))
    elif last:
        md(
            empty_state(
                "arrows-clockwise",
                "Inputs changed since the last prediction",
                "Press Predict win probability to update the result for the current players and match context.",
            )
        )
    else:
        md(
            empty_state(
                "tennis-ball",
                "No prediction yet",
                "Choose two players and the match context, then press Predict win probability. "
                "Both win probabilities appear here.",
            )
        )

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
st.divider()
section_title("tree-structure", "How does this model work?")
md(HOW_IT_WORKS)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.divider()
md(DISCLAIMER)