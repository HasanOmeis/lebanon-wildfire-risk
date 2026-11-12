"""Lebanon Fire Radar: Streamlit map of tomorrow's wildfire risk, read from the FastAPI service."""  # file description

import sys  # Python's module search path
from pathlib import Path  # file paths that work on Windows and Linux

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains app/)
if str(ROOT) not in sys.path:  # streamlit run only adds app/ itself to the path
    sys.path.insert(0, str(ROOT))  # add the root so "from app.api_client import ..." works

import copy  # deep copies, so we never change the cached API answer
import datetime as dt  # dates for the day picker and the freshness check

import pandas as pd  # tables for the top-risk list and the CSV (installed with Streamlit)
import pydeck as pdk  # map drawing (ships with Streamlit)
import streamlit as st  # the web app framework

from app.api_client import API_URL, ApiError, get_explain, get_health, get_risk  # all API calls go through the client (Step 2)

st.set_page_config(  # page settings; must be the first Streamlit command
    page_title="Lebanon Fire Radar",  # browser tab title
    page_icon="🔥",  # browser tab icon
    layout="wide",  # use the full screen width, the map needs space
)  # end of page settings

COMPACT_CSS = """
<style>
[data-testid="stHeader"] {height: 2.2rem; background: transparent;}  /* Streamlit's top bar (menu): shorter and see-through */
[data-testid="stMainBlockContainer"] {padding: 2.4rem 1.5rem 0.5rem 1.5rem;}  /* top padding = bar height, so the title starts below it */
[data-testid="stSidebar"][aria-expanded="true"] {min-width: 230px; max-width: 230px;}  /* narrower sidebar, only while it's open */
[data-testid="stSidebarContent"] {padding-top: 0.5rem;}  /* less empty space at the top of the sidebar */
h1 {font-size: 1.4rem !important; line-height: 1.4 !important; padding: 0 !important; margin: 0 !important;}  /* small title; line height keeps letters whole */
h2 {font-size: 1.05rem !important; padding: 0 0 0.3rem 0 !important;}  /* sidebar heading */
h3 {font-size: 1rem !important; line-height: 1.3 !important; padding: 0 !important;}  /* panel heading */
[data-testid="stCaptionContainer"] {font-size: 0.75rem;}  /* smaller grey captions */
[data-testid="stVerticalBlock"] {gap: 0.35rem;}  /* tighter space between stacked elements */
[data-testid="stHorizontalBlock"] {gap: 0.4rem;}  /* tighter space between columns */
[data-testid="stMetric"] {padding: 0.15rem 0.5rem; border: 1px solid rgba(128,128,128,0.2); border-radius: 0.4rem;}  /* each card: tight padding, thin border */
[data-testid="stMetricLabel"] p {font-size: 0.72rem;}  /* small card labels */
[data-testid="stMetricValue"] {font-size: 1.1rem;}  /* small card numbers */
[data-testid="stMetricDelta"] {font-size: 0.68rem;}  /* small grey notes under the numbers */
[data-testid="stMarkdownContainer"] p {margin-bottom: 0.15rem; font-size: 0.85rem;}  /* smaller text, less space under paragraphs */
.legend {display: flex; flex-wrap: wrap; gap: 0.9rem; font-size: 0.75rem; color: #6b5d50; margin: 0.3rem 0 0.7rem 0;}  /* legend row; bottom margin keeps it off the Top 10 bar */
.legend span.sw {display: inline-block; width: 0.9rem; height: 0.9rem; border: 1px solid #aaa; border-radius: 2px; vertical-align: -2px; margin-right: 0.3rem;}  /* the color square */
</style>
"""  # page-wide styles; data-testid names are the stable hooks Streamlit gives its elements
CACHE_SECONDS = 300  # keep API answers for 5 minutes: fast clicks, and fresh data still appears soon after a pipeline run
SERVED_DAYS = 400  # days the API serves; must match DAYS_KEPT in src/export_serving.py
MAP_HEIGHT = 480  # map height in pixels, so cards + map + panel fit on one screen
TOP_N = 10  # how many cells the top-risk table lists
REPO_URL = "https://github.com/HasanOmeis/lebanon-wildfire-risk"  # the project's GitHub repo, linked from "How it works"
LEVELS = ["low", "moderate", "high", "extreme"]  # risk levels from lowest to highest (same names as the API)
LEVEL_COLORS = {  # fill color per level as [red, green, blue, opacity 0-255]
    "low": [254, 240, 217, 60],  # pale cream, nearly see-through: most cells, so they shouldn't hide the base map
    "moderate": [253, 204, 138, 170],  # light orange
    "high": [252, 141, 89, 210],  # orange
    "extreme": [215, 48, 31, 235],  # deep red, almost solid
}  # end of the colors
LEVEL_DOTS = {"low": "⚪", "moderate": "🟡", "high": "🟠", "extreme": "🔴"}  # emoji dot per level, close to the map colors
LEBANON_VIEW = pdk.ViewState(  # where the map starts
    latitude=33.9,  # middle of Lebanon, north-south
    longitude=35.85,  # middle of Lebanon, east-west
    zoom=7.9,  # Lebanon fits the shorter map from north to south
    pitch=0,  # flat, looking straight down
)  # end of the view
HOW_IT_WORKS = f"""
1. **Data**: NASA FIRMS fire detections, Open-Meteo weather, MODIS vegetation (NDVI) and terrain, on a grid of 488 cells (~5 km).
2. **Features**: 52 per cell per day: dryness, heat, wind, rain history, vegetation, slope, and how often the area burned before.
3. **Model**: LightGBM, tuned with Optuna and checked on years and regions it never saw. Each day it scores how likely a fire is in every cell **tomorrow**.
4. **Daily pipeline**: every day it pulls fresh data, rebuilds the features, runs quality checks, scores all cells, uploads the model and serving files to the Hugging Face Hub, and reloads the live API.
5. **This page**: reads the FastAPI service (Docker on Render); clicking a cell shows its top SHAP reasons. Both run on free plans that sleep when idle, so the first visit can take about a minute.

[Code on GitHub]({REPO_URL})
"""  # sidebar text; Markdown, with the repo link filled in


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Waking up the server... free hosting sleeps when nobody visits, so the first load can take up to a minute.")  # first call of every visit; errors are NOT cached, so a failed call is retried next time (Day 17)
def load_health() -> dict:  # model info from /health; also wakes a sleeping API, so every later call is fast
    return get_health()  # one API call, with the long wake-up timeout (see api_client.py)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Loading the risk map...")  # cached per day; shows a spinner while the API builds the map
def load_risk(day: dt.date | None) -> dict:  # the map data for one features day (None = newest served day)
    return get_risk(day)  # one API call


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Asking the model why...")  # cached per (cell, day)
def load_explain(cell_id: int, day: dt.date) -> dict:  # one cell's risk with its top 3 SHAP reasons
    return get_explain(cell_id, day, k=3)  # one API call


def retry_button(key: str) -> None:  # a button that forgets cached answers and reloads the page
    if st.button("🔄 Retry", key=key):  # True only on the run right after the click
        st.cache_data.clear()  # drop every cached API answer, so the next calls really ask the API
        st.rerun()  # start the script again from the top


def freshness_banner(latest: dict) -> None:  # warns when the newest forecast is older than today
    today = dt.datetime.now(dt.timezone.utc).date()  # today in UTC (the pipeline and FIRMS work in UTC)
    newest = dt.date.fromisoformat(latest["forecast_for"])  # the newest day the model has a forecast for
    if newest >= today:  # today's (or tomorrow's) forecast is served: all good
        return  # no banner
    behind = (today - newest).days  # how many days the data is behind
    st.warning(  # yellow banner at the top of the page
        f"⏳ The newest forecast is for {newest} ({behind} day{'s' if behind > 1 else ''} old). "  # what's served and how old
        "Today's update hasn't run yet or failed, so treat this map as out of date."  # what it means for the visitor
    )  # end of the banner


def style_cells(geojson: dict) -> dict:  # add a fill color and a readable risk text to every cell
    styled = copy.deepcopy(geojson)  # work on a copy: the original belongs to Streamlit's cache
    for feature in styled["features"]:  # one polygon per cell
        props = feature["properties"]  # cell_id, lat, lon, land_frac, risk, level
        props["fill"] = LEVEL_COLORS[props["level"]]  # color for this cell's level
        props["risk_pct"] = f"{props['risk'] * 100:.2f}%"  # 0.0123 -> "1.23%", for the tooltip
        props["level_name"] = props["level"].capitalize()  # "high" -> "High", for the tooltip
    return styled  # the styled copy


def risk_map(geojson: dict) -> pdk.Deck:  # build the map: base map + one colored polygon per cell
    layer = pdk.Layer(  # one map layer
        "GeoJsonLayer",  # draws GeoJSON shapes
        data=style_cells(geojson),  # the styled cells
        id="cells",  # layer name; clicked cells come back under this name
        get_fill_color="properties.fill",  # each cell's color comes from its "fill" property
        get_line_color=[120, 120, 120, 90],  # thin grey outline so neighbouring cells are visible
        line_width_min_pixels=0.5,  # outline at least half a pixel wide at any zoom
        pickable=True,  # cells react to hover (tooltip) and clicks
        auto_highlight=True,  # brighten the cell under the mouse
    )  # end of the layer
    tooltip = {  # what shows when hovering a cell; {name} is replaced by that cell's property
        "html": "<b>Cell {cell_id}</b><br/>Risk: {risk_pct}<br/>Level: {level_name}<br/><i>Click for reasons</i>",  # tooltip text
        "style": {"backgroundColor": "#222", "color": "white", "fontSize": "12px"},  # dark box, white text
    }  # end of the tooltip
    return pdk.Deck(  # the full map
        layers=[layer],  # only our cells layer
        initial_view_state=LEBANON_VIEW,  # start over Lebanon
        map_style="light",  # light base map, so the risk colors stand out
        tooltip=tooltip,  # hover box
    )  # end of the map


def legend(cutoffs: dict) -> None:  # one row under the map: exact map colors + each level's probability range
    lower = {"low": 0.0, **cutoffs}  # minimum probability per level; low starts at 0 (cutoffs has moderate/high/extreme)
    items = []  # one HTML snippet per level
    for level in LEVELS:  # low, moderate, high, extreme
        r, g, b, a = LEVEL_COLORS[level]  # the exact fill color used on the map
        rule = "below " + f"{cutoffs['moderate'] * 100:.2f}%" if level == "low" else f"≥ {lower[level] * 100:.2f}%"  # "below 0.32%" or "≥ 1.03%"
        items.append(  # add this level's item
            f"<span><span class='sw' style='background: rgba({r},{g},{b},{a / 255:.2f})'></span>"  # the color square (opacity as 0-1)
            f"<b>{level.capitalize()}</b> {rule}</span>"  # level name + its probability rule
        )  # end of one item
    items.append("<span>· chance of a fire in the cell the next day</span>")  # what the % means
    st.markdown(f"<div class='legend'>{''.join(items)}</div>", unsafe_allow_html=True)  # draw the row


def pick_day(latest: dt.date) -> dt.date | None:  # sidebar control; returns the features day to show (None = latest)
    newest_forecast = latest + dt.timedelta(days=1)  # newest day with a forecast (features day + 1)
    oldest_forecast = newest_forecast - dt.timedelta(days=SERVED_DAYS - 1)  # oldest forecast still in the serving window
    mode = st.radio(  # two choices side by side
        "Which day?",  # label above the choices
        ["Latest", "Pick a day"],  # the options
        horizontal=True,  # in one row
        help="The map shows the risk FOR the chosen day, from data up to the day before.",  # ⓘ tooltip
    )  # end of the radio
    if mode == "Latest":  # default view
        return None  # load_risk(None) asks the API for /risk/latest
    forecast_day = st.date_input(  # a calendar picker
        "Forecast day",  # label
        value=newest_forecast,  # starts on the newest forecast
        min_value=oldest_forecast,  # can't go before the serving window
        max_value=newest_forecast,  # can't go into the future
        format="YYYY-MM-DD",  # same date format as everywhere else in the project
    )  # end of the picker
    return forecast_day - dt.timedelta(days=1)  # the API wants the features day (t), one day before the forecast


def summary_cards(risk: dict) -> int:  # one row of cards; returns the highest-risk cell's ID (the panel's default)
    counts = risk["level_counts"]  # {"low": 443, "moderate": 42, ...}
    cells = [f["properties"] for f in risk["geojson"]["features"]]  # every cell's properties
    top = max(cells, key=lambda c: c["risk"])  # the cell with the highest risk
    cols = st.columns(6, gap="small")  # six equal columns side by side, small gaps
    cols[0].metric("Forecast for", risk["forecast_for"], f"data up to {risk['date']}", delta_color="off", delta_arrow="off")  # the day the risk is about; grey note, no arrow
    for col, level in zip(cols[1:5], reversed(LEVELS)):  # extreme, high, moderate, low: worst first
        col.metric(f"{LEVEL_DOTS[level]} {level.capitalize()}", counts.get(level, 0))  # dot + level name, number of cells
    cols[5].metric("Highest risk", f"{top['risk'] * 100:.2f}%", f"cell {top['cell_id']}", delta_color="off", delta_arrow="off")  # top cell's risk, with its ID in grey
    return int(top["cell_id"])  # used when nothing is clicked yet


def clicked_cell(event) -> int | None:  # the cell ID from a map click, or None if nothing is selected
    objects = event.selection.get("objects", {}).get("cells", [])  # clicked objects, grouped by layer id
    if not objects:  # no click yet (or the click was cleared)
        return None  # nothing selected
    obj = objects[0]  # single selection: only one object
    props = obj.get("properties", obj)  # GeoJSON features keep their data under "properties"
    return int(props["cell_id"])  # the clicked cell


def format_value(value: float | None) -> str:  # a feature value, short and readable
    if value is None:  # the model saw this feature as missing
        return "missing"  # say so plainly
    if value != 0 and abs(value) < 1:  # small fractions like 0.0042 would show as 0.00 with 2 decimals
        return f"{value:.3g}"  # 3 significant digits instead: 0.0042 stays 0.0042
    return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"  # 2 decimals for normal numbers, none for big ones


def explain_panel(cell_id: int, day: dt.date, from_click: bool) -> None:  # side panel: risk + top 3 SHAP reasons for one cell
    try:  # /explain can fail like any API call
        info = load_explain(cell_id, day)  # risk, level and reasons
    except ApiError as err:  # API down, or cell/day not found
        st.warning(f"Couldn't explain cell {cell_id}: {err}")  # show the problem in the panel only
        retry_button("retry_explain")  # let the visitor try again without reloading the whole page by hand
        return  # leave the rest of the page working
    level = info["level"]  # this cell's level
    st.markdown(f"### Cell {cell_id}")  # panel heading
    st.caption("Your selection" if from_click else "Highest risk on this day. Click any cell to explain it.")  # why this cell is shown (true for past days too)
    st.metric(f"{LEVEL_DOTS[level]} {level.capitalize()} risk", f"{info['risk'] * 100:.2f}%")  # big number: the fire probability
    st.markdown("**Top reasons**")  # sub-heading
    for reason in info["reasons"] or []:  # top 3 SHAP reasons, strongest first
        arrow = "🔺" if reason["impact"] > 0 else "🔻"  # up = raises risk, down = lowers it
        st.markdown(  # one line per reason
            f"{arrow} **{reason['label']}**  \n"  # plain-English feature name (two spaces + \n = line break)
            f"<span style='color:grey;font-size:0.75rem'>value {format_value(reason['value'])} · {reason['direction']}</span>",  # value and direction, small and grey
            unsafe_allow_html=True,  # allow the grey <span>
        )  # end of one reason
    st.caption("Reasons come from SHAP: how much each feature pushed this cell's risk up or down.")  # what the reasons mean


def cells_frame(risk: dict) -> pd.DataFrame:  # every cell of one day as a table, riskiest first
    rows = [f["properties"] for f in risk["geojson"]["features"]]  # one dict per cell: cell_id, lat, lon, land_frac, risk, level
    df = pd.DataFrame(rows)[["cell_id", "lat", "lon", "risk", "level"]]  # keep the useful columns, in this order
    df = df.sort_values("risk", ascending=False).reset_index(drop=True)  # riskiest first; index 0..487
    df.insert(0, "rank", df.index + 1)  # rank column: 1 = riskiest
    df["forecast_for"] = risk["forecast_for"]  # the day the risk is about (same for every row)
    df["features_date"] = risk["date"]  # the day the features describe (same for every row)
    return df  # the full table


def top_table(risk: dict) -> None:  # collapsed section under the map: top cells + CSV download of all cells
    df = cells_frame(risk)  # all 488 cells, ranked
    with st.expander(f"📋 Top {TOP_N} riskiest cells · download all {len(df)}", expanded=False):  # closed by default, so the page stays short
        top = df.head(TOP_N).copy()  # the first TOP_N rows
        top["risk"] = top["risk"] * 100  # 0.0263 -> 2.63, shown with a % sign below
        top["level"] = top["level"].map(lambda lvl: f"{LEVEL_DOTS[lvl]} {lvl.capitalize()}")  # "high" -> "🟠 High"
        st.dataframe(  # an interactive table (sortable columns)
            top[["rank", "cell_id", "risk", "level", "lat", "lon"]],  # columns to show
            hide_index=True,  # the rank column already numbers the rows
            width="stretch",  # fill the section's width (replaces the deprecated use_container_width=True)
            column_config={  # how each column looks
                "rank": st.column_config.NumberColumn("#", width="small"),  # short header
                "cell_id": st.column_config.NumberColumn("Cell", width="small"),  # cell ID
                "risk": st.column_config.NumberColumn("Risk", format="%.2f%%"),  # 2.63 -> "2.63%"
                "level": st.column_config.TextColumn("Level"),  # dot + level name
                "lat": st.column_config.NumberColumn("Lat", format="%.3f"),  # cell center latitude
                "lon": st.column_config.NumberColumn("Lon", format="%.3f"),  # cell center longitude
            },  # end of the column settings
        )  # end of the table
        st.download_button(  # a button that downloads a file
            label=f"⬇️ Download all {len(df)} cells (CSV)",  # button text
            data=df.to_csv(index=False).encode("utf-8"),  # the full table as CSV bytes
            file_name=f"lebanon_fire_risk_{risk['forecast_for']}.csv",  # e.g. lebanon_fire_risk_2026-11-11.csv
            mime="text/csv",  # tells the browser it's a CSV file
        )  # end of the button


def main() -> None:  # builds the whole page, top to bottom, on every rerun
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)  # apply the compact styles before anything is drawn
    st.title("🔥 Lebanon Fire Radar")  # page heading
    st.caption("Next-day wildfire risk for every ~5 km cell in Lebanon, from a LightGBM model updated daily.")  # one-line explanation

    try:  # both calls can fail if the API is down
        health = load_health()  # model info
        latest = load_risk(None)  # newest day's map data (also tells us the newest served day)
    except ApiError as err:  # any failure from the client
        st.error(f"Couldn't load data: {err}")  # show it on the page instead of a Python traceback
        retry_button("retry_main")  # one click to try again once the API is back
        st.stop()  # end this run here; nothing below can work without data
    latest_day = dt.date.fromisoformat(latest["date"])  # newest features day, as a real date
    freshness_banner(latest)  # yellow warning if today's update is missing

    with st.sidebar:  # everything in this block goes into the left sidebar
        st.header("Settings")  # sidebar heading
        day = pick_day(latest_day)  # the features day chosen by the visitor (None = latest)
        st.divider()  # thin line between the controls and the info
        with st.expander("ℹ️ How it works"):  # closed by default; opens to explain the system
            st.markdown(HOW_IT_WORKS)  # the 5-step explanation + GitHub link
        st.caption(f"Model `{health['model_file']}`, trained on data up to {health['trained_to']}")  # which model is serving
        st.caption(f"API: `{API_URL}`")  # shows which API the page is reading, useful after the Day 17 deploy

    risk = latest  # by default, show the latest map we already have
    if day is not None and day != latest_day:  # an older day was picked
        try:  # that day might not be served
            risk = load_risk(day)  # that day's map data
        except ApiError as err:  # e.g. 404 if the serving window moved since the page loaded
            st.warning(f"Couldn't load that day, showing the latest instead. ({err})")  # explain, then keep going
            risk = latest  # fall back to the latest map
    shown_day = dt.date.fromisoformat(risk["date"])  # the features day actually on screen (after any fallback)

    top_cell = summary_cards(risk)  # the cards row, above the map; also gives the highest-risk cell
    map_col, panel_col = st.columns([3, 1], gap="small")  # map takes 3/4 of the width, the panel 1/4
    with map_col:  # left side
        event = st.pydeck_chart(  # draw the map and listen for clicks
            risk_map(risk["geojson"]),  # the map
            height=MAP_HEIGHT,  # shorter map, so everything fits on one screen
            on_select="rerun",  # a click reruns the script with the selection
            selection_mode="single-object",  # one cell at a time
            key="risk_map",  # fixed name, so the selection survives reruns
        )  # end of the map
        legend(health["risk_cutoffs"])  # color key with the live cutoffs, right under the map
    picked = clicked_cell(event)  # the clicked cell, if any
    with panel_col:  # right side
        explain_panel(picked if picked is not None else top_cell, shown_day, from_click=picked is not None)  # explain the click, or the top cell
    top_table(risk)  # collapsed top-risk table + CSV download, under the map


main()  # Streamlit runs this file top to bottom; this call builds the page