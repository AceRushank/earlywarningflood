"""
historical_flood_analysis.py
-----------------------------
Descriptive analysis of IMD historical rainfall during Mumbai flood events (1969-2023).

Inputs:
  data/historical/mumbai_flood_events_with_rainfall.csv
  data/historical/mumbai_daily_rainfall_imd.csv

Outputs (outputs/historical_analysis/):
  rainfall_event_distribution.png
  antecedent_rainfall_distribution.png
  major_flood_event_timelines.png
  rainfall_spatial_variation.png
  historical_flood_rainfall_summary.csv
  historical_analysis_summary.txt

IMPORTANT DATA NOTES:
- All rainfall is IMD 0.25 deg gridded daily rainfall (NOT station/point rainfall).
- Four grid cells extracted: lat 19.00/19.25 x lon 72.75/73.00.
- rainfall_event_day_mm = arithmetic mean of the four cells on the event start date.
- This is a descriptive analysis only; no causal claims, no ML, no thresholds.
- 1-day, 3-day, and 7-day windows represent different accumulation periods
  and are NOT directly comparable quantities.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "historical_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

EVENTS_FILE = os.path.join(DATA_DIR, "mumbai_flood_events_with_rainfall.csv")
DAILY_FILE  = os.path.join(DATA_DIR, "mumbai_daily_rainfall_imd.csv")

C_BLUE   = "#2E86AB"
C_ORANGE = "#F6AE2D"
C_RED    = "#F26419"
C_GREEN  = "#33658A"
C_DARK   = "#1B2A41"

CELL_COLS = ["rainfall_19.00_72.75_mm","rainfall_19.00_73.00_mm",
             "rainfall_19.25_72.75_mm","rainfall_19.25_73.00_mm"]
CELL_LABELS = {
    "rainfall_19.00_72.75_mm": "19.00N/72.75E",
    "rainfall_19.00_73.00_mm": "19.00N/73.00E",
    "rainfall_19.25_72.75_mm": "19.25N/72.75E",
    "rainfall_19.25_73.00_mm": "19.25N/73.00E",
}
CELL_COLORS = {
    "rainfall_19.00_72.75_mm": "#4E79A7",
    "rainfall_19.00_73.00_mm": "#F28E2B",
    "rainfall_19.25_72.75_mm": "#E15759",
    "rainfall_19.25_73.00_mm": "#76B7B2",
}

events = pd.read_csv(EVENTS_FILE, parse_dates=["start_date","end_date"])
daily  = pd.read_csv(DAILY_FILE,  parse_dates=["date"])

if "rainfall_mumbai_mean_mm" not in events.columns:
    events["rainfall_mumbai_mean_mm"] = events[CELL_COLS].mean(axis=1)
events["year"] = events["start_date"].dt.year

print(f"Events loaded : {len(events)}")
print(f"Daily rows    : {len(daily)}")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved -> {path}")
    plt.close(fig)


def styled_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F7F9FC")
    ax.grid(axis="y", color="#C8D4E3", linewidth=0.6, linestyle="--")
    ax.grid(axis="x", color="#C8D4E3", linewidth=0.4, linestyle=":")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#8D9DB6")
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", color=C_DARK, pad=7)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=C_DARK)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=C_DARK)
    ax.tick_params(colors=C_DARK, labelsize=8)


# -- FIG 1: Event-day rainfall distribution -------------------------------------
print("\n[1/4] rainfall_event_distribution.png")
col  = "rainfall_event_day_mm"
vals = events[col].dropna()
med, mn = vals.median(), vals.mean()
p25, p75 = vals.quantile(0.25), vals.quantile(0.75)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
fig.suptitle(
    "IMD 0.25 deg Gridded Regional Mean Rainfall on Flood Event Start Dates\n"
    "(182 Mumbai flood events, 1969-2023)",
    fontsize=12, fontweight="bold", color=C_DARK, y=1.01
)

ax = axes[0]
counts, bin_edges, patches = ax.hist(vals, bins=30, color=C_BLUE, alpha=0.85,
                                     edgecolor="white", linewidth=0.6)
for patch, left in zip(patches, bin_edges[:-1]):
    if left >= 100:
        patch.set_facecolor(C_RED)
ax.axvline(med, color=C_ORANGE, linewidth=2.0, linestyle="--", label=f"Median {med:.1f} mm")
ax.axvline(mn,  color=C_RED,    linewidth=2.0, linestyle="-",  label=f"Mean {mn:.1f} mm")
ax.axvspan(p25, p75, alpha=0.12, color=C_GREEN, label=f"IQR [{p25:.0f}-{p75:.0f} mm]")
styled_ax(ax,
    title="Histogram of Event-Day Rainfall",
    xlabel="Rainfall on event start date (mm/day, IMD 0.25 deg regional mean)",
    ylabel="Number of events")
ax.legend(fontsize=8)
note = ("Bars in red >= 100 mm.\nEvent-day = mean of four 0.25 deg cells.\nNOT station or point rainfall.")
ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=7, color="#555",
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C8D4E3", alpha=0.9))

ax2 = axes[1]
x_sorted = np.sort(vals)
y_ecdf   = np.arange(1, len(x_sorted)+1) / len(x_sorted)
ax2.step(x_sorted, y_ecdf, color=C_BLUE, linewidth=1.8, where="post")
ax2.axvline(med, color=C_ORANGE, linewidth=1.8, linestyle="--", label=f"Median {med:.1f} mm")
ax2.axvline(mn,  color=C_RED,    linewidth=1.8, linestyle="-",  label=f"Mean {mn:.1f} mm")
for q in [0.25, 0.50, 0.75]:
    ax2.axhline(q, color="#CCCCCC", linewidth=0.8, linestyle=":")
styled_ax(ax2,
    title="Cumulative Distribution (ECDF)",
    xlabel="Rainfall on event start date (mm/day)",
    ylabel="Cumulative proportion of events")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "rainfall_event_distribution.png")

# -- FIG 2: Antecedent rainfall -------------------------------------------------
print("[2/4] antecedent_rainfall_distribution.png")
windows = {
    "1-day preceding\n(prior day only)":         ("rainfall_1day_before_mm",       C_BLUE),
    "3-day accumulation\n(3 days before event)": ("rainfall_3day_accumulation_mm",  C_ORANGE),
    "7-day accumulation\n(7 days before event)": ("rainfall_7day_accumulation_mm",  C_RED),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
fig.suptitle(
    "Antecedent IMD Rainfall Preceding Mumbai Flood Events (1969-2023)\n"
    "WARNING: These three panels represent DIFFERENT accumulation windows -- they are NOT directly comparable quantities.",
    fontsize=10, fontweight="bold", color=C_DARK, y=1.03
)
for ax, (label, (col, color)) in zip(axes, windows.items()):
    data = events[col].dropna()
    med_ = data.median(); mn_ = data.mean()
    ax.hist(data, bins=25, color=color, alpha=0.8, edgecolor="white", linewidth=0.6)
    ax.axvline(med_, color="#222", linewidth=1.8, linestyle="--", label=f"Median {med_:.1f} mm")
    ax.axvline(mn_,  color="#666", linewidth=1.8, linestyle="-",  label=f"Mean {mn_:.1f} mm")
    p90 = data.quantile(0.90)
    ax.axvline(p90, color="#999", linewidth=1.0, linestyle=":", alpha=0.7)
    top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
    ax.text(p90, top*0.92, f"p90={p90:.0f}mm", fontsize=7, color="#666",
            ha="left", rotation=90, va="top")
    styled_ax(ax, title=label.replace("\n"," "), xlabel="Accumulation (mm)", ylabel="Number of events")
    ax.legend(fontsize=8)

fig.text(0.5, -0.04,
    "Source: IMD Pune RF25 0.25 deg gridded daily rainfall. "
    "Values are arithmetic means of four grid cells (lat 19.00/19.25N, lon 72.75/73.00E). "
    "NOT station or point rainfall.",
    ha="center", fontsize=8, color="#777", style="italic")
fig.tight_layout()
save_fig(fig, "antecedent_rainfall_distribution.png")

# -- FIG 3: Major event timelines ----------------------------------------------
print("[3/4] major_flood_event_timelines.png")
notable = [
    ("MUM_FL_181", "2023-Jul-20 (Mumbai)"),
    ("MUM_FL_149", "2017-Aug-30 (Palghar/Thane/Mumbai)"),
    ("MUM_FL_037", "1990-Jun-16 (Mumbai/Nanded)"),
    ("MUM_FL_062", "1994-Jul-13 (Mumbai/Pune)"),
    ("MUM_FL_030", "1988-Jul-16 (Mumbai/Thane)"),
    ("MUM_FL_094", "2001-Jul-09 (Mumbai)"),
]

fig, axes = plt.subplots(3, 2, figsize=(15, 13), facecolor="white")
fig.suptitle(
    "Daily IMD Rainfall Surrounding Selected Major Mumbai Flood Events\n"
    "(30-day window: 14 days before to 15 days after event start)",
    fontsize=12, fontweight="bold", color=C_DARK
)
daily_idx = daily.set_index("date")

for idx, (ev_id, ev_label) in enumerate(notable):
    ax = axes.flatten()[idx]
    row = events[events["event_id"] == ev_id]
    if row.empty:
        ax.set_visible(False); continue
    row = row.iloc[0]
    ev_start = row["start_date"]; ev_end = row["end_date"]
    win_start = ev_start - pd.Timedelta(days=14)
    win_end   = ev_start + pd.Timedelta(days=15)
    mask    = (daily_idx.index >= win_start) & (daily_idx.index <= win_end)
    win_day = daily_idx[mask].copy()
    if win_day.empty:
        ax.set_visible(False); continue

    bar_colors = []
    for r in win_day["rainfall_mumbai_mean_mm"]:
        if r >= 115:   bar_colors.append("#C0392B")
        elif r >= 64:  bar_colors.append("#E67E22")
        elif r >= 7:   bar_colors.append(C_BLUE)
        else:          bar_colors.append("#BDC3C7")

    ax.bar(win_day.index, win_day["rainfall_mumbai_mean_mm"],
           color=bar_colors, width=0.9, edgecolor="none", alpha=0.9)
    ev_span_end = max(ev_end, ev_start) + pd.Timedelta(hours=12)
    ax.axvspan(ev_start - pd.Timedelta(hours=12), ev_span_end,
               alpha=0.18, color=C_RED)

    styled_ax(ax,
        title=f"{ev_label}  |  event-day mean: {row['rainfall_event_day_mm']:.1f} mm  |  7-day acc: {row['rainfall_7day_accumulation_mm']:.1f} mm",
        xlabel="Date", ylabel="Rainfall (mm/day)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    fat = row.get("human_fatality")
    if pd.notna(fat) and fat > 0:
        ax.text(0.02, 0.95, f"Fatalities: {int(fat)}", transform=ax.transAxes,
                fontsize=8, color="#C0392B", va="top", fontweight="bold")

    leg_handles = [
        Patch(fc="#C0392B", label=">=115 mm (Extremely Heavy)"),
        Patch(fc="#E67E22", label="64-115 mm (Very Heavy)"),
        Patch(fc=C_BLUE,    label="7-64 mm (Moderate/Heavy)"),
        Patch(fc="#BDC3C7", label="<7 mm (Light/None)"),
        Patch(fc=C_RED, alpha=0.3, label="Event period"),
    ]
    ax.legend(handles=leg_handles, fontsize=6.5, loc="upper right", ncol=1,
              framealpha=0.85, handlelength=1.2)

fig.text(0.5, -0.01,
    "Rainfall = IMD 0.25 deg gridded daily regional mean over four Mumbai-area cells. "
    "IMD intensity thresholds applied for bar colouring only (no causal inference).",
    ha="center", fontsize=8, color="#777", style="italic")
fig.tight_layout(h_pad=2.5, w_pad=2)
save_fig(fig, "major_flood_event_timelines.png")

# -- FIG 4: Spatial variation ---------------------------------------------------
print("[4/4] rainfall_spatial_variation.png")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="white")
fig.suptitle(
    "Spatial Variation in IMD 0.25 deg Gridded Rainfall Across Four Mumbai-Area Grid Cells\n"
    "Demonstrates why retaining individual cell values provides more information than the regional mean alone",
    fontsize=10, fontweight="bold", color=C_DARK, y=1.02
)

cell_labels_list = [CELL_LABELS[c] for c in CELL_COLS]
colors_list      = [CELL_COLORS[c] for c in CELL_COLS]

ax = axes[0]
box_data = [events[c].dropna().values for c in CELL_COLS]
bp = ax.boxplot(box_data, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color); patch.set_alpha(0.85)
for flier, color in zip(bp["fliers"], colors_list):
    flier.set_markerfacecolor(color)
ax.set_xticks(range(1,5))
ax.set_xticklabels(cell_labels_list, fontsize=7, rotation=12, ha="right")
styled_ax(ax, title="Box Plot: Event-Day Rainfall per Cell\n(flood event start dates)",
          ylabel="Rainfall (mm/day)")

ax2 = axes[1]
c1, c2 = "rainfall_19.00_72.75_mm", "rainfall_19.25_73.00_mm"
valid = events[[c1, c2]].dropna()
sc = ax2.scatter(valid[c1], valid[c2],
                 c=events.loc[valid.index,"year"], cmap="viridis",
                 s=28, alpha=0.75, edgecolors="none")
lims = [0, max(valid[c1].max(), valid[c2].max()) * 1.05]
ax2.plot(lims, lims, "--", color="#AAAAAA", linewidth=1.0, label="1:1 line")
ax2.set_xlim(lims); ax2.set_ylim(lims)
cbar = plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.03)
cbar.set_label("Year", fontsize=8); cbar.ax.tick_params(labelsize=7)
styled_ax(ax2,
    title="Cell Comparison: 19.00N/72.75E vs 19.25N/73.00E\n(event start-day rainfall)",
    xlabel="19.00N / 72.75E  (mm/day)",
    ylabel="19.25N / 73.00E  (mm/day)")
ax2.legend(fontsize=8)
corr = valid.corr().iloc[0,1]
ax2.text(0.03, 0.96, f"Pearson r = {corr:.2f}", transform=ax2.transAxes, fontsize=8, va="top")

ax3 = axes[2]
diff_vals = events[CELL_COLS].max(axis=1) - events[CELL_COLS].min(axis=1)
ax3.hist(diff_vals.dropna(), bins=28, color=C_GREEN, alpha=0.82, edgecolor="white", linewidth=0.6)
ax3.axvline(diff_vals.median(), color=C_ORANGE, linewidth=1.8, linestyle="--",
            label=f"Median {diff_vals.median():.1f} mm")
ax3.axvline(diff_vals.mean(), color=C_RED, linewidth=1.8, linestyle="-",
            label=f"Mean {diff_vals.mean():.1f} mm")
styled_ax(ax3,
    title="Intra-Event Spatial Spread\n(max - min across 4 cells on event start date)",
    xlabel="Rainfall range across four cells (mm/day)",
    ylabel="Number of events")
ax3.legend(fontsize=8)
frac_high = (diff_vals >= 50).sum() / len(diff_vals) * 100
ax3.text(0.97, 0.97, f"{frac_high:.0f}% of events\nhave >=50 mm spread\nacross cells",
         transform=ax3.transAxes, fontsize=8, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C8D4E3", alpha=0.9))

fig.text(0.5, -0.02,
    "Each cell is an IMD 0.25 deg grid point (approx 28 km spacing). "
    "Spatial spread shows different urban areas can experience substantially different totals.",
    ha="center", fontsize=8, color="#777", style="italic")
fig.tight_layout()
save_fig(fig, "rainfall_spatial_variation.png")

# -- CSV summary ----------------------------------------------------------------
print("\n[CSV] historical_flood_rainfall_summary.csv")
def pct(s, q): return round(float(s.quantile(q)), 2)

rows = []
window_notes = {
    "rainfall_event_day_mm":         "Single day (event start date)",
    "rainfall_1day_before_mm":       "Single day (day preceding event)",
    "rainfall_3day_accumulation_mm": "3-day sum (3 days before event); NOT comparable to 1d or 7d",
    "rainfall_7day_accumulation_mm": "7-day sum (7 days before event); NOT comparable to 1d or 3d",
}
for col, note in window_notes.items():
    s = events[col].dropna()
    rows.append({"metric":col,"n":int(s.count()),"mean":round(s.mean(),2),"median":round(s.median(),2),
                 "std":round(s.std(),2),"p10":pct(s,0.1),"p25":pct(s,0.25),"p75":pct(s,0.75),
                 "p90":pct(s,0.9),"min":round(s.min(),2),"max":round(s.max(),2),"window_note":note})
for c in CELL_COLS:
    s = events[c].dropna()
    rows.append({"metric":c,"n":int(s.count()),"mean":round(s.mean(),2),"median":round(s.median(),2),
                 "std":round(s.std(),2),"p10":pct(s,0.1),"p25":pct(s,0.25),"p75":pct(s,0.75),
                 "p90":pct(s,0.9),"min":round(s.min(),2),"max":round(s.max(),2),
                 "window_note":"Single day (event start date); individual IMD 0.25 deg cell"})

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"historical_flood_rainfall_summary.csv"), index=False)
print(f"  Saved -> {os.path.join(OUT_DIR,'historical_flood_rainfall_summary.csv')}")

# -- Text summary ---------------------------------------------------------------
print("\n[TXT] historical_analysis_summary.txt")
ev_day = events["rainfall_event_day_mm"].dropna()
ante1  = events["rainfall_1day_before_mm"].dropna()
ante3  = events["rainfall_3day_accumulation_mm"].dropna()
ante7  = events["rainfall_7day_accumulation_mm"].dropna()
diff_sp= events[CELL_COLS].max(axis=1) - events[CELL_COLS].min(axis=1)

top5 = events.nlargest(5,"rainfall_event_day_mm")[
    ["event_id","start_date","districts","rainfall_event_day_mm","rainfall_7day_accumulation_mm","human_fatality"]].copy()
top5["start_date"] = top5["start_date"].dt.strftime("%Y-%m-%d")

cell_means = {CELL_LABELS[c]: events[c].dropna().mean() for c in CELL_COLS}
max_cell = max(cell_means, key=cell_means.get)
min_cell = min(cell_means, key=cell_means.get)

lines = [
    "="*72,
    "HISTORICAL FLOOD-RAINFALL ANALYSIS SUMMARY",
    "Mumbai Flood Events x IMD 0.25 deg Gridded Daily Rainfall",
    "="*72,
    "",
    "DATA SOURCES",
    "  Events  : data/historical/mumbai_flood_events_with_rainfall.csv",
    "  Rainfall: data/historical/mumbai_daily_rainfall_imd.csv",
    "  Source  : IMD Pune RF25 0.25 deg gridded daily rainfall",
    "  Cells   : lat 19.00N, 19.25N  x  lon 72.75E, 73.00E",
    "",
    "IMPORTANT CAVEATS",
    "  * All rainfall is IMD 0.25 deg GRIDDED data, NOT station/point rainfall.",
    "  * rainfall_event_day_mm = arithmetic mean of four cells on event start date.",
    "  * 1-day, 3-day, 7-day antecedent values are DIFFERENT accumulation windows.",
    "    They are NOT directly comparable quantities.",
    "  * Descriptive analysis only. No causal claims. No ML. No thresholds.",
    "  * Co-occurrence of high rainfall with flood events does not imply causation.",
    "  * No interpolation, imputation, or synthetic values used.",
    "",
    "-"*72,
    "1. EVENTS ANALYSED",
    f"   Number of events          : {len(events)}",
    f"   Year range                : {events['year'].min()}-{events['year'].max()}",
    f"   Match status (COMPLETE)   : {(events['rainfall_match_status']=='COMPLETE').sum()}",
    "",
    "2. EVENT-DAY RAINFALL STATISTICS",
    "   (IMD 0.25 deg regional mean on flood event start date)",
    f"   Mean          : {ev_day.mean():.2f} mm",
    f"   Median        : {ev_day.median():.2f} mm",
    f"   Std deviation : {ev_day.std():.2f} mm",
    f"   p10           : {ev_day.quantile(0.10):.2f} mm",
    f"   p25 (Q1)      : {ev_day.quantile(0.25):.2f} mm",
    f"   p75 (Q3)      : {ev_day.quantile(0.75):.2f} mm",
    f"   p90           : {ev_day.quantile(0.90):.2f} mm",
    f"   Maximum       : {ev_day.max():.2f} mm  ({events.loc[ev_day.idxmax(),'event_id']})",
    "",
    "3. ANTECEDENT RAINFALL STATISTICS",
    "   NOTE: Different accumulation windows. CANNOT be compared directly.",
    "",
    "   1-day preceding (single prior day):",
    f"     Mean {ante1.mean():.2f} mm  Median {ante1.median():.2f} mm  p90 {ante1.quantile(0.9):.2f} mm",
    "",
    "   3-day accumulation (3-day sum before event start):",
    f"     Mean {ante3.mean():.2f} mm  Median {ante3.median():.2f} mm  p90 {ante3.quantile(0.9):.2f} mm",
    "",
    "   7-day accumulation (7-day sum before event start):",
    f"     Mean {ante7.mean():.2f} mm  Median {ante7.median():.2f} mm  p90 {ante7.quantile(0.9):.2f} mm",
    "",
    "4. HIGHEST OBSERVED EVENT-DAY RAINFALL (TOP 5)",
]
for _, r in top5.iterrows():
    fat_str = f"fatalities={int(r['human_fatality'])}" if pd.notna(r["human_fatality"]) else "fatalities=NR"
    lines.append(f"   {r['event_id']}  {r['start_date']}  event_day={r['rainfall_event_day_mm']:.1f}mm  7d_acc={r['rainfall_7day_accumulation_mm']:.1f}mm  {fat_str}")

lines += [
    "",
    "5. SPATIAL VARIATION ACROSS FOUR GRID CELLS",
    "   (Event start-date rainfall, mean across all 182 events)",
]
for label, mv in cell_means.items():
    lines.append(f"   {label:20s}: {mv:.2f} mm")
lines += [
    f"   Highest mean cell : {max_cell}",
    f"   Lowest mean cell  : {min_cell}",
    f"   Median intra-event spread (max-min across cells): {diff_sp.median():.1f} mm",
    f"   Events with >=50 mm spread: {(diff_sp>=50).sum()} ({(diff_sp>=50).mean()*100:.0f}%)",
    "",
    "   OBSERVATION: Substantial spatial spread is common, confirming that",
    "   retaining individual cell values is useful for sub-urban spatial analysis.",
    "",
    "6. LIMITATIONS",
    "   * IMD gridded rainfall is a spatial average over ~28 km cells.",
    "     It does not capture peak point rainfall at specific stations.",
    "   * Daily resolution does not capture sub-daily intensity patterns.",
    "   * Event start dates may not always correspond to peak rainfall days.",
    "   * No formal statistical testing performed (descriptive only).",
    "   * Correlation with flood events does not imply causation.",
    "",
    "="*72,
    "Generated by: scripts/historical_flood_analysis.py",
    "Output dir  : outputs/historical_analysis/",
    "="*72,
]

txt_path = os.path.join(OUT_DIR, "historical_analysis_summary.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  Saved -> {txt_path}")
print("\nAll outputs written to outputs/historical_analysis/")
