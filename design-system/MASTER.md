# Design System: Vishnupuri Barrage River Forecasting Dashboard
**Authority:** UI UX Pro Max Skill  
**Domain:** Hydrological Research & Time-Series Forecasting  
**Product Type:** Scientific Environmental Data Dashboard  
**Aesthetic Profile:** Quiet, Precise, Data-Dense, High-Legibility Light Surface  

---

## 1. Design Philosophy & Guardrails
- **Quiet, Precise, and Data-First:** Visual embellishment must never compete with hydrological observations or model verification.
- **Strict Anti-AI-Slop:** No neon glows, no synthetic glassmorphism, no animated pulsing gradients, no emojis as UI icons.
- **High Contrast & Accessible:** Text contrast > 7:1 against background surfaces (WCAG AAA/AA compliant).
- **Physical Provenance:** Distinguish observed data from model forecasts and distinguish real measurements from missing/unobserved intervals.

---

## 2. Color Palette & Semantic Tokens

### Core Neutral Surfaces
- `--bg-canvas`: `#f8fafc` (Slate 50 — calm neutral foundation)
- `--bg-surface`: `#ffffff` (Pure white card and chart container backgrounds)
- `--bg-subtle`: `#f1f5f9` (Slate 100 — secondary toolbars, table headers)
- `--border-default`: `#e2e8f0` (Slate 200 — crisp 1px borders)
- `--border-active`: `#cbd5e1` (Slate 300 — active control borders)
- `--border-focus`: `#2563eb` (Blue 600 — accessible focus outline)

### Typography Tokens
- `--text-primary`: `#0f172a` (Slate 900 — high-contrast headings and primary readings)
- `--text-secondary`: `#334155` (Slate 700 — labels and body text)
- `--text-muted`: `#64748b` (Slate 500 — secondary metadata and units)
- `--text-caption`: `#94a3b8` (Slate 400 — timestamps and subtle guides)

### Hydrological & Model Semantic Colors
- `--stage-observed`: `#1e40af` (Deep Blue 800 — solid 2.5px line for historical stage observations)
- `--stage-future-ref`: `#94a3b8` (Slate 400 — dashed 1px reference line for ground truth after forecast origin)
- `--forecast-pers`: `#059669` (Emerald 600 — persistence forecast vector and markers)
- `--forecast-ridge`: `#7c3aed` (Violet 600 — water-level-only Ridge forecast vector and markers)
- `--rain-satellite`: `#0284c7` (Sky 600 — INSAT-3DS half-hourly/hourly catchment precipitation)
- `--rain-imd`: `#d97706` (Amber 600 — IMD 0.25° 3-day antecedent catchment precipitation)
- `--ref-danger`: `#dc2626` (Red 600 — 354.80 m MSL barrage danger level)
- `--ref-warning`: `#ea580c` (Orange 600 — 354.00 m MSL warning level)
- `--timeline-cursor`: `#0f172a` (Slate 900 — synchronized vertical cursor line spanning hydrograph and hyetograph)

---

## 3. Typography Hierarchy
- **Body Font:** `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Data & Numeric Font:** `'JetBrains Mono', 'Fira Code', 'Consolas', monospace` (for measurements, timestamps, metrics)
- **Scale:**
  - Header Title: 16px (1rem), font-weight 700, uppercase letter-spacing 0.05em
  - Subtitle / Location: 12px (0.75rem), font-weight 400, color `--text-muted`
  - Section Headings: 13px (0.8125rem), font-weight 600, uppercase letter-spacing 0.04em
  - Large Telemetry Numbers: 20px–22px, font-family mono, font-weight 700
  - Table / Label text: 12px (0.75rem), font-weight 500
  - Small annotations / units: 11px (0.6875rem), font-weight 400

---

## 4. Layout & Spacing Rules
- **Header:** Compact 48px fixed height bar.
- **Control Bar:** Unified horizontal toolbar housing Event Selector, Horizon Toggles, Playback Controls, Timeline Scrubber, and Origin Readout.
- **KPI Telemetry Bar:** Compact single-row metrics strip with clear borders, avoiding oversized cards.
- **Main Viewport Grid:** 2-column layout (68% synchronized dual-axis charts, 32% tabbed inspection sidebar).
- **Responsive Breakpoint:** Stacks vertically below 1024px.

---

## 5. Chart Guidelines & Playback Architecture
1. **Vertical Timeline Cursor:** Drawn synchronously across both stage and rainfall charts at the exact origin $t$.
2. **Forecast Origin Vector:** At time $t$, dynamic line segments originate from $(t, H_t)$ and project forward to $(t+h, \hat{H}_{t+h})$ for the active lead time ($+1h, +2h, +3h$).
3. **No Target Leakage Illusion:** Observations prior to $t$ are rendered in full opacity solid blue. Subsequent observations are rendered in muted reference dash, clearly identifying what was known at forecast issue time.
4. **Distinguishable Markers:**
   - Observations: Circle (`circle`)
   - Persistence Forecast: Square (`rect`)
   - Ridge Forecast: Triangle (`triangle`)
   - Ground Truth Target: Diamond (`crossRot` / `diamond`)
5. **Zero-Lag Scrubbing:** Chart updates use `chart.update('none')` during drag and animation loops to guarantee 60fps responsiveness.
