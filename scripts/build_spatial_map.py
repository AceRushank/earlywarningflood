"""
Build self-contained interactive Leaflet HTML visualization for Mumbai 500m Spatial Foundation.
Uses ONLY:
  data/spatial/mumbai_500m_grid.geojson
  data/spatial/mumbai_500m_grid_features.csv

Outputs:
  outputs/mumbai_spatial_features_map.html
"""
import os
import json
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "spatial", "mumbai_500m_grid.geojson")
CSV_PATH = os.path.join(BASE_DIR, "data", "spatial", "mumbai_500m_grid_features.csv")
OUT_HTML = os.path.join(BASE_DIR, "outputs", "mumbai_spatial_features_map.html")

def build_map():
    print("Loading GeoJSON and CSV...")
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        raw_geojson = json.load(f)
    
    csv_df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(raw_geojson['features'])} GeoJSON features and {len(csv_df)} CSV rows.")

    # Create compact feature collection
    # Round coords to 5 decimals (~1.1m precision), only store essential props
    features = []
    for feat in raw_geojson["features"]:
        props = feat["properties"]
        coords = [[[round(c[0], 5), round(c[1], 5)] for c in ring] for ring in feat["geometry"]["coordinates"]]
        features.append({
            "type": "Feature",
            "id": props["grid_id"],
            "properties": {
                "id": props["grid_id"],
                "e": round(float(props["elevation_mean_m"]), 1),
                "s": round(float(props["slope_mean_deg"]), 1),
                "w": round(float(props["water_pct_gsw"]) * 100, 1)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            }
        })
    
    compact_geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    geojson_str = json.dumps(compact_geojson, separators=(',', ':'))
    print(f"Compact GeoJSON size: {len(geojson_str) / (1024*1024):.2f} MB")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mumbai 500m Spatial Foundation Map</title>
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    #header {{
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 1000;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title-group {{
      display: flex;
      flex-direction: column;
    }}
    .title-group h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .title-group span {{
      font-size: 0.78rem;
      color: #94a3b8;
    }}
    .layer-selector {{
      display: flex;
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
    }}
    .layer-btn {{
      background: transparent;
      border: none;
      color: #94a3b8;
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 7px 16px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .layer-btn:hover {{
      color: #f8fafc;
      background: rgba(255, 255, 255, 0.05);
    }}
    .layer-btn.active {{
      background: #0284c7;
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4);
    }}
    #map-container {{
      flex: 1;
      position: relative;
      width: 100%;
      height: 100%;
    }}
    #map {{
      width: 100%;
      height: 100%;
      background: #0f172a;
    }}
    /* Floating Panels */
    .glass-panel {{
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      color: #f1f5f9;
      padding: 14px 16px;
      position: absolute;
      z-index: 1000;
    }}
    #info-panel {{
      top: 16px;
      left: 16px;
      width: 320px;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
    }}
    .panel-section-title {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #38bdf8;
      font-weight: 700;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .metric-card {{
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      padding: 8px;
    }}
    .metric-card .label {{
      font-size: 0.68rem;
      color: #94a3b8;
    }}
    .metric-card .val {{
      font-size: 0.95rem;
      font-weight: 700;
      color: #f8fafc;
      font-family: 'JetBrains Mono', monospace;
      margin-top: 2px;
    }}
    .source-box {{
      background: rgba(30, 41, 59, 0.5);
      border-left: 3px solid #38bdf8;
      border-radius: 0 6px 6px 0;
      padding: 8px 10px;
      margin-bottom: 12px;
      font-size: 0.76rem;
      line-height: 1.4;
    }}
    .source-box strong {{
      color: #e2e8f0;
    }}
    .source-box .src-detail {{
      color: #94a3b8;
      font-size: 0.72rem;
      margin-top: 2px;
    }}
    /* Inspection details */
    #hover-info {{
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 6px;
      padding: 10px;
      font-size: 0.78rem;
    }}
    #hover-info table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 4px;
    }}
    #hover-info td {{
      padding: 2px 4px;
    }}
    #hover-info td.val {{
      text-align: right;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      color: #38bdf8;
    }}
    /* Legend Panel */
    #legend-panel {{
      bottom: 24px;
      right: 16px;
      min-width: 220px;
    }}
    .legend-title {{
      font-size: 0.75rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 5px;
      font-size: 0.76rem;
      color: #cbd5e1;
    }}
    .legend-color {{
      width: 20px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      flex-shrink: 0;
    }}
    /* Opacity Control */
    .opacity-control {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 10px;
      font-size: 0.72rem;
      color: #94a3b8;
    }}
    .opacity-control input {{
      width: 100px;
      cursor: pointer;
    }}
    /* Custom Leaflet Controls */
    .leaflet-control-zoom a {{
      background: rgba(15, 23, 42, 0.85) !important;
      color: #f1f5f9 !important;
      border-color: rgba(255, 255, 255, 0.15) !important;
    }}
    .leaflet-control-zoom a:hover {{
      background: #1e293b !important;
    }}
    .leaflet-control-attribution {{
      background: rgba(15, 23, 42, 0.75) !important;
      color: #64748b !important;
      font-size: 0.65rem !important;
    }}
    .leaflet-control-attribution a {{
      color: #94a3b8 !important;
    }}
  </style>
</head>
<body>

  <header id="header">
    <div class="title-group">
      <h1>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon><line x1="8" y1="2" x2="8" y2="18"></line><line x1="16" y1="6" x2="16" y2="22"></line></svg>
        Mumbai 500m Spatial Foundation
      </h1>
      <span>Canonical 8,600-Cell Environmental Grid Framework</span>
    </div>

    <div class="layer-selector">
      <button class="layer-btn active" id="btn-elevation" onclick="switchLayer('elevation')">
        ⛰️ Elevation
      </button>
      <button class="layer-btn" id="btn-slope" onclick="switchLayer('slope')">
        📐 Slope
      </button>
      <button class="layer-btn" id="btn-water" onclick="switchLayer('water')">
        💧 Water %
      </button>
    </div>
  </header>

  <div id="map-container">
    <div id="map"></div>

    <!-- Info Panel -->
    <div id="info-panel" class="glass-panel">
      <div class="panel-section-title">
        <span>Spatial Framework</span>
        <span style="color: #10b981; font-weight: 600;">ACTIVE</span>
      </div>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="label">Total Cells</div>
          <div class="val">8,600</div>
        </div>
        <div class="metric-card">
          <div class="label">Resolution</div>
          <div class="val">500 m</div>
        </div>
        <div class="metric-card">
          <div class="label">Native CRS</div>
          <div class="val" style="font-size: 0.78rem;">EPSG:32643</div>
        </div>
        <div class="metric-card">
          <div class="label">Display CRS</div>
          <div class="val" style="font-size: 0.78rem;">WGS84 (4326)</div>
        </div>
      </div>

      <div class="panel-section-title">Active Layer Metadata</div>
      <div id="active-source-box" class="source-box">
        <!-- Populated via JS -->
      </div>

      <div class="panel-section-title">Cell Inspection (Hover / Click)</div>
      <div id="hover-info">
        <div style="color: #94a3b8; font-style: italic;">Hover over or click a grid cell to inspect values</div>
      </div>

      <div class="opacity-control">
        <span>Layer Opacity</span>
        <input type="range" id="opacity-slider" min="0.1" max="1" step="0.05" value="0.75" oninput="changeOpacity(this.value)">
      </div>
    </div>

    <!-- Legend Panel -->
    <div id="legend-panel" class="glass-panel">
      <div class="legend-title" id="legend-title">Elevation (m)</div>
      <div id="legend-items">
        <!-- Populated via JS -->
      </div>
    </div>
  </div>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  
  <script>
    // Embedded GeoJSON data (8,600 cells, exact data from GeoJSON & CSV)
    const GRID_GEOJSON = {geojson_str};

    // Layer metadata definitions
    const LAYER_META = {{
      elevation: {{
        name: "Elevation (Mean)",
        unit: "m",
        source: "Copernicus DEM GLO-30",
        source_res: "30 m spatial resolution",
        domain: "0.0 to 447.1 m (Mean: 19.8 m)",
        description: "Surface elevation derived from 30m Copernicus digital elevation model, aggregated as mean elevation per 500m cell.",
        legend: [
          {{ color: '#084594', label: '0 m (Sea level / Coast)' }},
          {{ color: '#2171b5', label: '0 – 10 m (Low Coastal Plain)' }},
          {{ color: '#4292c6', label: '10 – 25 m (Suburban Flats)' }},
          {{ color: '#6baed6', label: '25 – 50 m (Elevated Ridges)' }},
          {{ color: '#feb24c', label: '50 – 100 m (Foothills)' }},
          {{ color: '#f03b20', label: '100 – 200 m (SGNP Hills)' }},
          {{ color: '#bd0026', label: '> 200 m (Peaks / High Ridges)' }}
        ],
        getColor: function(v) {{
          if (v <= 0) return '#084594';
          if (v <= 10) return '#2171b5';
          if (v <= 25) return '#4292c6';
          if (v <= 50) return '#6baed6';
          if (v <= 100) return '#feb24c';
          if (v <= 200) return '#f03b20';
          return '#bd0026';
        }},
        getValue: f => f.properties.e
      }},
      slope: {{
        name: "Slope (Mean)",
        unit: "°",
        source: "Horn (1981) via Copernicus GLO-30",
        source_res: "30 m spatial resolution",
        domain: "0.0° to 30.2° (Mean: 2.92°)",
        description: "Terrain slope computed from 30m GLO-30 DEM using Horn's finite-difference method, aggregated to 500m cells.",
        legend: [
          {{ color: '#ffffb2', label: '0° – 1° (Flat terrain / Water)' }},
          {{ color: '#fed976', label: '1° – 2.5° (Gentle slope)' }},
          {{ color: '#feb24c', label: '2.5° – 5° (Moderate slope)' }},
          {{ color: '#fd8d3c', label: '5° – 10° (Strong slope)' }},
          {{ color: '#f03b20', label: '10° – 18° (Steep terrain)' }},
          {{ color: '#bd0026', label: '> 18° (Very steep slopes)' }}
        ],
        getColor: function(v) {{
          if (v <= 1.0) return '#ffffb2';
          if (v <= 2.5) return '#fed976';
          if (v <= 5.0) return '#feb24c';
          if (v <= 10.0) return '#fd8d3c';
          if (v <= 18.0) return '#f03b20';
          return '#bd0026';
        }},
        getValue: f => f.properties.s
      }},
      water: {{
        name: "Water Percentage",
        unit: "%",
        source: "JRC Global Surface Water (GSW) v1.4",
        source_res: "30 m spatial resolution (Occ ≥ 10%)",
        domain: "0.0% to 100.0% (Mean: 42.5%)",
        description: "Percentage of 30m sub-pixels inside the 500m cell classified as surface water (occurrence >= 10%). Captures creeks, bays, Arabian Sea, and reservoirs.",
        legend: [
          {{ color: '#f7fbff', label: '0% (Dry Land)' }},
          {{ color: '#deebf7', label: '0.1% – 10% (Low Water/Marsh)' }},
          {{ color: '#9ecae1', label: '10% – 30% (Moderate Water)' }},
          {{ color: '#4292c6', label: '30% – 60% (Creeks & Mudflats)' }},
          {{ color: '#2171b5', label: '60% – 90% (Estuaries & Inlets)' }},
          {{ color: '#084594', label: '90% – 100% (Open Water / Sea)' }}
        ],
        getColor: function(v) {{
          if (v <= 0.0) return '#f7fbff';
          if (v <= 10.0) return '#deebf7';
          if (v <= 30.0) return '#9ecae1';
          if (v <= 60.0) return '#4292c6';
          if (v <= 90.0) return '#2171b5';
          return '#084594';
        }},
        getValue: f => f.properties.w
      }}
    }};

    let currentLayer = 'elevation';
    let currentOpacity = 0.75;
    let geojsonLayer = null;

    // Initialize Map with Canvas renderer for smooth 8600-polygon rendering
    const map = L.map('map', {{
      preferCanvas: true,
      zoomControl: true,
      attributionControl: true
    }}).setView([19.0760, 72.8777], 11);

    // CartoDB Dark Matter base tile
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      subdomains: 'abcd',
      maxZoom: 19
    }}).addTo(map);

    function getStyle(feature) {{
      const meta = LAYER_META[currentLayer];
      const val = meta.getValue(feature);
      return {{
        fillColor: meta.getColor(val),
        weight: 0.4,
        opacity: 0.6,
        color: '#0f172a',
        fillOpacity: currentOpacity
      }};
    }}

    function onEachFeature(feature, layer) {{
      layer.on({{
        mouseover: function(e) {{
          const l = e.target;
          l.setStyle({{
            weight: 2,
            color: '#38bdf8',
            opacity: 1
          }});
          l.bringToFront();
          updateHoverInfo(feature.properties);
        }},
        mouseout: function(e) {{
          geojsonLayer.resetStyle(e.target);
        }},
        click: function(e) {{
          updateHoverInfo(feature.properties);
        }}
      }});
    }}

    function updateHoverInfo(p) {{
      const html = `
        <strong>Cell: ${{p.id}}</strong>
        <table>
          <tr><td>Elevation:</td><td class="val">${{p.e.toFixed(1)}} m</td></tr>
          <tr><td>Slope:</td><td class="val">${{p.s.toFixed(1)}}°</td></tr>
          <tr><td>Water Fraction:</td><td class="val">${{p.w.toFixed(1)}}%</td></tr>
        </table>
      `;
      document.getElementById('hover-info').innerHTML = html;
    }}

    function updateMetadataPanel() {{
      const meta = LAYER_META[currentLayer];
      const box = document.getElementById('active-source-box');
      box.innerHTML = `
        <div><strong>Layer:</strong> ${{meta.name}}</div>
        <div class="src-detail"><strong>Source:</strong> ${{meta.source}}</div>
        <div class="src-detail"><strong>Source Resolution:</strong> ${{meta.source_res}}</div>
        <div class="src-detail"><strong>Value Domain:</strong> ${{meta.domain}}</div>
        <div class="src-detail" style="margin-top: 4px; line-height: 1.3;">${{meta.description}}</div>
      `;

      // Update Legend
      document.getElementById('legend-title').innerText = `${{meta.name}} (${{meta.unit}})`;
      const legendContainer = document.getElementById('legend-items');
      legendContainer.innerHTML = '';
      meta.legend.forEach(item => {{
        const div = document.createElement('div');
        div.className = 'legend-item';
        div.innerHTML = `<span class="legend-color" style="background:${{item.color}}"></span><span>${{item.label}}</span>`;
        legendContainer.appendChild(div);
      }});
    }}

    function switchLayer(layerKey) {{
      if (currentLayer === layerKey) return;
      currentLayer = layerKey;

      // Update button styles
      ['elevation', 'slope', 'water'].forEach(k => {{
        const btn = document.getElementById(`btn-${{k}}`);
        if (k === layerKey) {{
          btn.classList.add('active');
        }} else {{
          btn.classList.remove('active');
        }}
      }});

      // Update layer styles
      geojsonLayer.eachLayer(layer => {{
        layer.setStyle(getStyle(layer.feature));
      }});

      updateMetadataPanel();
    }}

    function changeOpacity(val) {{
      currentOpacity = parseFloat(val);
      geojsonLayer.eachLayer(layer => {{
        layer.setStyle({{ fillOpacity: currentOpacity }});
      }});
    }}

    // Add GeoJSON Layer
    geojsonLayer = L.geoJSON(GRID_GEOJSON, {{
      style: getStyle,
      onEachFeature: onEachFeature
    }}).addTo(map);

    // Zoom to grid bounds
    map.fitBounds(geojsonLayer.getBounds(), {{ padding: [20, 20] }});

    // Initial panel setup
    updateMetadataPanel();
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    file_size_mb = os.path.getsize(OUT_HTML) / (1024 * 1024)
    print(f"Map successfully generated at: {OUT_HTML}")
    print(f"Output file size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    build_map()
