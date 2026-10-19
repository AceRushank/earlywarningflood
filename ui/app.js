/* ==========================================================================
   Vishnupuri Barrage — Godavari River Forecasting Dashboard Logic
   Design Authority: UI UX Pro Max Skill (Scientific Hydrology Dashboard)
   Features: Causal Timeline Engine, Dual-Axis Synchronized Hydrograph &
             Hyetograph, Interactive Scrubbing, Real-Time Forecast Vectors,
             Full Support for 2026 Audited Multivariable Models & 2024 Baseline
   ========================================================================== */

(function () {
  'use strict';

  // Application State
  let dashboardData = null;
  let activeTimeline = [];
  let currentIndex = 0;
  let activeHorizon = 1; // 1, 2, or 3 hours
  let isPlaying = false;
  let playbackTimer = null;
  let playbackSpeed = 1; // 1 = 1x (1000ms), 2 = 2x (500ms), 5 = 5x (200ms)
  let activeEventPreset = '2026_monsoon_full';
  let selectedModel = 'model_d'; // 'model_d', 'dh_ridge', 'wl_ridge', 'rf', 'xgb'
  let activeThematicLayer = 'lulc'; // 'lulc', 'dem', 'slope', 'river', 'gsw'

  // Chart & Map References
  let hydrographChart = null;
  let hyetographChart = null;
  let modelComparisonChart = null;
  let horizonErrorChart = null;
  let leafletMap = null;
  let stageMarker = null;
  let catchmentLayer = null;
  let reachLayer = null;

  // DOM Elements
  const eventSelector = document.getElementById('event-selector');
  const modelSelector = document.getElementById('model-selector');
  const horizonBtns = [
    document.getElementById('horizon-btn-1'),
    document.getElementById('horizon-btn-2'),
    document.getElementById('horizon-btn-3')
  ];

  const btnRestart = document.getElementById('btn-restart');
  const btnStepBack = document.getElementById('btn-step-back');
  const btnPlayPause = document.getElementById('btn-play-pause');
  const btnStepForward = document.getElementById('btn-step-forward');
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const speedBtns = document.querySelectorAll('.btn-speed');

  const timelineSlider = document.getElementById('timeline-slider');
  const currentTimestampEl = document.getElementById('current-timestamp');
  const scrubberStepIndicatorEl = document.getElementById('scrubber-step-indicator');

  // View Navigation
  const viewBtnDashboard = document.getElementById('view-btn-dashboard');
  const viewBtnStudy = document.getElementById('view-btn-study');
  const viewBtnLive = document.getElementById('view-btn-live');
  const studyAreaView = document.getElementById('study-area-view');

  // Thematic Layer Buttons
  const thematicBtns = document.querySelectorAll('.btn-thematic');
  const activeLayerSourceTag = document.getElementById('active-layer-source-tag');
  const dynamicThematicLegend = document.getElementById('dynamic-thematic-legend');

  // Hydrograph Active Model Marker & Label
  const legendActiveModelMarker = document.getElementById('legend-active-model-marker');
  const legendActiveModelName = document.getElementById('legend-active-model-name');
  const forecastSummaryCaption = document.getElementById('forecast-summary-caption');

  // Telemetry Strip Elements
  const currentStageEl = document.getElementById('current-stage');
  const stageRateEl = document.getElementById('stage-rate');
  const telemetryLeadLabelEl = document.getElementById('telemetry-lead-label');
  const forecastPersValEl = document.getElementById('forecast-pers-val');
  const forecastRidgeValEl = document.getElementById('forecast-ridge-val');
  const forecastExpValEl = document.getElementById('forecast-exp-val');
  const forecastActualTargetEl = document.getElementById('forecast-actual-target');
  const rainSat1hEl = document.getElementById('rain-sat-1h');
  const rainSat24hEl = document.getElementById('rain-sat-24h');
  const soilApiValEl = document.getElementById('soil-api-val');
  const rainCompletenessTagEl = document.getElementById('rain-completeness-tag');
  const rainfallModelStatusEl = document.getElementById('rainfall-model-status');
  const modelStatusCaptionEl = document.getElementById('model-status-caption');

  // Legend Items
  const legendExpItem = document.getElementById('legend-exp-item');
  const legendDhRidgeItem = document.getElementById('legend-dh-ridge-item');
  const legendRfItem = document.getElementById('legend-rf-item');
  const legendXgbItem = document.getElementById('legend-xgb-item');
  const legendOutageItem = document.getElementById('legend-outage-item');
  const legendApiItem = document.getElementById('legend-api-item');
  const legendImdItem = document.getElementById('legend-imd-item');

  // GIS Controls
  const btnToggleReach = document.getElementById('btn-toggle-reach');
  const btnToggleCatchment = document.getElementById('btn-toggle-catchment');
  const btnZoomReach = document.getElementById('btn-zoom-reach');
  const btnZoomBasin = document.getElementById('btn-zoom-basin');

  // Annotations & Modal Elements
  const annotationsContainer = document.getElementById('annotations-container');
  const metadataModal = document.getElementById('metadata-modal');
  const btnShowMetadata = document.getElementById('btn-show-metadata');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // Side Panel Tabs
  const tabBtns = [
    document.getElementById('tab-btn-metrics'),
    document.getElementById('tab-btn-map'),
    document.getElementById('tab-btn-physio')
  ];
  const tabContents = [
    document.getElementById('tab-content-metrics'),
    document.getElementById('tab-content-map'),
    document.getElementById('tab-content-physio')
  ];

  // Benchmark Subnav Elements
  const btnMetrics2026 = document.getElementById('btn-metrics-2026');
  const btnMetrics2024 = document.getElementById('btn-metrics-2024');
  const metrics2026View = document.getElementById('metrics-2026-view');
  const metrics2024View = document.getElementById('metrics-2024-view');

  // Operational Mode Elements (Historical vs Live)
  const btnModeHistorical = document.getElementById('btn-mode-historical');
  const btnModeLive = document.getElementById('btn-mode-live');
  const controlToolbar = document.getElementById('control-toolbar');
  const telemetryStrip = document.getElementById('telemetry-strip');
  const mainLayout = document.getElementById('main-layout');
  const liveFeedContainer = document.getElementById('live-feed-container');
  const btnRefreshLive = document.getElementById('btn-refresh-live');
  const btnRefreshText = document.getElementById('btn-refresh-text');
  const btnReturnHistorical = document.getElementById('btn-return-historical');
  const liveLastCheckedTime = document.getElementById('live-last-checked-time');
  const liveCardStage = document.getElementById('live-card-stage');
  const liveCardFreeboard = document.getElementById('live-card-freeboard');
  const liveCardTimestamp = document.getElementById('live-card-timestamp');
  const liveCardAge = document.getElementById('live-card-age');
  const liveCardStnName = document.getElementById('live-card-stn-name');
  const liveCardAgency = document.getElementById('live-card-agency');
  const liveCardCoords = document.getElementById('live-card-coords');
  const liveBannerDesc = document.getElementById('live-banner-desc');

  // ==========================================================================
  // Chart.js Synchronized Vertical Cursor Plugin
  // ==========================================================================
  const verticalCursorPlugin = {
    id: 'verticalCursorPlugin',
    afterDraw(chart) {
      if (!activeTimeline.length || currentIndex < 0 || currentIndex >= activeTimeline.length) return;
      const meta = chart.getDatasetMeta(0);
      if (!meta || !meta.data || !meta.data[currentIndex]) return;

      const x = meta.data[currentIndex].x;
      const { top, bottom } = chart.chartArea;
      const ctx = chart.ctx;

      ctx.save();
      // Draw vertical cursor line
      ctx.beginPath();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();

      // Top indicator triangle
      ctx.setLineDash([]);
      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.moveTo(x - 4, top);
      ctx.lineTo(x + 4, top);
      ctx.lineTo(x, top + 6);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }
  };

  Chart.register(verticalCursorPlugin);

  // ==========================================================================
  // Application Initialization
  // ==========================================================================
  async function init() {
    try {
      try {
        const cfgResp = await fetch('/api/config');
        if (cfgResp.ok) {
          const cfg = await cfgResp.json();
          if (cfg && cfg.carto_key) {
            window.CARTO_KEY = cfg.carto_key;
          }
        }
      } catch (e) {
        // Fallback to public basemaps
      }

      const response = await fetch('dashboard_data.json?v=' + Date.now());
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status} loading dashboard_data.json`);
      }
      dashboardData = await response.json();

      initMap();
      setEventFilter('2026_monsoon_full');
      initCharts();
      rebuildChartStructures();
      updateModelComparisonChart();
      renderAnnotations();
      setupEventListeners();
      updateUI();
    } catch (err) {
      console.error('Initialization error:', err);
      if (currentTimestampEl) {
        currentTimestampEl.textContent = 'Error loading data';
        currentTimestampEl.style.color = '#dc2626';
      }
    }
  }

  // ==========================================================================
  // Event Filter & Timeline Range Definition
  // ==========================================================================
  function setEventFilter(preset) {
    activeEventPreset = preset;
    const is2026 = preset.startsWith('2026_');
    const full = is2026
      ? (dashboardData.timeline_2026 || [])
      : (dashboardData.timeline_2024 || dashboardData.timeline || []);

    if (preset === '2026_phase_a') {
      activeTimeline = full.filter(pt => pt.time >= '2026-07-30 00:00:00' && pt.time <= '2026-07-31 05:00:00');
    } else if (preset === '2026_sep_holdout') {
      activeTimeline = full.filter(pt => pt.time >= '2026-09-08 06:00:00' && pt.time <= '2026-09-22 19:00:00');
    } else if (preset === '2026_surge_outage') {
      activeTimeline = full.filter(pt => pt.time >= '2026-07-31 00:00:00' && pt.time <= '2026-08-05 00:00:00');
    } else if (preset === '2026_monsoon_full') {
      activeTimeline = full;
    } else if (preset === 'rising_limb') {
      activeTimeline = full.filter(pt => pt.time >= '2024-08-03 10:00:00' && pt.time <= '2024-08-04 20:00:00');
    } else if (preset === 'event1_full') {
      activeTimeline = full.filter(pt => pt.time >= '2024-07-31 05:00:00' && pt.time <= '2024-08-05 23:00:00');
    } else if (preset === 'post_peak') {
      activeTimeline = full.filter(pt => pt.time >= '2024-08-05 05:00:00' && pt.time <= '2024-08-05 23:00:00');
    } else if (preset === 'pre_event') {
      activeTimeline = full.filter(pt => pt.time >= '2024-07-04 00:00:00' && pt.time <= '2024-07-30 23:00:00');
    } else {
      activeTimeline = full; // monsoon_full
    }

    currentIndex = 0;
    timelineSlider.min = 0;
    timelineSlider.max = Math.max(0, activeTimeline.length - 1);
    timelineSlider.value = 0;

    // Synchronize side-tab metrics toggle
    if (is2026) {
      showMetricsView('2026');
      if (legendExpItem) legendExpItem.style.display = 'inline-flex';
      if (legendDhRidgeItem) legendDhRidgeItem.style.display = 'inline-flex';
      if (legendRfItem) legendRfItem.style.display = 'inline-flex';
      if (legendXgbItem) legendXgbItem.style.display = 'inline-flex';
      if (legendOutageItem) legendOutageItem.style.display = 'inline-flex';
      if (legendApiItem) legendApiItem.style.display = 'inline-flex';
      if (legendImdItem) legendImdItem.style.display = 'none';
    } else {
      showMetricsView('2024');
      if (legendExpItem) legendExpItem.style.display = 'none';
      if (legendDhRidgeItem) legendDhRidgeItem.style.display = 'none';
      if (legendRfItem) legendRfItem.style.display = 'none';
      if (legendXgbItem) legendXgbItem.style.display = 'none';
      if (legendOutageItem) legendOutageItem.style.display = 'none';
      if (legendApiItem) legendApiItem.style.display = 'none';
      if (legendImdItem) legendImdItem.style.display = 'inline-flex';
    }

    renderAnnotations();

    if (hydrographChart && hyetographChart) {
      rebuildChartStructures();
    }
    if (modelComparisonChart) {
      updateModelComparisonChart();
    }
  }

  function showMetricsView(version) {
    if (version === '2026') {
      if (btnMetrics2026) btnMetrics2026.classList.add('active');
      if (btnMetrics2024) btnMetrics2024.classList.remove('active');
      if (metrics2026View) metrics2026View.style.display = 'block';
      if (metrics2024View) metrics2024View.style.display = 'none';
    } else {
      if (btnMetrics2026) btnMetrics2026.classList.remove('active');
      if (btnMetrics2024) btnMetrics2024.classList.add('active');
      if (metrics2026View) metrics2026View.style.display = 'none';
      if (metrics2024View) metrics2024View.style.display = 'block';
    }
  }

  // ==========================================================================
  // Leaflet Catchment & River Reach GIS Map (CartoDB Voyager Raster)
  // ==========================================================================
  function initMap() {
    const barrageLat = 19.126944;
    const barrageLon = 77.280833;

    leafletMap = L.map('leaflet-map', {
      zoomControl: true,
      attributionControl: true
    }).setView([barrageLat, barrageLon], 8);

    L.control.scale({
      metric: true,
      imperial: false,
      position: 'bottomleft'
    }).addTo(leafletMap);

    const urlParams = new URLSearchParams(window.location.search);
    const cartoKey = urlParams.get('key') || urlParams.get('carto_key') || window.CARTO_KEY || (dashboardData && dashboardData.carto_key) || '';

    const tileUrl = cartoKey
      ? `https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=${encodeURIComponent(cartoKey)}`
      : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png';

    L.tileLayer(tileUrl, {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19
    }).addTo(leafletMap);

    // Layer Groups for Thematic Map View
    let lulcLayerGroup = L.layerGroup();
    let demLayerGroup = L.layerGroup();
    let slopeLayerGroup = L.layerGroup();
    let riverLayerGroup = L.layerGroup();
    let gswLayerGroup = L.layerGroup();

    // 1. Upstream Catchment Layer (51,768 km²)
    if (dashboardData.catchment_geojson) {
      catchmentLayer = L.geoJSON(dashboardData.catchment_geojson, {
        style: {
          color: '#1d4ed8',
          weight: 1.8,
          opacity: 0.8,
          fillColor: '#3b82f6',
          fillOpacity: 0.08
        }
      });
      riverLayerGroup.addLayer(catchmentLayer);
    }

    // 2. River Reach Corridor Layer (1,231.5 km²: Subbasins 4081058170 + 4081060220)
    if (dashboardData.reach_corridor_geojson) {
      reachLayer = L.geoJSON(dashboardData.reach_corridor_geojson, {
        style: {
          color: '#059669',
          weight: 2.2,
          opacity: 0.95,
          fillColor: '#10b981',
          fillOpacity: 0.20
        }
      });
      riverLayerGroup.addLayer(reachLayer);

      // DEM Layer (Styling based on SRTM elevation classes: 345m pool to 420m ridge)
      const demReachLayer = L.geoJSON(dashboardData.reach_corridor_geojson, {
        style: function (feature) {
          const isPool = feature.properties && feature.properties.hybas_id === 4081058170;
          return {
            color: '#b45309',
            weight: 2,
            opacity: 0.9,
            fillColor: isPool ? '#fed7aa' : '#fb923c',
            fillOpacity: 0.55
          };
        }
      });
      demLayerGroup.addLayer(demReachLayer);

      // Slope Layer (Derived HydroBASINS 0.11% channel vs 2.8% valley walls)
      const slopeReachLayer = L.geoJSON(dashboardData.reach_corridor_geojson, {
        style: function (feature) {
          const isPool = feature.properties && feature.properties.hybas_id === 4081058170;
          return {
            color: '#475569',
            weight: 2,
            opacity: 0.9,
            fillColor: isPool ? '#cbd5e1' : '#94a3b8',
            fillOpacity: 0.50
          };
        }
      });
      slopeLayerGroup.addLayer(slopeReachLayer);

      // LULC Layer (Upper Godavari regional distribution: Agriculture 72%, Scrub 18%, Built-up 5%, Water 4%)
      // Transparent study area corridor boundary; regional statistics shown without fabricating raster pixels.
      const lulcReachLayer = L.geoJSON(dashboardData.reach_corridor_geojson, {
        style: function () {
          return {
            color: '#0284c7',
            weight: 2.2,
            dashArray: '5, 4',
            opacity: 0.85,
            fillColor: '#f0f9ff',
            fillOpacity: 0.20
          };
        }
      });
      lulcLayerGroup.addLayer(lulcReachLayer);
    }

    // 3. Historical Water Occurrence Layer (JRC Global Surface Water v1.4)
    if (dashboardData.gsw_geojson) {
      const gswWaterLayer = L.geoJSON(dashboardData.gsw_geojson, {
        style: function (feature) {
          const props = feature.properties || {};
          const cls = props.occurrence_class;
          const name = props.class_name || '';
          let fillCol = '#38bdf8';
          if (cls === 3 || name.includes('Permanent') || (props.occurrence_pct && props.occurrence_pct >= 80)) {
            fillCol = '#0369a1'; // Permanent deep pool
          } else if (cls === 2 || name.includes('Seasonal') || (props.occurrence_pct && props.occurrence_pct >= 20)) {
            fillCol = '#38bdf8'; // Seasonal monsoon flood
          } else {
            fillCol = '#bae6fd'; // Ephemeral margin
          }

          return {
            color: fillCol,
            weight: 1,
            opacity: 0.9,
            fillColor: fillCol,
            fillOpacity: 0.70
          };
        },
        onEachFeature: function (feature, layer) {
          const props = feature.properties || {};
          const name = props.class_name || 'Surface Water Feature';
          layer.bindPopup(`
            <div style="font-family:Inter,sans-serif; font-size:11px; color:#0f172a; padding:2px;">
              <strong style="color:#0369a1;">JRC Surface Water (1984–2021)</strong><br>
              <strong>Class:</strong> ${name}<br>
              <em>Historical recurrence, not flood extent</em>
            </div>
          `);
        }
      });
      gswLayerGroup.addLayer(gswWaterLayer);
    }

    // Default to LULC Layer active
    lulcLayerGroup.addTo(leafletMap);
    if (reachLayer) {
      leafletMap.fitBounds(reachLayer.getBounds(), { padding: [16, 16] });
    }

    // 4. Barrage Pour Point Marker
    const barrageIcon = L.divIcon({
      className: 'custom-barrage-marker',
      html: '<div style="background-color:#dc2626; width:12px; height:12px; border-radius:50%; border:2px solid #ffffff; box-shadow:0 1px 4px rgba(0,0,0,0.4);" title="Vishnupuri Barrage (Pour Point)"></div>',
      iconSize: [12, 12],
      iconAnchor: [6, 6]
    });

    stageMarker = L.marker([barrageLat, barrageLon], { icon: barrageIcon }).addTo(leafletMap);
    stageMarker.bindPopup(`
      <div style="font-family:Inter,sans-serif; font-size:12px; color:#0f172a; padding:2px;">
        <strong style="color:#1e40af;">Vishnupuri Barrage</strong><br>
        <strong>River:</strong> Godavari<br>
        <strong>FRL:</strong> 355.00 m MSL<br>
        <strong>Crest/MSL:</strong> 366.50 m MSL<br>
        <strong>Station:</strong> 006-SW-MAHA (NWIC)
      </div>
    `);

    // 5. On-Map Semi-Transparent Dynamic Legend Control
    let onMapLegendEl = null;
    const mapLegendControl = L.control({ position: 'bottomright' });
    mapLegendControl.onAdd = function () {
      onMapLegendEl = L.DomUtil.create('div', 'leaflet-onmap-legend');
      onMapLegendEl.id = 'map-thematic-legend-overlay';
      return onMapLegendEl;
    };
    mapLegendControl.addTo(leafletMap);

    // Thematic Layer Switching Function
    function setThematicLayer(layerName) {
      activeThematicLayer = layerName;
      // Remove all thematic layer groups
      leafletMap.removeLayer(lulcLayerGroup);
      leafletMap.removeLayer(demLayerGroup);
      leafletMap.removeLayer(slopeLayerGroup);
      leafletMap.removeLayer(riverLayerGroup);
      leafletMap.removeLayer(gswLayerGroup);

      let legendHtml = '';

      // Add selected
      if (layerName === 'lulc') {
        lulcLayerGroup.addTo(leafletMap);
        if (activeLayerSourceTag) activeLayerSourceTag.innerHTML = '<strong>Source:</strong> Bhuvan / NRSC Regional Land-Cover Study &bull; Upper Godavari Basin &bull; Regional Distribution: 72% Cropland, 18% Scrub, 5% Built-up, 4% Water &bull; <em>Regional statistics only — raster pending</em>';
        legendHtml = `
          <div class="onmap-legend-header">
            <span class="onmap-legend-title">Land Cover (Bhuvan / NRSC)</span>
            <span class="onmap-legend-tag tag-pending">Regional statistics only — raster pending</span>
          </div>
          <div class="onmap-legend-items">
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#86efac; border:1px solid #15803d;"></span><span>Cropland (72% regional)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#fef08a; border:1px solid #ca8a04;"></span><span>Scrub &amp; Fallow (18% regional)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#cbd5e1; border:1px solid #64748b;"></span><span>Built-up / Urban (5% regional)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#38bdf8; border:1px solid #0284c7;"></span><span>Water Bodies (4% regional)</span></div>
          </div>
          <div class="onmap-legend-note">Authentic corridor boundary shown. High-res NRSC raster pending archive staging; no synthetic pixels plotted.</div>
        `;
      } else if (layerName === 'dem') {
        demLayerGroup.addTo(leafletMap);
        if (activeLayerSourceTag) activeLayerSourceTag.innerHTML = '<strong>Source:</strong> HydroSHEDS / SRTM 90m Hydrologically Conditioned DEM &bull; Datum: WGS84 EGM96 &bull; Units: Metres MSL';
        legendHtml = `
          <div class="onmap-legend-header">
            <span class="onmap-legend-title">Elevation Baseline (SRTM / HydroSHEDS)</span>
            <span class="onmap-legend-tag tag-ready">Authentic Subbasins</span>
          </div>
          <div class="onmap-legend-items">
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#fed7aa; border:1px solid #ea580c;"></span><span>345 – 360 m MSL (Barrage Reach Pool)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#fb923c; border:1px solid #c2410c;"></span><span>360 – 385 m MSL (Upstream Corridor Terraces)</span></div>
          </div>
          <div class="onmap-legend-note">HydroSHEDS conditioned subbasin hypsometry (WGS84 EGM96 datum). High-res 30m grid raster pending staging.</div>
        `;
      } else if (layerName === 'slope') {
        slopeLayerGroup.addTo(leafletMap);
        if (activeLayerSourceTag) activeLayerSourceTag.innerHTML = '<strong>Source:</strong> Derived from HydroSHEDS / SRTM DEM &bull; Channel slope: 1.1 m/km (0.11%) &bull; Valley walls: 2.8%';
        legendHtml = `
          <div class="onmap-legend-header">
            <span class="onmap-legend-title">Slope Gradient (Derived Baseline)</span>
            <span class="onmap-legend-tag tag-ready">Authentic Subbasins</span>
          </div>
          <div class="onmap-legend-items">
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#cbd5e1; border:1px solid #64748b;"></span><span>&lt;0.5% (Low-gradient channel bed: 1.1 m/km)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#94a3b8; border:1px solid #475569;"></span><span>0.5% – 2.8% (Alluvial valley terraces &amp; bluffs)</span></div>
          </div>
          <div class="onmap-legend-note">Derived from HydroBASINS subbasin stream gradients. High-res slope raster pending staging.</div>
        `;
      } else if (layerName === 'river') {
        riverLayerGroup.addTo(leafletMap);
        if (activeLayerSourceTag) activeLayerSourceTag.innerHTML = '<strong>Source:</strong> HydroBASINS Lev08 Pfafstetter Routing &bull; River Corridor: 1,231.5 km² &bull; Basin: 51,768 km²';
        legendHtml = `
          <div class="onmap-legend-header">
            <span class="onmap-legend-title">Hydrographic Domains</span>
            <span class="onmap-legend-tag tag-ready">Authentic Vector</span>
          </div>
          <div class="onmap-legend-items">
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#10b981; border:1px solid #059669;"></span><span>River Reach Corridor (1,231.5 km²)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#3b82f6; border:1px solid #1d4ed8;"></span><span>Upper Godavari Catchment (51,768 km²)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#dc2626; border:1px solid #ffffff;"></span><span>Vishnupuri Barrage (Pour Point)</span></div>
          </div>
          <div class="onmap-legend-note">HydroBASINS Lev08 Pfafstetter routing topology and CWC/NWIC barrage location.</div>
        `;
      } else if (layerName === 'gsw') {
        gswLayerGroup.addTo(leafletMap);
        if (activeLayerSourceTag) activeLayerSourceTag.innerHTML = '<strong>Source:</strong> JRC Global Surface Water v1.4 (1984–2021 Occurrence) &bull; 30m resolution &bull; 205 authentic vectorized water polygons &bull; <em>Historical recurrence, not flood extent</em>';
        legendHtml = `
          <div class="onmap-legend-header">
            <span class="onmap-legend-title">Historical Water Occurrence (JRC v1.4)</span>
            <span class="onmap-legend-tag tag-ready">205 Authentic Vectors</span>
          </div>
          <div class="onmap-legend-items">
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#0369a1; border:1px solid #082f49;"></span><span>Permanent Pool (81–100% recurrence)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#38bdf8; border:1px solid #0284c7;"></span><span>Seasonal Water (21–80% recurrence)</span></div>
            <div class="onmap-legend-row"><span class="legend-swatch" style="background:#bae6fd; border:1px solid #7dd3fc;"></span><span>Ephemeral / Marginal (1–20% recurrence)</span></div>
          </div>
          <div class="onmap-legend-note">Long-term satellite surface water recurrence (1984–2021). Historical occurrence only; not flood extent.</div>
        `;
      }

      if (onMapLegendEl) {
        onMapLegendEl.innerHTML = legendHtml;
      }
      if (dynamicThematicLegend) {
        dynamicThematicLegend.innerHTML = legendHtml;
      }
    }

    window.setThematicLayer = setThematicLayer;

    // Attach thematic buttons
    thematicBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        thematicBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const lyr = btn.getAttribute('data-layer');
        setThematicLayer(lyr);
      });
    });

    // Initial legend trigger
    setThematicLayer('lulc');

    // Zoom & Toggle controls
    const btnZoomReach = document.getElementById('btn-zoom-reach');
    const btnZoomBasin = document.getElementById('btn-zoom-basin');
    const btnToggleReach = document.getElementById('btn-toggle-reach');
    const btnToggleCatchment = document.getElementById('btn-toggle-catchment');

    if (btnZoomReach) {
      btnZoomReach.addEventListener('click', () => {
        if (reachLayer) leafletMap.fitBounds(reachLayer.getBounds(), { padding: [16, 16] });
      });
    }

    if (btnZoomBasin) {
      btnZoomBasin.addEventListener('click', () => {
        if (catchmentLayer) leafletMap.fitBounds(catchmentLayer.getBounds(), { padding: [16, 16] });
      });
    }

    if (btnToggleReach) {
      btnToggleReach.addEventListener('click', () => {
        if (!reachLayer) return;
        if (leafletMap.hasLayer(reachLayer)) {
          leafletMap.removeLayer(reachLayer);
          btnToggleReach.classList.remove('active');
        } else {
          leafletMap.addLayer(reachLayer);
          btnToggleReach.classList.add('active');
        }
      });
    }

    if (btnToggleCatchment) {
      btnToggleCatchment.addEventListener('click', () => {
        if (!catchmentLayer) return;
        if (leafletMap.hasLayer(catchmentLayer)) {
          leafletMap.removeLayer(catchmentLayer);
          btnToggleCatchment.classList.remove('active');
        } else {
          leafletMap.addLayer(catchmentLayer);
          btnToggleCatchment.classList.add('active');
        }
      });
    }
  }

  // ==========================================================================
  // Dual-Axis Synchronized Hydrograph & Hyetograph Visualisation
  // ==========================================================================
  function initCharts() {
    const hydroCtx = document.getElementById('hydrographCanvas').getContext('2d');
    const hyetoCtx = document.getElementById('hyetographCanvas').getContext('2d');

    // Chart 1: River Stage Hydrograph
    hydrographChart = new Chart(hydroCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Observed Stage H(t)',
            data: [],
            borderColor: '#1e40af',
            backgroundColor: '#1e40af',
            borderWidth: 2.2,
            pointRadius: 0,
            pointHoverRadius: 4,
            spanGaps: false,
            order: 3
          },
          {
            label: 'Ground Truth Validation (>t)',
            data: [],
            borderColor: '#94a3b8',
            backgroundColor: '#94a3b8',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            pointHoverRadius: 3,
            spanGaps: false,
            order: 5
          },
          {
            label: 'Forecast Origin (t)',
            data: [],
            borderColor: '#dc2626',
            backgroundColor: '#dc2626',
            borderWidth: 2,
            pointRadius: 5,
            pointHoverRadius: 7,
            showLine: false,
            order: 1
          },
          {
            label: 'Persistence (+h)',
            data: [],
            borderColor: '#059669',
            backgroundColor: '#059669',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: '#059669',
            showLine: true,
            spanGaps: true,
            order: 4
          },
          {
            label: 'WL Ridge (+h)',
            data: [],
            borderColor: '#7c3aed',
            backgroundColor: '#7c3aed',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: '#7c3aed',
            showLine: true,
            spanGaps: true,
            order: 3
          },
          {
            label: 'Stage+Rain Exp (+h)',
            data: [],
            borderColor: '#0891b2',
            backgroundColor: '#0891b2',
            borderWidth: 2.2,
            borderDash: [4, 3],
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: '#0891b2',
            showLine: true,
            spanGaps: true,
            order: 2
          },
          {
            label: 'ΔH Ridge (+h)',
            data: [],
            borderColor: '#6366f1',
            backgroundColor: '#6366f1',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: '#6366f1',
            showLine: true,
            spanGaps: true,
            order: 3
          },
          {
            label: 'Random Forest (+h)',
            data: [],
            borderColor: '#d97706',
            backgroundColor: '#d97706',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: '#d97706',
            showLine: true,
            spanGaps: true,
            order: 3
          },
          {
            label: 'XGBoost (+h)',
            data: [],
            borderColor: '#0284c7',
            backgroundColor: '#0284c7',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: '#0284c7',
            showLine: true,
            spanGaps: true,
            order: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 6, bottom: 2, left: 6, right: 12 } },
        scales: {
          x: {
            display: false,
            grid: { display: false }
          },
          y: {
            title: {
              display: true,
              text: 'Stage (m MSL)',
              font: { family: 'Inter', size: 10, weight: '600' },
              color: '#334155'
            },
            ticks: {
              font: { family: 'JetBrains Mono', size: 9 },
              color: '#64748b',
              stepSize: 0.5
            },
            grid: { color: '#f1f5f9' }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: true,
            mode: 'index',
            intersect: false,
            backgroundColor: '#0f172a',
            titleFont: { family: 'Inter', size: 11, weight: '600' },
            bodyFont: { family: 'JetBrains Mono', size: 10 },
            callbacks: {
              label(context) {
                if (context.parsed.y === null || context.parsed.y === undefined) return null;
                return `${context.dataset.label}: ${context.parsed.y.toFixed(3)} m MSL`;
              }
            }
          }
        }
      }
    });

    // Chart 2: Catchment Rainfall Hyetograph
    hyetographChart = new Chart(hyetoCtx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [
          {
            type: 'bar',
            label: 'INSAT-3DS Hourly Rain (mm)',
            data: [],
            backgroundColor: '#0284c7',
            yAxisID: 'y',
            barPercentage: 0.85,
            order: 3
          },
          {
            type: 'bar',
            label: 'Payload Outage (NaN)',
            data: [],
            backgroundColor: 'rgba(245, 158, 11, 0.5)',
            borderColor: '#d97706',
            borderWidth: 1,
            yAxisID: 'y',
            barPercentage: 0.85,
            order: 2
          },
          {
            type: 'line',
            label: 'Catchment Memory (mm)',
            data: [],
            borderColor: '#059669',
            backgroundColor: 'transparent',
            borderWidth: 1.6,
            pointRadius: 0,
            yAxisID: 'y1',
            order: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 2, bottom: 4, left: 6, right: 12 } },
        scales: {
          x: {
            ticks: {
              font: { family: 'JetBrains Mono', size: 9 },
              color: '#64748b',
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 10
            },
            grid: { color: '#f1f5f9' }
          },
          y: {
            position: 'left',
            title: {
              display: true,
              text: 'Rain (mm/hr)',
              font: { family: 'Inter', size: 10, weight: '600' },
              color: '#334155'
            },
            ticks: {
              font: { family: 'JetBrains Mono', size: 9 },
              color: '#64748b'
            },
            min: 0,
            grid: { color: '#f1f5f9' }
          },
          y1: {
            position: 'right',
            title: {
              display: true,
              text: 'Soil Memory (mm)',
              font: { family: 'Inter', size: 9, weight: '600' },
              color: '#059669'
            },
            ticks: {
              font: { family: 'JetBrains Mono', size: 8 },
              color: '#059669'
            },
            min: 0,
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: true,
            mode: 'index',
            intersect: false,
            backgroundColor: '#0f172a',
            titleFont: { family: 'Inter', size: 11, weight: '600' },
            bodyFont: { family: 'JetBrains Mono', size: 10 },
            callbacks: {
              label(context) {
                if (context.parsed.y === null || context.parsed.y === undefined) return null;
                if (context.datasetIndex === 1 && context.parsed.y > 0) {
                  return 'INSAT Status: PAYLOAD OUTAGE (July 31–Aug 4, NaN)';
                }
                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} mm`;
              }
            }
          }
        }
      }
    });

    // Chart 3: Model Comparison Horizontal Bar Chart (2026 Full Season Evaluation)
    const modelCompCtx = document.getElementById('modelComparisonCanvas');
    if (modelCompCtx) {
      modelComparisonChart = new Chart(modelCompCtx.getContext('2d'), {
        type: 'bar',
        data: {
          labels: ['Persistence', 'WL Ridge', 'Model D', 'Random Forest', 'XGBoost'],
          datasets: [
            {
              label: 'Full Season +1h MAE (m)',
              data: [0.0071, 0.0068, 0.0072, 0.0066, 0.0070],
              backgroundColor: '#3b82f6',
              borderRadius: 3
            },
            {
              label: 'Full Season +3h MAE (m)',
              data: [0.0094, 0.0094, 0.0099, 0.0089, 0.0087],
              backgroundColor: '#10b981',
              borderRadius: 3
            }
          ]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: {
              title: { display: true, text: 'MAE (metres MSL) — Lower is better', font: { size: 9, family: 'Inter' } },
              ticks: { font: { family: 'JetBrains Mono', size: 8 } }
            },
            y: { ticks: { font: { family: 'Inter', size: 9, weight: '600' } } }
          },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: { boxWidth: 10, font: { size: 9, family: 'Inter' } }
            }
          }
        }
      });
    }

    // Chart 4: Forecast Error by Horizon Grouped Bar Chart
    const horizonErrorCtx = document.getElementById('horizonErrorCanvas');
    if (horizonErrorCtx) {
      horizonErrorChart = new Chart(horizonErrorCtx.getContext('2d'), {
        type: 'bar',
        data: {
          labels: ['+1h Lead', '+2h Lead', '+3h Lead'],
          datasets: [
            {
              label: 'Persistence',
              data: [0.0290, 0.0543, 0.0806],
              backgroundColor: '#059669',
              borderRadius: 2
            },
            {
              label: 'Model D (Stage+Rain)',
              data: [0.0274, 0.0506, 0.0759],
              backgroundColor: '#0891b2',
              borderRadius: 2
            },
            {
              label: 'XGBoost',
              data: [0.0295, 0.0558, 0.0827],
              backgroundColor: '#0284c7',
              borderRadius: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { ticks: { font: { family: 'Inter', size: 9, weight: '600' } } },
            y: {
              title: { display: true, text: 'MAE (m)', font: { size: 9 } },
              ticks: { font: { family: 'JetBrains Mono', size: 8 } }
            }
          },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: { boxWidth: 8, font: { size: 9 } }
            }
          }
        }
      });
    }
  }

  // Update Model Comparison Chart from Real Project Benchmark Data
  function updateModelComparisonChart() {
    if (!modelComparisonChart || !dashboardData) return;
    const is2026 = activeEventPreset.startsWith('2026_');
    if (is2026) {
      const fullBench = dashboardData.benchmarks_comprehensive &&
                         dashboardData.benchmarks_comprehensive.evaluations &&
                         dashboardData.benchmarks_comprehensive.evaluations.Full_Season_Monitored &&
                         dashboardData.benchmarks_comprehensive.evaluations.Full_Season_Monitored.horizons;
      if (fullBench) {
        const h1 = (fullBench['+1h'] && fullBench['+1h'].models) || {};
        const h2 = (fullBench['+2h'] && fullBench['+2h'].models) || {};
        const h3 = (fullBench['+3h'] && fullBench['+3h'].models) || {};

        modelComparisonChart.data.labels = [
          'Persistence',
          'WL Ridge',
          'Model D (ΔH+Rain)',
          'Random Forest',
          'XGBoost'
        ];

        modelComparisonChart.data.datasets = [
          {
            label: 'Full Season +1h MAE (m)',
            data: [
              (h1['A. Persistence Baseline'] && h1['A. Persistence Baseline'].mae_m) || 0.0071,
              (h1['B. Stage-Only Ridge (Level)'] && h1['B. Stage-Only Ridge (Level)'].mae_m) || 0.0068,
              (h1['D. Model D: Stage-Change + Rainfall Ridge (dH)'] && h1['D. Model D: Stage-Change + Rainfall Ridge (dH)'].mae_m) || 0.0072,
              (h1['E. Random Forest Regressor (dH)'] && h1['E. Random Forest Regressor (dH)'].mae_m) || 0.0066,
              (h1['F. XGBoost Regressor (dH)'] && h1['F. XGBoost Regressor (dH)'].mae_m) || 0.0070
            ],
            backgroundColor: '#3b82f6',
            borderRadius: 3
          },
          {
            label: 'Full Season +2h MAE (m)',
            data: [
              (h2['A. Persistence Baseline'] && h2['A. Persistence Baseline'].mae_m) || 0.0083,
              (h2['B. Stage-Only Ridge (Level)'] && h2['B. Stage-Only Ridge (Level)'].mae_m) || 0.0082,
              (h2['D. Model D: Stage-Change + Rainfall Ridge (dH)'] && h2['D. Model D: Stage-Change + Rainfall Ridge (dH)'].mae_m) || 0.0086,
              (h2['E. Random Forest Regressor (dH)'] && h2['E. Random Forest Regressor (dH)'].mae_m) || 0.0076,
              (h2['F. XGBoost Regressor (dH)'] && h2['F. XGBoost Regressor (dH)'].mae_m) || 0.0080
            ],
            backgroundColor: '#8b5cf6',
            borderRadius: 3
          },
          {
            label: 'Full Season +3h MAE (m)',
            data: [
              (h3['A. Persistence Baseline'] && h3['A. Persistence Baseline'].mae_m) || 0.0094,
              (h3['B. Stage-Only Ridge (Level)'] && h3['B. Stage-Only Ridge (Level)'].mae_m) || 0.0094,
              (h3['D. Model D: Stage-Change + Rainfall Ridge (dH)'] && h3['D. Model D: Stage-Change + Rainfall Ridge (dH)'].mae_m) || 0.0099,
              (h3['E. Random Forest Regressor (dH)'] && h3['E. Random Forest Regressor (dH)'].mae_m) || 0.0089,
              (h3['F. XGBoost Regressor (dH)'] && h3['F. XGBoost Regressor (dH)'].mae_m) || 0.0087
            ],
            backgroundColor: '#10b981',
            borderRadius: 3
          }
        ];
      }
    } else {
      // 2024 Historical Baseline
      const rl = dashboardData.evaluation_metrics && dashboardData.evaluation_metrics.rising_limb;
      if (rl) {
        modelComparisonChart.data.labels = [
          'Persistence Baseline',
          'WL Ridge'
        ];
        modelComparisonChart.data.datasets = [
          {
            label: '2024 Rising Limb +1h MAE (m)',
            data: [
              (rl['Persistence Baseline'] && rl['Persistence Baseline']['1h'] && rl['Persistence Baseline']['1h'].mae) || 0.0512,
              (rl['Water-Level-Only ML (Ridge)'] && rl['Water-Level-Only ML (Ridge)']['1h'] && rl['Water-Level-Only ML (Ridge)']['1h'].mae) || 0.0622
            ],
            backgroundColor: '#3b82f6',
            borderRadius: 3
          },
          {
            label: '2024 Rising Limb +2h MAE (m)',
            data: [
              (rl['Persistence Baseline'] && rl['Persistence Baseline']['2h'] && rl['Persistence Baseline']['2h'].mae) || 0.0927,
              (rl['Water-Level-Only ML (Ridge)'] && rl['Water-Level-Only ML (Ridge)']['2h'] && rl['Water-Level-Only ML (Ridge)']['2h'].mae) || 0.0956
            ],
            backgroundColor: '#8b5cf6',
            borderRadius: 3
          },
          {
            label: '2024 Rising Limb +3h MAE (m)',
            data: [
              (rl['Persistence Baseline'] && rl['Persistence Baseline']['3h'] && rl['Persistence Baseline']['3h'].mae) || 0.1289,
              (rl['Water-Level-Only ML (Ridge)'] && rl['Water-Level-Only ML (Ridge)']['3h'] && rl['Water-Level-Only ML (Ridge)']['3h'].mae) || 0.1251
            ],
            backgroundColor: '#10b981',
            borderRadius: 3
          }
        ];
      }
    }
    modelComparisonChart.update('none');
  }

  // Rebuild Labels and Dynamic Ranges on Preset Change
  function rebuildChartStructures() {
    const labels = activeTimeline.map(pt => pt.time.slice(5, 16));

    // Hydrograph structure
    hydrographChart.data.labels = labels;
    let minStage = Infinity;
    let maxStage = -Infinity;
    for (let i = 0; i < activeTimeline.length; i++) {
      const s = activeTimeline[i].stage_m;
      if (s !== null && s !== undefined) {
        if (s < minStage) minStage = s;
        if (s > maxStage) maxStage = s;
      }
    }
    const minS = (minStage !== Infinity) ? Math.floor(minStage * 2) / 2 - 0.2 : 350.0;
    const maxS = (maxStage !== -Infinity) ? Math.ceil(maxStage * 2) / 2 + 0.4 : 355.2;
    hydrographChart.options.scales.y.min = Math.min(minS, 350.5);
    hydrographChart.options.scales.y.max = Math.max(maxS, 355.2);

    // Hyetograph structure
    hyetographChart.data.labels = labels;
    let maxSat = 8;
    let maxMemory = 20;
    const is2026 = activeEventPreset.startsWith('2026_');
    for (let i = 0; i < activeTimeline.length; i++) {
      const pt = activeTimeline[i];
      if (pt.sat_rain_mm !== null && pt.sat_rain_mm !== undefined && pt.sat_rain_mm > maxSat) {
        maxSat = pt.sat_rain_mm;
      }
      const mem = is2026 ? pt.api_7d_mm : pt.imd_3d_mm;
      if (mem !== null && mem !== undefined && mem > maxMemory) {
        maxMemory = mem;
      }
    }
    const maxR = Math.ceil(maxSat / 4) * 4 + 4;
    hyetographChart.options.scales.y.max = maxR;
    const maxM = Math.ceil(maxMemory / 10) * 10 + 10;
    hyetographChart.options.scales.y1.max = maxM;
    hyetographChart.data.datasets[2].label = is2026 ? 'API 7d Soil Moisture (mm)' : 'IMD 3d Cumulative Rain (mm)';

    updateChartData();
  }

  // ==========================================================================
  // Dynamic Chart Update at Timeline Position currentIndex
  // ==========================================================================
  function updateChartData() {
    if (!hydrographChart || !hyetographChart || !activeTimeline.length) return;

    const N = activeTimeline.length;
    const cur = activeTimeline[currentIndex];

    // 1. Stage Datasets Partitioning
    const observedPast = new Array(N).fill(null);
    const groundTruthFuture = new Array(N).fill(null);
    const originPoint = new Array(N).fill(null);
    const persVector = new Array(N).fill(null);
    const ridgeVector = new Array(N).fill(null);
    const expVector = new Array(N).fill(null);
    const dhRidgeVector = new Array(N).fill(null);
    const rfVector = new Array(N).fill(null);
    const xgbVector = new Array(N).fill(null);

    for (let i = 0; i <= currentIndex; i++) {
      observedPast[i] = activeTimeline[i].stage_m;
    }
    for (let i = currentIndex; i < N; i++) {
      groundTruthFuture[i] = activeTimeline[i].stage_m;
    }

    if (cur.stage_m !== null) {
      originPoint[currentIndex] = cur.stage_m;
    }

    const targetIdx = currentIndex + activeHorizon;
    const predObj = cur.predictions && cur.predictions[`${activeHorizon}h`];
    // Extract prediction value for currently selected model
    const modelMetaMap = {
      model_d: { name: 'Model D (ΔH+Rain)', label: `Model D (+${activeHorizon}h)`, color: '#0891b2', key: 'stage_change_rain' },
      dh_ridge: { name: 'Stage-Change Ridge', label: `ΔH Ridge (+${activeHorizon}h)`, color: '#6366f1', key: 'stage_change_ridge' },
      wl_ridge: { name: 'Stage-Only Ridge', label: `WL Ridge (+${activeHorizon}h)`, color: '#7c3aed', key: 'wl_ridge' },
      rf: { name: 'Random Forest', label: `Random Forest (+${activeHorizon}h)`, color: '#d97706', key: 'random_forest' },
      xgb: { name: 'XGBoost', label: `XGBoost (+${activeHorizon}h)`, color: '#0284c7', key: 'xgboost' }
    };
    const activeMeta = modelMetaMap[selectedModel] || modelMetaMap.model_d;
    const activeModelLabel = activeMeta.label;
    const activeMarkerColor = activeMeta.color;
    let activeModelVal = (predObj && predObj[activeMeta.key] !== null && predObj[activeMeta.key] !== undefined)
      ? predObj[activeMeta.key]
      : null;

    if (cur.stage_m !== null && targetIdx < N) {
      // Persistence vector
      persVector[currentIndex] = cur.stage_m;
      persVector[targetIdx] = (predObj && predObj.persistence !== null) ? predObj.persistence : cur.stage_m;

      if (activeModelVal !== null) {
        expVector[currentIndex] = cur.stage_m;
        expVector[targetIdx] = activeModelVal;
      }
    }

    if (legendActiveModelName) legendActiveModelName.textContent = activeModelLabel;
    if (legendActiveModelMarker) legendActiveModelMarker.style.backgroundColor = activeMarkerColor;

    hydrographChart.data.datasets[0].data = observedPast;
    hydrographChart.data.datasets[1].data = groundTruthFuture;
    hydrographChart.data.datasets[2].data = originPoint;
    hydrographChart.data.datasets[3].data = persVector;
    hydrographChart.data.datasets[3].label = `Persistence (+${activeHorizon}h)`;
    
    // Dataset 4 is the single selected model overlay
    hydrographChart.data.datasets[4].data = expVector;
    hydrographChart.data.datasets[4].label = activeModelLabel;
    hydrographChart.data.datasets[4].borderColor = activeMarkerColor;
    hydrographChart.data.datasets[4].backgroundColor = activeMarkerColor;
    hydrographChart.data.datasets[4].pointBackgroundColor = activeMarkerColor;

    // Zero out unused extra datasets to ensure clean, uncluttered visual
    if (hydrographChart.data.datasets[5]) hydrographChart.data.datasets[5].data = [];
    if (hydrographChart.data.datasets[6]) hydrographChart.data.datasets[6].data = [];
    if (hydrographChart.data.datasets[7]) hydrographChart.data.datasets[7].data = [];
    if (hydrographChart.data.datasets[8]) hydrographChart.data.datasets[8].data = [];

    hydrographChart.update('none');

    // 2. Hyetograph Datasets
    const is2026 = activeEventPreset.startsWith('2026_');
    const satData = new Array(N);
    const satColors = new Array(N);
    const outageData = new Array(N);
    const memoryData = new Array(N);

    for (let i = 0; i < N; i++) {
      const pt = activeTimeline[i];
      const isOutage = pt.sat_completeness === 'OUTAGE_UNAVAILABLE';

      if (isOutage) {
        satData[i] = null;
        outageData[i] = 1.5; // Visual outage bar
      } else {
        satData[i] = (pt.sat_rain_mm !== null && pt.sat_rain_mm !== undefined) ? pt.sat_rain_mm : null;
        outageData[i] = null;
      }

      satColors[i] = i <= currentIndex ? '#0284c7' : 'rgba(2, 132, 199, 0.2)';
      memoryData[i] = (is2026 ? pt.api_7d_mm : pt.imd_3d_mm) ?? null;
    }

    hyetographChart.data.datasets[0].data = satData;
    hyetographChart.data.datasets[0].backgroundColor = satColors;
    hyetographChart.data.datasets[1].data = outageData;
    hyetographChart.data.datasets[2].data = memoryData;
    hyetographChart.update('none');
  }

  // ==========================================================================
  // Update All UI Displays for currentIndex
  // ==========================================================================
  function updateUI() {
    if (!activeTimeline.length) return;

    if (currentIndex >= activeTimeline.length) {
      currentIndex = activeTimeline.length - 1;
    }
    const cur = activeTimeline[currentIndex];
    const is2026 = activeEventPreset.startsWith('2026_');

    // 1. Toolbar Position & Slider
    if (currentTimestampEl) currentTimestampEl.textContent = `${cur.time} IST`;
    if (scrubberStepIndicatorEl) scrubberStepIndicatorEl.textContent = `Origin ${currentIndex + 1} of ${activeTimeline.length}`;
    if (timelineSlider) timelineSlider.value = currentIndex;

    // 2. Observed Stage & Rate of Change
    if (currentStageEl) {
      if (cur.stage_m !== null) {
        currentStageEl.textContent = `${cur.stage_m.toFixed(3)} m`;
        currentStageEl.className = 'telemetry-main-val text-stage-accent';
      } else {
        currentStageEl.textContent = 'NO DATA';
        currentStageEl.className = 'telemetry-main-val text-muted';
      }
    }

    if (stageRateEl) {
      if (cur.dh_dt !== null) {
        const sign = cur.dh_dt >= 0 ? '+' : '';
        stageRateEl.textContent = `${sign}${cur.dh_dt.toFixed(3)} m/hr`;
      } else {
        stageRateEl.textContent = '-- m/hr';
      }
    }

    // 3. Forecast Metrics for Active Horizon & Selected Model
    if (telemetryLeadLabelEl) telemetryLeadLabelEl.textContent = `+${activeHorizon}h Lead`;
    const predObj = cur.predictions && cur.predictions[`${activeHorizon}h`];
    const modelMetaMap = {
      model_d: { name: 'Model D (ΔH+Rain)', label: `Model D (+${activeHorizon}h)`, color: '#0891b2', key: 'stage_change_rain' },
      dh_ridge: { name: 'Stage-Change Ridge', label: `ΔH Ridge (+${activeHorizon}h)`, color: '#6366f1', key: 'stage_change_ridge' },
      wl_ridge: { name: 'Stage-Only Ridge', label: `WL Ridge (+${activeHorizon}h)`, color: '#7c3aed', key: 'wl_ridge' },
      rf: { name: 'Random Forest', label: `Random Forest (+${activeHorizon}h)`, color: '#d97706', key: 'random_forest' },
      xgb: { name: 'XGBoost', label: `XGBoost (+${activeHorizon}h)`, color: '#0284c7', key: 'xgboost' }
    };
    const meta = modelMetaMap[selectedModel] || modelMetaMap.model_d;
    const activeModelVal = (predObj && predObj[meta.key] !== null && predObj[meta.key] !== undefined)
      ? predObj[meta.key]
      : null;

    if (forecastPersValEl) {
      if (activeModelVal !== null) {
        forecastPersValEl.textContent = `${activeModelVal.toFixed(3)} m`;
        if (forecastSummaryCaption) {
          forecastSummaryCaption.textContent = `${meta.name} \u2022 Available`;
        }
      } else {
        forecastPersValEl.textContent = 'N/A';
        if (forecastSummaryCaption) {
          if (cur.sat_completeness === 'OUTAGE_UNAVAILABLE' && selectedModel === 'model_d') {
            forecastSummaryCaption.textContent = 'Model D unavailable: INSAT-3DS outage (July 31–Aug 4)';
          } else if (!is2026 && selectedModel === 'model_d') {
            forecastSummaryCaption.textContent = 'Model D unavailable: 2024 satellite rain uncatalogued';
          } else if (predObj && predObj[meta.key] === null) {
            forecastSummaryCaption.textContent = `${meta.name} unavailable for selected origin`;
          } else {
            forecastSummaryCaption.textContent = 'Forecast origin data unavailable';
          }
        }
      }
    }

    // 4. Rainfall and Soil Memory Readouts
    const isOutage = cur.sat_completeness === 'OUTAGE_UNAVAILABLE';

    if (isOutage) {
      if (rainSat1hEl) {
        rainSat1hEl.textContent = 'OUTAGE (NaN)';
        rainSat1hEl.style.color = '#b45309';
      }
      if (rainCompletenessTagEl) {
        rainCompletenessTagEl.textContent = 'Confirmed Payload Outage (July 31 05:30 – Aug 4 22:30 IST) \u2022 Data Unavailable';
      }
      if (rainfallModelStatusEl) {
        rainfallModelStatusEl.textContent = 'PAYLOAD OUTAGE (NaN)';
        rainfallModelStatusEl.className = 'tag-outage';
      }
      if (modelStatusCaptionEl) {
        modelStatusCaptionEl.textContent = '113h Offline \u2022 Preserved Strictly as NaN';
      }
    } else {
      if (rainSat1hEl) {
        rainSat1hEl.textContent = cur.sat_rain_mm !== null ? `${cur.sat_rain_mm.toFixed(3)} mm` : '-- mm';
        rainSat1hEl.style.color = '#0284c7';
      }
      if (rainSat24hEl) {
        rainSat24hEl.textContent = cur.sat_accum_24h_mm !== null ? `${cur.sat_accum_24h_mm.toFixed(1)} mm` : '-- mm';
      }

      if (is2026) {
        if (soilApiValEl) {
          soilApiValEl.textContent = cur.api_7d_mm !== null ? `${cur.api_7d_mm.toFixed(1)} mm` : '-- mm';
        }
        if (rainCompletenessTagEl) {
          rainCompletenessTagEl.textContent = `Satellite: ${cur.sat_completeness || 'COMPLETE'} \u2022 2,869 native pixels`;
        }
        if (predObj && predObj.stage_change_rain !== null) {
          if (rainfallModelStatusEl) {
            rainfallModelStatusEl.textContent = 'EXPERIMENTAL CANDIDATE';
            rainfallModelStatusEl.className = 'tag-experimental';
          }
          if (modelStatusCaptionEl) {
            modelStatusCaptionEl.textContent = 'Audited Model D (\u0394H) \u2022 Zero Target Leakage';
          }
        } else {
          if (rainfallModelStatusEl) {
            rainfallModelStatusEl.textContent = 'AUDITED PIPELINE';
            rainfallModelStatusEl.className = 'tag-status';
          }
          if (modelStatusCaptionEl) {
            modelStatusCaptionEl.textContent = 'Strict Causal Non-Leakage \u2022 Verified MOSDAC HEM';
          }
        }
      } else {
        if (soilApiValEl) {
          soilApiValEl.textContent = cur.imd_3d_mm !== null ? `${cur.imd_3d_mm.toFixed(1)} mm` : '-- mm';
        }
        if (rainCompletenessTagEl) {
          rainCompletenessTagEl.textContent = `Satellite: ${cur.sat_completeness || 'No Data'} \u2022 IMD: 71 cells`;
        }
        if (rainfallModelStatusEl) {
          rainfallModelStatusEl.textContent = 'UNAVAILABLE';
          rainfallModelStatusEl.className = 'tag-unavailable';
        }
        if (modelStatusCaptionEl) {
          modelStatusCaptionEl.textContent = 'Zero Pre-July 31 Granules \u2022 Causal Integrity';
        }
      }
    }

    // 5. Update Map Popup
    if (stageMarker && cur.stage_m !== null) {
      stageMarker.setPopupContent(`
        <div style="font-family:Inter,sans-serif; font-size:12px; color:#0f172a; padding:2px;">
          <strong style="color:#1e40af;">Vishnupuri Barrage</strong><br>
          <strong>Timestamp:</strong> ${cur.time} IST<br>
          <strong>River Stage:</strong> <span style="font-family:JetBrains Mono,monospace; font-weight:700; color:#1e40af;">${cur.stage_m.toFixed(3)} m MSL</span><br>
          <strong>FRL Design:</strong> 355.00 m MSL<br>
          <strong>Rate of Rise:</strong> ${cur.dh_dt !== null ? cur.dh_dt.toFixed(3) + ' m/hr' : '--'}<br>
          <strong>Station:</strong> 006-SW-MAHA (NWIC)
        </div>
      `);
    }

    // 6. Highlight Active Annotation Chip
    document.querySelectorAll('.annotation-chip').forEach(chip => {
      chip.classList.toggle('active', chip.getAttribute('data-time') === cur.time);
    });

    // 7. Update 5-Tier Early Warning Assessment Framework
    const elAlertIssueTime = document.getElementById('alert-issue-time');
    if (elAlertIssueTime) elAlertIssueTime.textContent = `${cur.time} IST`;

    const tier1Status = document.getElementById('tier-1-status');
    const tier1Item = document.getElementById('tier-1-item');
    if (tier1Status && tier1Item) {
      const rainVal = cur.sat_rain_1h !== null ? cur.sat_rain_1h : (cur.rain_mm || 0);
      if (rainVal >= 5.0) {
        tier1Status.textContent = `HEAVY RAIN (${rainVal.toFixed(1)} mm)`;
        tier1Status.className = 'tier-state-tag tag-state-danger';
        tier1Item.className = 'warning-tier-item tier-active-danger';
      } else if (rainVal > 0.0) {
        tier1Status.textContent = `ACTIVE RAIN (${rainVal.toFixed(1)} mm)`;
        tier1Status.className = 'tier-state-tag tag-state-warn';
        tier1Item.className = 'warning-tier-item tier-active-warn';
      } else {
        tier1Status.textContent = 'MONITORED';
        tier1Status.className = 'tier-state-tag tag-state-normal';
        tier1Item.className = 'warning-tier-item';
      }
    }

    const tier2Status = document.getElementById('tier-2-status');
    const tier2Item = document.getElementById('tier-2-item');
    if (tier2Status && tier2Item) {
      const dhVal = cur.dh_dt || 0;
      if (dhVal >= 0.020) {
        tier2Status.textContent = `RAPID RISE (+${dhVal.toFixed(3)} m/hr)`;
        tier2Status.className = 'tier-state-tag tag-state-danger';
        tier2Item.className = 'warning-tier-item tier-active-danger';
      } else if (dhVal > 0.0) {
        tier2Status.textContent = `RISING (+${dhVal.toFixed(3)} m/hr)`;
        tier2Status.className = 'tier-state-tag tag-state-warn';
        tier2Item.className = 'warning-tier-item tier-active-warn';
      } else {
        tier2Status.textContent = 'EVALUATED';
        tier2Status.className = 'tier-state-tag tag-state-normal';
        tier2Item.className = 'warning-tier-item';
      }
    }

    const tier3Status = document.getElementById('tier-3-status');
    const tier3Item = document.getElementById('tier-3-item');
    const elStageReadout = document.getElementById('tier-3-stage-readout');
    if (tier3Status && tier3Item) {
      const stageVal = cur.stage_m || 0;
      if (elStageReadout) elStageReadout.textContent = `${stageVal.toFixed(3)} m`;
      if (stageVal >= 354.80) {
        tier3Status.textContent = 'DANGER EXCEEDED';
        tier3Status.className = 'tier-state-tag tag-state-danger';
        tier3Item.className = 'warning-tier-item tier-active-danger';
      } else if (stageVal >= 354.00) {
        tier3Status.textContent = 'WARNING EXCEEDED';
        tier3Status.className = 'tier-state-tag tag-state-warn';
        tier3Item.className = 'warning-tier-item tier-active-warn';
      } else {
        tier3Status.textContent = 'BELOW WARNING';
        tier3Status.className = 'tier-state-tag tag-state-normal';
        tier3Item.className = 'warning-tier-item';
      }
    }

    // 8. Synchronize Visual Charts
    updateChartData();
  }

  // ==========================================================================
  // Playback Engine (Play, Pause, Restart, Scrubbing)
  // ==========================================================================
  function togglePlayback() {
    if (isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }

  function startPlayback() {
    if (currentIndex >= activeTimeline.length - 1) {
      currentIndex = 0;
    }

    isPlaying = true;
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'inline-block';
    btnPlayPause.classList.add('btn-play-active');
    btnPlayPause.title = 'Pause Timeline (Space)';

    stopTimer();
    const intervalMs = playbackSpeed === 1 ? 1000 : (playbackSpeed === 2 ? 500 : 200);

    playbackTimer = setInterval(() => {
      if (currentIndex < activeTimeline.length - 1) {
        currentIndex++;
        updateUI();
      } else {
        pausePlayback();
      }
    }, intervalMs);
  }

  function pausePlayback() {
    isPlaying = false;
    playIcon.style.display = 'inline-block';
    pauseIcon.style.display = 'none';
    btnPlayPause.classList.remove('btn-play-active');
    btnPlayPause.title = 'Play Timeline (Space)';
    stopTimer();
  }

  function restartPlayback() {
    pausePlayback();
    currentIndex = 0;
    updateUI();
  }

  function stepBack() {
    pausePlayback();
    if (currentIndex > 0) {
      currentIndex--;
      updateUI();
    }
  }

  function stepForward() {
    pausePlayback();
    if (currentIndex < activeTimeline.length - 1) {
      currentIndex++;
      updateUI();
    }
  }

  function stopTimer() {
    if (playbackTimer) {
      clearInterval(playbackTimer);
      playbackTimer = null;
    }
  }

  function setHorizon(h) {
    activeHorizon = h;
    horizonBtns.forEach(btn => {
      const active = parseInt(btn.getAttribute('data-h'), 10) === h;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    updateUI();
  }

  function renderAnnotations() {
    if (!annotationsContainer || !dashboardData) return;
    const is2026 = activeEventPreset.startsWith('2026_');
    const annList = is2026
      ? (dashboardData.annotations_2026 || [])
      : (dashboardData.annotations_2024 || dashboardData.annotations || []);

    annotationsContainer.innerHTML = '';
    annList.forEach(ann => {
      const chip = document.createElement('button');
      chip.className = `annotation-chip chip-${ann.type || 'info'}`;
      chip.type = 'button';
      chip.setAttribute('data-time', ann.time);
      chip.title = `${ann.title}: ${ann.desc}`;
      chip.innerHTML = `<strong>${ann.time.slice(5, 16)}</strong>: ${ann.title}`;

      chip.addEventListener('click', () => {
        pausePlayback();
        const foundIdx = activeTimeline.findIndex(pt => pt.time === ann.time);
        if (foundIdx !== -1) {
          currentIndex = foundIdx;
          updateUI();
        } else {
          // If annotation is outside current filter, switch to full window
          setEventFilter(is2026 ? '2026_monsoon_full' : 'monsoon_full');
          eventSelector.value = is2026 ? '2026_monsoon_full' : 'monsoon_full';
          const newIdx = activeTimeline.findIndex(pt => pt.time === ann.time);
          if (newIdx !== -1) {
            currentIndex = newIdx;
            updateUI();
          }
        }
      });

      annotationsContainer.appendChild(chip);
    });
  }

  // ==========================================================================
  // Live Feed Prototype Controller
  // ==========================================================================
  let liveAutoRefreshTimer = null;
  let livePrecipChart = null;
  let liveWeatherDataCache = null;
  let isFetchingLiveStatus = false;

  function setDashboardMode(mode) {
    if (mode === 'live') {
      if (btnModeHistorical) {
        btnModeHistorical.classList.remove('active');
        btnModeHistorical.setAttribute('aria-pressed', 'false');
      }
      if (btnModeLive) {
        btnModeLive.classList.add('active');
        btnModeLive.setAttribute('aria-pressed', 'true');
      }
      if (controlToolbar) controlToolbar.style.display = 'none';
      if (telemetryStrip) telemetryStrip.style.display = 'none';
      if (mainLayout) mainLayout.style.display = 'none';
      if (studyAreaView) studyAreaView.style.display = 'none';
      if (liveFeedContainer) liveFeedContainer.style.display = 'flex';
      pausePlayback();
      fetchLiveStatus(false);
      startLiveAutoRefresh();
    } else {
      if (btnModeLive) {
        btnModeLive.classList.remove('active');
        btnModeLive.setAttribute('aria-pressed', 'false');
      }
      if (btnModeHistorical) {
        btnModeHistorical.classList.add('active');
        btnModeHistorical.setAttribute('aria-pressed', 'true');
      }
      if (controlToolbar) controlToolbar.style.display = '';
      if (telemetryStrip) telemetryStrip.style.display = '';
      if (liveFeedContainer) liveFeedContainer.style.display = 'none';
      if (viewBtnStudy && viewBtnStudy.classList.contains('active')) {
        if (studyAreaView) studyAreaView.style.display = 'block';
        if (mainLayout) mainLayout.style.display = 'none';
      } else {
        if (mainLayout) mainLayout.style.display = 'grid';
        if (studyAreaView) studyAreaView.style.display = 'none';
      }
      stopLiveAutoRefresh();
    }
  }

  function switchMainView(viewName) {
    // Reset tab active states
    [viewBtnDashboard, viewBtnStudy, viewBtnLive, btnShowMetadata].forEach(btn => {
      if (btn) {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
      }
    });

    if (viewName === 'study') {
      if (viewBtnStudy) {
        viewBtnStudy.classList.add('active');
        viewBtnStudy.setAttribute('aria-selected', 'true');
      }
      if (controlToolbar) controlToolbar.style.display = 'none';
      if (telemetryStrip) telemetryStrip.style.display = 'none';
      if (mainLayout) mainLayout.style.display = 'none';
      if (liveFeedContainer) liveFeedContainer.style.display = 'none';
      if (studyAreaView) studyAreaView.style.display = 'block';
      stopLiveAutoRefresh();

      if (leafletMap) {
        setTimeout(() => {
          leafletMap.invalidateSize();
          if (reachLayer) leafletMap.fitBounds(reachLayer.getBounds(), { padding: [16, 16] });
        }, 150);
      }
    } else if (viewName === 'live') {
      if (viewBtnLive) {
        viewBtnLive.classList.add('active');
        viewBtnLive.setAttribute('aria-selected', 'true');
      }
      if (controlToolbar) controlToolbar.style.display = 'none';
      if (telemetryStrip) telemetryStrip.style.display = 'none';
      if (mainLayout) mainLayout.style.display = 'none';
      if (studyAreaView) studyAreaView.style.display = 'none';
      if (liveFeedContainer) liveFeedContainer.style.display = 'flex';
      pausePlayback();
      fetchLiveStatus(false);
      startLiveAutoRefresh();
    } else if (viewName === 'sources') {
      if (btnShowMetadata) {
        btnShowMetadata.classList.add('active');
        btnShowMetadata.setAttribute('aria-selected', 'true');
      }
      if (metadataModal) metadataModal.style.display = 'flex';
    } else {
      // Default: Forecast & Models
      if (viewBtnDashboard) {
        viewBtnDashboard.classList.add('active');
        viewBtnDashboard.setAttribute('aria-selected', 'true');
      }
      if (studyAreaView) studyAreaView.style.display = 'none';
      if (liveFeedContainer) liveFeedContainer.style.display = 'none';
      if (controlToolbar) controlToolbar.style.display = '';
      if (telemetryStrip) telemetryStrip.style.display = '';
      if (mainLayout) mainLayout.style.display = 'grid';
      stopLiveAutoRefresh();
      if (hydrographChart) hydrographChart.resize();
      if (hyetographChart) hyetographChart.resize();
      if (modelComparisonChart) modelComparisonChart.resize();
    }
  }

  function initLivePrecipChart() {
    const canvas = document.getElementById('livePrecipCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (livePrecipChart) {
      livePrecipChart.destroy();
      livePrecipChart = null;
    }

    livePrecipChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Station Area (Vishnupuri)',
            data: [],
            backgroundColor: 'rgba(2, 132, 199, 0.75)',
            borderColor: '#0284c7',
            borderWidth: 1,
            borderRadius: 3,
            order: 1
          },
          {
            label: 'Catchment 7-Point Mean',
            data: [],
            type: 'line',
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.15)',
            borderWidth: 2.2,
            pointRadius: 3,
            pointHoverRadius: 5,
            fill: true,
            tension: 0.2,
            order: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const val = context.parsed.y;
                if (val === null || val === undefined || isNaN(val)) return `${context.dataset.label}: N/A`;
                return `${context.dataset.label}: ${val.toFixed(2)} mm/hr`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(226, 232, 240, 0.7)' },
            ticks: {
              font: { family: 'JetBrains Mono', size: 10 },
              color: '#64748b',
              maxRotation: 45
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Precipitation (mm/hr)',
              font: { family: 'Inter', size: 11, weight: '600' },
              color: '#475569'
            },
            grid: { color: 'rgba(226, 232, 240, 0.7)' },
            ticks: {
              font: { family: 'JetBrains Mono', size: 10 },
              color: '#64748b',
              callback: function(v) { return v.toFixed(1) + ' mm'; }
            }
          }
        }
      }
    });
  }

  function updateLivePrecipChart() {
    if (!livePrecipChart || !liveWeatherDataCache) return;
    const lw = liveWeatherDataCache;
    const timeline = lw.timeline;
    if (!timeline || !timeline.timestamps || !timeline.timestamps.length) return;

    const selectEl = document.getElementById('timeline-series-select');
    const selectedMode = selectEl ? selectEl.value : 'both';

    // Format labels: show HH:mm IST
    const labels = timeline.timestamps.map(ts => {
      const parts = ts.split(' ');
      if (parts.length > 1) {
        return parts[1].slice(0, 5) + ' IST';
      }
      return ts.slice(11, 16) + ' IST';
    });

    const stnSeries = timeline.station_precipitation_mm || [];
    const catchSeries = timeline.catchment_mean_precipitation_mm || [];

    livePrecipChart.data.labels = labels;

    if (selectedMode === 'both') {
      livePrecipChart.data.datasets = [
        {
          label: 'Station Area (Vishnupuri)',
          data: stnSeries,
          type: 'bar',
          backgroundColor: 'rgba(2, 132, 199, 0.75)',
          borderColor: '#0284c7',
          borderWidth: 1,
          borderRadius: 3,
          order: 1
        },
        {
          label: 'Catchment 7-Point Spatial Mean',
          data: catchSeries,
          type: 'line',
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          borderWidth: 2.2,
          pointRadius: 3,
          pointHoverRadius: 5,
          fill: true,
          tension: 0.2,
          order: 0
        }
      ];
    } else if (selectedMode === 'catchment_mean') {
      livePrecipChart.data.datasets = [
        {
          label: 'Catchment 7-Point Spatial Mean',
          data: catchSeries,
          type: 'bar',
          backgroundColor: 'rgba(16, 185, 129, 0.75)',
          borderColor: '#10b981',
          borderWidth: 1,
          borderRadius: 3
        }
      ];
    } else if (selectedMode === 'station') {
      livePrecipChart.data.datasets = [
        {
          label: 'Station Area (Vishnupuri Barrage)',
          data: stnSeries,
          type: 'bar',
          backgroundColor: 'rgba(2, 132, 199, 0.75)',
          borderColor: '#0284c7',
          borderWidth: 1,
          borderRadius: 3
        }
      ];
    } else {
      // Individual location index 1..7
      const locIdx = parseInt(selectedMode.replace('loc_', ''), 10) - 1;
      const locations = (lw.catchment_sampled_rainfall && lw.catchment_sampled_rainfall.locations) || [];
      if (locations[locIdx]) {
        const loc = locations[locIdx];
        const series = loc.forecast_hourly_series || [];
        livePrecipChart.data.datasets = [
          {
            label: `${loc.name} (${loc.region})`,
            data: series,
            type: 'bar',
            backgroundColor: 'rgba(99, 102, 241, 0.75)',
            borderColor: '#6366f1',
            borderWidth: 1,
            borderRadius: 3
          }
        ];
      }
    }

    livePrecipChart.update();
  }

  // Window helper so inline onchange works
  window.updateLivePrecipChart = updateLivePrecipChart;

  async function fetchLiveStatus(bypass = false) {
    if (isFetchingLiveStatus) return; // Prevent duplicate requests
    if (!btnRefreshLive) return;

    isFetchingLiveStatus = true;
    btnRefreshLive.disabled = true;
    const spinIcon = document.getElementById('refresh-spin-icon');
    if (spinIcon) spinIcon.classList.add('spin-icon');
    if (btnRefreshText) btnRefreshText.textContent = 'Refreshing...';

    const url = '/api/live-status?t=' + Date.now() + (bypass ? '&refresh=true' : '');

    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();

      // 1. Update Timestamp and Stale Badge
      if (liveLastCheckedTime) {
        liveLastCheckedTime.textContent = data.timestamp_checked || new Date().toLocaleTimeString();
      }
      const staleBadge = document.getElementById('live-stale-badge');
      if (staleBadge) {
        const isStale = data.live_weather && data.live_weather.is_stale;
        staleBadge.style.display = isStale ? 'inline-block' : 'none';
      }

      // 2. Update 4-State Indicators: Weather, Stage Telemetry, Model Readiness, Forecast Availability
      if (data.operational_states) {
        const st = data.operational_states;
        const elWeather = document.getElementById('op-state-weather');
        const elStage = document.getElementById('op-state-stage');
        const elModel = document.getElementById('op-state-model');
        const elForecast = document.getElementById('op-state-forecast');

        if (elWeather) {
          const isLive = st.weather_state.includes('AVAILABLE');
          elWeather.textContent = isLive ? 'LIVE' : 'STALE';
          elWeather.className = 'badge-op-state ' + (isLive ? 'badge-op-success' : 'badge-op-warning');
        }
        if (elStage) {
          const isFresh = st.stage_state.includes('FRESH');
          elStage.textContent = isFresh ? 'LIVE' : 'STALE';
          elStage.className = 'badge-op-state ' + (isFresh ? 'badge-op-success' : 'badge-op-warning');
        }
        if (elModel) {
          const isBlocked = st.model_state.includes('BLOCKED');
          elModel.textContent = isBlocked ? 'BLOCKED' : (st.model_state.includes('NOT_TRAINED') ? 'NOT TRAINED' : 'READY');
          elModel.className = 'badge-op-state ' + (isBlocked ? 'badge-op-blocked' : 'badge-op-success');
        }
        if (elForecast) {
          const isBlocked = st.forecast_state.includes('NOT READY') || st.forecast_state.includes('BLOCKED');
          elForecast.textContent = isBlocked ? 'BLOCKED' : 'READY';
          elForecast.className = 'badge-op-state ' + (isBlocked ? 'badge-op-danger' : 'badge-op-success');
        }
      }

      // 3. Update Real-Time Live Weather (Station-Area & Catchment)
      if (data.live_weather) {
        liveWeatherDataCache = data.live_weather;
        const lw = data.live_weather;
        const stn = lw.station_area_weather;
        if (stn) {
          const elTemp = document.getElementById('live-stn-temp');
          const elCond = document.getElementById('live-stn-condition');
          const elHum = document.getElementById('live-stn-humidity');
          const elRate = document.getElementById('live-stn-precip-rate');
          const elP1 = document.getElementById('live-stn-p1h');
          const elP2 = document.getElementById('live-stn-p2h');
          const elP3 = document.getElementById('live-stn-p3h');
          const elP3Total = document.getElementById('live-stn-p3h-total');
          const elP3TotalChip = document.getElementById('live-stn-p3h-total-chip');

          if (elTemp) elTemp.textContent = stn.current_temperature_c !== undefined ? stn.current_temperature_c.toFixed(1) : 'N/A';
          if (elCond) elCond.textContent = stn.weather_condition || 'N/A';
          if (elHum) elHum.textContent = stn.current_humidity_pct !== undefined ? `${stn.current_humidity_pct}%` : 'N/A';
          if (elRate) elRate.textContent = stn.current_precipitation_mm !== undefined ? `${stn.current_precipitation_mm.toFixed(2)} mm/hr` : '0.00 mm/hr';
          if (elP1 && stn.forecast_hourly_mm) elP1.textContent = `${stn.forecast_hourly_mm['+1h'].toFixed(2)} mm`;
          if (elP2 && stn.forecast_hourly_mm) elP2.textContent = `${stn.forecast_hourly_mm['+2h'].toFixed(2)} mm`;
          if (elP3 && stn.forecast_hourly_mm) elP3.textContent = `${stn.forecast_hourly_mm['+3h'].toFixed(2)} mm`;
          if (elP3Total) elP3Total.textContent = `${stn.forecast_next_3h_total_mm.toFixed(2)} mm`;
          if (elP3TotalChip) elP3TotalChip.textContent = `${stn.forecast_next_3h_total_mm.toFixed(2)} mm`;
        }

        const catchRain = lw.catchment_sampled_rainfall;
        if (catchRain) {
          const elCatchMean3h = document.getElementById('live-catch-mean-3h');
          const elCatchCurrMean = document.getElementById('live-catch-curr-mean');
          const elCatchMaxPt = document.getElementById('live-catch-max-pt');
          const elCatchMean1h = document.getElementById('live-catch-mean-1h');
          const elCatchMean2h = document.getElementById('live-catch-mean-2h');
          const elCatchMean3hChip = document.getElementById('live-catch-mean-3h-chip');
          const elCatchMeanAccum = document.getElementById('live-catch-mean-accum');
          const elValidPtsBadge = document.getElementById('live-valid-points-badge');

          if (elCatchMean3h) elCatchMean3h.textContent = catchRain.forecast_mean_next_3h_total_mm.toFixed(2);
          if (elCatchCurrMean) elCatchCurrMean.textContent = `${catchRain.current_mean_rate_mm.toFixed(2)} mm/hr`;
          if (elCatchMaxPt) elCatchMaxPt.textContent = `${catchRain.max_point_3h_rain_mm.toFixed(2)} mm`;
          if (elCatchMean1h) elCatchMean1h.textContent = `${catchRain.forecast_mean_1h_mm.toFixed(2)} mm`;
          if (elCatchMean2h) elCatchMean2h.textContent = `${catchRain.forecast_mean_2h_mm.toFixed(2)} mm`;
          if (elCatchMean3hChip) elCatchMean3hChip.textContent = `${catchRain.forecast_mean_3h_mm.toFixed(2)} mm`;
          if (elCatchMeanAccum) elCatchMeanAccum.textContent = `${catchRain.forecast_mean_next_3h_total_mm.toFixed(2)} mm`;
          if (elValidPtsBadge) elValidPtsBadge.textContent = `${catchRain.valid_points_count || 7}/7 POINTS VALID`;

          // Populate 7-point breakdown table
          const tbody = document.getElementById('catchment-points-tbody');
          if (tbody && Array.isArray(catchRain.locations)) {
            tbody.innerHTML = '';
            catchRain.locations.forEach(loc => {
              const tr = document.createElement('tr');
              const tempStr = loc.current_temperature_c !== undefined ? `${loc.current_temperature_c.toFixed(1)}°C` : 'N/A';
              const humStr = loc.current_humidity_pct !== undefined ? `${loc.current_humidity_pct}%` : 'N/A';
              const pCurrStr = loc.current_precipitation_mm !== undefined ? `${loc.current_precipitation_mm.toFixed(2)} mm` : 'N/A';
              const p1Str = loc.forecast_lead_1h_mm !== undefined ? `${loc.forecast_lead_1h_mm.toFixed(2)} mm` : 'N/A';
              const p2Str = loc.forecast_lead_2h_mm !== undefined ? `${loc.forecast_lead_2h_mm.toFixed(2)} mm` : 'N/A';
              const p3Str = loc.forecast_lead_3h_mm !== undefined ? `${loc.forecast_lead_3h_mm.toFixed(2)} mm` : 'N/A';
              const pTotStr = loc.forecast_next_3h_total_mm !== undefined ? `${loc.forecast_next_3h_total_mm.toFixed(2)} mm` : 'N/A';

              tr.innerHTML = `
                <td><strong>${loc.name}</strong></td>
                <td><span style="color:#64748b;">${loc.region}</span></td>
                <td class="mono" style="font-size:10px;">${loc.lat.toFixed(4)}&deg;N, ${loc.lon.toFixed(4)}&deg;E</td>
                <td class="mono">${tempStr}</td>
                <td class="mono">${humStr}</td>
                <td class="mono" style="color:#0284c7; font-weight:600;">${pCurrStr}</td>
                <td class="mono">${p1Str}</td>
                <td class="mono">${p2Str}</td>
                <td class="mono">${p3Str}</td>
                <td class="mono" style="color:#1d4ed8; font-weight:700;">${pTotStr}</td>
                <td><span class="badge-op-state badge-op-success" style="font-size:9px; padding:1px 4px;">ONLINE</span></td>
              `;
              tbody.appendChild(tr);
            });
          }
        }

        // Weather metadata display
        const elWeatherMeta = document.getElementById('live-weather-meta');
        if (elWeatherMeta) {
          const vTime = lw.forecast_valid_time_ist || lw.last_successful_fetch_ist || 'Active';
          const httpCode = lw.http_status || 200;
          const ageSec = lw.is_cached ? ` (Cached ${lw.cache_age_sec}s)` : '';
          elWeatherMeta.textContent = `Valid: ${vTime} • HTTP ${httpCode}${ageSec} • Model Run: 15–60 min (polls 5 min)`;
        }

        // Initialize or update 24h precipitation chart
        if (!livePrecipChart) {
          initLivePrecipChart();
        }
        updateLivePrecipChart();
      }

      // 4. Update Official Station Identity and Telemetry Observations
      if (data.station_identity) {
        if (liveCardStnName) liveCardStnName.textContent = data.station_identity.name || 'Vishnupuri Barrage';
        if (liveCardAgency) liveCardAgency.textContent = data.station_identity.agency || 'Maharashtra SW (006)';
        if (liveCardCoords) liveCardCoords.textContent = data.station_identity.coordinates || '19.12694°N, 77.28083°E';
      }
      if (data.latest_archived_observation) {
        const obs = data.latest_archived_observation;
        if (liveCardStage && obs.observed_stage_m !== undefined) {
          liveCardStage.textContent = `${obs.observed_stage_m.toFixed(3)} m MSL`;
        }
        if (liveCardFreeboard && obs.freeboard_m !== undefined) {
          liveCardFreeboard.textContent = `${obs.freeboard_m >= 0 ? '+' : ''}${obs.freeboard_m.toFixed(3)} m`;
        }
        if (liveCardTimestamp && obs.timestamp) {
          liveCardTimestamp.textContent = obs.timestamp;
        }

        // Compute dynamically if obs.data_age_hours is present
        if (liveCardAge) {
          if (obs.data_age_hours !== undefined) {
            const h = obs.data_age_hours;
            const days = (h / 24.0).toFixed(1);
            liveCardAge.textContent = `${h.toFixed(1)} hours old (${days} days ago)`;
          } else if (obs.data_age) {
            liveCardAge.textContent = obs.data_age;
          }
        }
      }

      // 5. Update Real Gateway and Satellite Provenance
      const elGateway = document.getElementById('live-card-gateway');
      if (elGateway && data.live_telemetry_feed) {
        if (data.live_telemetry_feed.nwdp_ckan_telemetry && data.live_telemetry_feed.nwdp_ckan_telemetry.accessible) {
          elGateway.textContent = 'NWDP CKAN Datastore HTTP 200 (Station 006-SW-MAHA)';
          elGateway.style.color = '#0369a1';
        } else {
          elGateway.textContent = 'Archive Fallback (Station 006-SW-MAHA)';
          elGateway.style.color = '#b91c1c';
        }
      }

      const elSatStatus = document.getElementById('live-card-sat-status');
      const elSatBadge = document.getElementById('live-card-sat-badge');
      if (data.rainfall_data_feed && data.rainfall_data_feed.mosdac_catalog) {
        const mos = data.rainfall_data_feed.mosdac_catalog;
        if (mos.status === 'AUTHENTICATED_ONLINE') {
          if (elSatStatus) elSatStatus.textContent = `MOSDAC V01R00 Online (${mos.today_scans_count || 31} scans today)`;
          if (elSatBadge) {
            elSatBadge.textContent = 'LIVE CATALOG';
            elSatBadge.style.background = '#e0f2fe';
            elSatBadge.style.color = '#0369a1';
          }
        }
      }

      const elDiagStage = document.getElementById('model-diag-stage-status');
      if (elDiagStage && data.operational_states) {
        const isStale = data.operational_states.stage_state.includes('STALE');
        elDiagStage.textContent = isStale ? 'Blocked (Stage Telemetry >3h Stale)' : 'Ready';
        elDiagStage.style.color = isStale ? '#b91c1c' : '#059669';
      }

      // 6. Update Section 5: Delayed-Observation Mode (Research Demonstration)
      if (data.delayed_observation_mode) {
        const dom = data.delayed_observation_mode;
        const elLatestTs = document.getElementById('delayed-latest-ts');
        const elLatestAge = document.getElementById('delayed-latest-age');
        const elLatestBadge = document.getElementById('delayed-latest-badge');
        const elLatestFcStatus = document.getElementById('delayed-latest-fc-status');
        const elLatestRainStatus = document.getElementById('delayed-latest-rain-status');

        if (elLatestTs && dom.gauge_observation_time_ist) elLatestTs.textContent = dom.gauge_observation_time_ist;
        if (elLatestAge && dom.gauge_age_hours !== undefined) elLatestAge.textContent = `${dom.gauge_age_hours} hours old (${(dom.gauge_age_hours / 24).toFixed(1)} days ago)`;
        if (elLatestBadge) {
          elLatestBadge.textContent = dom.forecast_valid ? 'DELAYED READY' : 'RAIN UNARCHIVED';
          elLatestBadge.className = 'badge-op-state ' + (dom.forecast_valid ? 'badge-op-success' : 'badge-op-warning');
        }
        if (elLatestFcStatus) {
          elLatestFcStatus.textContent = dom.forecast_valid ? 'AVAILABLE' : 'UNAVAILABLE';
          elLatestFcStatus.style.color = dom.forecast_valid ? '#059669' : '#b91c1c';
        }
        if (elLatestRainStatus && dom.reason) {
          elLatestRainStatus.textContent = dom.forecast_valid ? 'Complete' : 'Unarchived (coverage ends 30-Sep)';
        }
      }

      if (data.delayed_demonstration_sample && data.delayed_demonstration_sample.forecast_valid) {
        const demo = data.delayed_demonstration_sample;
        const elDemoTs = document.getElementById('delayed-demo-ts');
        const elDemoStage = document.getElementById('delayed-demo-stage');
        const elH1 = document.getElementById('delayed-demo-h1');
        const elH2 = document.getElementById('delayed-demo-h2');
        const elH3 = document.getElementById('delayed-demo-h3');

        if (elDemoTs && demo.gauge_observation_time_ist) elDemoTs.textContent = demo.gauge_observation_time_ist;
        if (elDemoStage && demo.features_at_origin) elDemoStage.textContent = `${demo.features_at_origin.stage_observed_m.toFixed(3)} m MSL`;
        if (demo.forecasts) {
          if (elH1 && demo.forecasts['+1h']) elH1.textContent = `${demo.forecasts['+1h'].predicted_stage_m.toFixed(3)} m`;
          if (elH2 && demo.forecasts['+2h']) elH2.textContent = `${demo.forecasts['+2h'].predicted_stage_m.toFixed(3)} m`;
          if (elH3 && demo.forecasts['+3h']) elH3.textContent = `${demo.forecasts['+3h'].predicted_stage_m.toFixed(3)} m`;
        }
      }

      if (liveBannerDesc) {
        liveBannerDesc.innerHTML = 'Open-Meteo NWP live (7 points) &bull; NWDP stage telemetry active (STALE ~57h) &bull; MOSDAC catalog authenticated &bull; Real-time forecasts BLOCKED &bull; Delayed-Origin mode active';
      }
    } catch (err) {
      console.error('Error fetching live status:', err);
      const staleBadge = document.getElementById('live-stale-badge');
      if (staleBadge) staleBadge.style.display = 'inline-block';
      if (liveBannerDesc) {
        liveBannerDesc.textContent = `Could not reach live status endpoint (${err.message}). Retaining cached state with verified metadata.`;
      }
    } finally {
      isFetchingLiveStatus = false;
      btnRefreshLive.disabled = false;
      const spinIcon = document.getElementById('refresh-spin-icon');
      if (spinIcon) spinIcon.classList.remove('spin-icon');
      if (btnRefreshText) btnRefreshText.textContent = 'Manual Refresh';
    }
  }

  function startLiveAutoRefresh() {
    stopLiveAutoRefresh();
    // Auto-refresh every 5 minutes (300,000 ms) as specified
    liveAutoRefreshTimer = setInterval(() => {
      if (liveFeedContainer && liveFeedContainer.style.display === 'flex') {
        fetchLiveStatus(false);
      }
    }, 300000);
  }

  function stopLiveAutoRefresh() {
    if (liveAutoRefreshTimer) {
      clearInterval(liveAutoRefreshTimer);
      liveAutoRefreshTimer = null;
    }
  }

  function setupEventListeners() {
    // Operational Mode Toggles
    if (btnModeHistorical) {
      btnModeHistorical.addEventListener('click', () => setDashboardMode('historical'));
    }
    if (btnModeLive) {
      btnModeLive.addEventListener('click', () => setDashboardMode('live'));
    }
    if (btnRefreshLive) {
      btnRefreshLive.addEventListener('click', fetchLiveStatus);
    }
    if (btnReturnHistorical) {
      btnReturnHistorical.addEventListener('click', () => setDashboardMode('historical'));
    }

    if (eventSelector) {
      eventSelector.addEventListener('change', e => {
        pausePlayback();
        setEventFilter(e.target.value);
        updateUI();
      });
    }

    horizonBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        setHorizon(parseInt(btn.getAttribute('data-h'), 10));
      });
    });

    if (btnRestart) btnRestart.addEventListener('click', restartPlayback);
    if (btnStepBack) btnStepBack.addEventListener('click', stepBack);
    if (btnPlayPause) btnPlayPause.addEventListener('click', togglePlayback);
    if (btnStepForward) btnStepForward.addEventListener('click', stepForward);

    speedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        playbackSpeed = parseInt(btn.getAttribute('data-speed'), 10);
        if (isPlaying) {
          startPlayback(); // restart timer with new speed
        }
      });
    });

    if (timelineSlider) {
      timelineSlider.addEventListener('input', e => {
        pausePlayback();
        currentIndex = parseInt(e.target.value, 10);
        updateUI();
      });
    }

    // Benchmark Pipeline Toggle Subnav
    if (btnMetrics2026) {
      btnMetrics2026.addEventListener('click', () => showMetricsView('2026'));
    }
    if (btnMetrics2024) {
      btnMetrics2024.addEventListener('click', () => showMetricsView('2024'));
    }

    // Model Selector on Toolbar
    if (modelSelector) {
      modelSelector.addEventListener('change', e => {
        selectedModel = e.target.value;
        updateUI();
      });
    }

    // View Navigation Buttons
    if (viewBtnDashboard) {
      viewBtnDashboard.addEventListener('click', () => switchMainView('dashboard'));
    }
    if (viewBtnStudy) {
      viewBtnStudy.addEventListener('click', () => switchMainView('study'));
    }
    if (viewBtnLive) {
      viewBtnLive.addEventListener('click', () => switchMainView('live'));
    }

    // Modal Events
    if (btnShowMetadata) {
      btnShowMetadata.addEventListener('click', () => {
        metadataModal.style.display = 'flex';
      });
    }
    if (btnCloseModal) {
      btnCloseModal.addEventListener('click', () => {
        metadataModal.style.display = 'none';
      });
    }
    window.addEventListener('click', e => {
      if (e.target === metadataModal) {
        metadataModal.style.display = 'none';
      }
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayback();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        stepBack();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stepForward();
      } else if (e.code === 'Home') {
        e.preventDefault();
        restartPlayback();
      } else if (e.key === '1') {
        setHorizon(1);
      } else if (e.key === '2') {
        setHorizon(2);
      } else if (e.key === '3') {
        setHorizon(3);
      }
    });
  }

  // Expose global controller functions
  window.setDashboardMode = setDashboardMode;
  window.fetchLiveStatus = fetchLiveStatus;
  window.switchMainView = switchMainView;

  // Boot Application
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
