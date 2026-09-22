/**
 * gee_imerg_rainfall_extractor.js
 *
 * Google Earth Engine Script to extract NASA GPM IMERG V07B precipitation
 * for the 5 verified Mumbai flood and control events.
 *
 * Dataset: NASA/GPM_L3/IMERG_V07
 * Temporal resolution: 30 minutes
 * Spatial resolution: 0.1 degree (~11 km)
 * Band: 'precipitation' (mm/hr) -> converted to mm by multiplying by 0.5 hr
 *
 * CRITICAL TEMPORAL RULE:
 * Rainfall strictly ends at or before the exact Sentinel-1 post-acquisition time (T).
 * No future rainfall is included.
 *
 * HOW TO RUN:
 * 1. Open Google Earth Engine Code Editor (https://code.earthengine.google.com/).
 * 2. Paste this script and click "Run".
 * 3. The exact event rainfall summary will be printed to the Console.
 * 4. Go to the "Tasks" tab and click "Run" on "Export_mumbai_imerg_rainfall_events"
 *    to export the CSV to Google Drive (folder: Mumbai_Flood_Project).
 */

// ── 1. Study Area Geometry ──────────────────────────────────────────────────
var mumbai = ee.Geometry.Rectangle([72.75, 18.85, 73.05, 19.32]);

// ── 2. Event Catalog with Exact Sentinel-1 Post Timestamps ──────────────────
var events = [
  {
    event_id: "EV_2018_07_19",
    event_type: "flood",
    post_acquisition_time: "2018-07-19 01:02:53",
    t_end: "2018-07-19T01:00:00",
    t_24h_start: "2018-07-18T01:00:00",
    t_72h_start: "2018-07-16T01:00:00"
  },
  {
    event_id: "EV_2018_08_24",
    event_type: "flood",
    post_acquisition_time: "2018-08-24 01:02:55",
    t_end: "2018-08-24T01:00:00",
    t_24h_start: "2018-08-23T01:00:00",
    t_72h_start: "2018-08-21T01:00:00"
  },
  {
    event_id: "EV_2019_09_24",
    event_type: "flood",
    post_acquisition_time: "2019-09-24 01:03:02",
    t_end: "2019-09-24T01:00:00",
    t_24h_start: "2019-09-23T01:00:00",
    t_72h_start: "2019-09-21T01:00:00"
  },
  {
    event_id: "EV_2023_07_29",
    event_type: "flood",
    post_acquisition_time: "2023-07-29 01:03:31",
    t_end: "2023-07-29T01:00:00",
    t_24h_start: "2023-07-28T01:00:00",
    t_72h_start: "2023-07-26T01:00:00"
  },
  {
    event_id: "CTRL_2024_03_12",
    event_type: "control",
    post_acquisition_time: "2024-03-13 01:03:29",
    t_end: "2024-03-13T01:00:00",
    t_24h_start: "2024-03-12T01:00:00",
    t_72h_start: "2024-03-10T01:00:00"
  }
];

// ── 3. IMERG Collection & Extraction ────────────────────────────────────────
var imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
  .select("precipitation");

print("=== EXTRACTING REAL NASA GPM IMERG V07B RAINFALL ===");

var features = events.map(function(ev) {
  // 24h accumulation: 48 half-hour images (mm/hr * 0.5 hr = mm)
  var col24h = imerg.filterDate(ev.t_24h_start, ev.t_end);
  var rain24hImg = col24h.map(function(img) {
    return img.multiply(0.5);
  }).sum().rename("rainfall_24h_mm");

  // 72h accumulation: 144 half-hour images (mm/hr * 0.5 hr = mm)
  var col72h = imerg.filterDate(ev.t_72h_start, ev.t_end);
  var rain72hImg = col72h.map(function(img) {
    return img.multiply(0.5);
  }).sum().rename("antecedent_rainfall_72h_mm");

  // Stack images
  var combinedRain = ee.Image([rain24hImg, rain72hImg]);

  // Spatial reduction: mean areal rainfall across Mumbai study area
  var stats = combinedRain.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: mumbai,
    scale: 11132, // ~0.1 degree native IMERG resolution
    bestEffort: true
  });

  return ee.Feature(null, {
    'event_id': ev.event_id,
    'event_type': ev.event_type,
    'post_acquisition_time': ev.post_acquisition_time,
    't_window_24h_start': ev.t_24h_start,
    't_window_72h_start': ev.t_72h_start,
    't_window_end': ev.t_end,
    'rainfall_24h_mm': stats.get('rainfall_24h_mm'),
    'antecedent_rainfall_72h_mm': stats.get('antecedent_rainfall_72h_mm'),
    'rainfall_source': 'GPM_IMERG_V07B',
    'rainfall_temporal_resolution': '30_min',
    'rainfall_spatial_resolution': '0.1_degree',
    'spatial_aggregation_method': 'study_area_mean'
  });
});

var rainCollection = ee.FeatureCollection(features);

// Print results to GEE Console
print("IMERG Rainfall Events Summary Table:", rainCollection);

// Export intermediate table to Drive
Export.table.toDrive({
  collection: rainCollection,
  description: "Export_mumbai_imerg_rainfall_events",
  folder: "Mumbai_Flood_Project",
  fileNamePrefix: "mumbai_imerg_rainfall_events",
  fileFormat: "CSV",
  selectors: [
    'event_id',
    'event_type',
    'post_acquisition_time',
    't_window_24h_start',
    't_window_72h_start',
    't_window_end',
    'rainfall_24h_mm',
    'antecedent_rainfall_72h_mm',
    'rainfall_source',
    'rainfall_temporal_resolution',
    'rainfall_spatial_resolution',
    'spatial_aggregation_method'
  ]
});

// Evaluate and print formatted JSON directly to GEE console for quick copy
rainCollection.evaluate(function(fc) {
  if (fc && fc.features) {
    var rows = fc.features.map(function(f) { return f.properties; });
    print("=== COPYABLE JSON ARRAY FOR build_imerg_rainfall.py ===");
    print(JSON.stringify(rows));
  }
});

print("=== READY ===");
print("1. Inspect printed rainfall summary in the Console tab.");
print("2. In the Tasks tab, click 'Run' on 'Export_mumbai_imerg_rainfall_events' to save CSV to Drive.");

