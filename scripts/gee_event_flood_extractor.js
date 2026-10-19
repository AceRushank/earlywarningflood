/**
 * scripts/gee_event_flood_extractor.js
 * =====================================
 * Google Earth Engine (GEE) Sentinel-1 SAR Flood Inundation Extractor
 * Location: Godavari River, Vishnupuri Barrage, Nanded, Maharashtra
 * Bounding Box: [77.0208°E, 18.8958°N, 77.5250°E, 19.3042°N] (EPSG:4326)
 *
 * METHODOLOGY & NON-NEGOTIABLE SCIENTIFIC RULES:
 * 1. Uses Sentinel-1 SAR GRD (C-band, VV/VH polarization, IW mode, 10m).
 * 2. Compares flood event imagery against dry-season pre-event reference baseline.
 * 3. Change detection: thresholding ratio/difference (VH < -2.5 dB decrease or < -16 dB absolute).
 * 4. Masks permanent water using JRC Global Surface Water seasonality >= 10 months/year.
 * 5. Masks steep terrain using Copernicus DEM / SRTM 30m slope < 5 degrees.
 * 6. Eliminates isolated speckle noise using connectedPixelCount >= 8 pixels.
 * 7. Aggregates inundation onto the standardized 500m x 500m projected grid (EPSG:32643).
 * 8. Exports flood_area_m2, flood_fraction, and binary observed_flooded (fraction >= 0.05).
 *
 * NOTE: GEE requires execution within the Earth Engine Code Editor or GEE Python API
 * authenticated with cloud credentials. Never substitute synthetic labels if unexecuted.
 */

// Define Vishnupuri River Reach Corridor Study Area (1,231.5 km²)
var aoi = ee.Geometry.Rectangle([77.0208, 18.8958, 77.5250, 19.3042]);

// Target Grid Configuration
var TARGET_PROJECTION = 'EPSG:32643'; // WGS 84 / UTM Zone 43N
var GRID_SCALE = 500; // 500 meters

// Documented Flood Events for Godavari Reach at Nanded
var events = [
  {
    name: '2024_August_Peak',
    event_start: '2024-08-02',
    event_end: '2024-08-06',
    pre_event_start: '2024-05-01',
    pre_event_end: '2024-05-31',
    cwc_peak_stage: 354.798 // m MSL (CWC Alert Level 354.0m, Danger 354.8m)
  },
  {
    name: '2026_July_Inception',
    event_start: '2026-07-28',
    event_end: '2026-08-01',
    pre_event_start: '2026-05-01',
    pre_event_end: '2026-05-31',
    cwc_peak_stage: 351.468 // Rising limb onset
  }
];

// 1. Terrain & Permanent Water Conditioning Masks
var dem = ee.Image('COPERNICUS/DEM/GLO30').select('DEM').clip(aoi);
var slope = ee.Terrain.slope(dem);
var flatMask = slope.lt(5.0); // Only flat floodplain slopes (< 5°) are flood prone

// JRC Global Surface Water (Permanent water: occurrence >= 80% or seasonality >= 10 months)
var jrcGsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(aoi);
var permanentWater = jrcGsw.select('seasonality').gte(10).or(jrcGsw.select('occurrence').gte(80));
var nonPermanentWater = permanentWater.not();

// 2. Process each event
events.forEach(function(ev) {
  print('Processing event:', ev.name);

  // Filter Sentinel-1 GRD SAR Collection (Descending or Ascending IW, VH polarization)
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));

  // Pre-event reference baseline (dry season median)
  var preEvent = s1.filterDate(ev.pre_event_start, ev.pre_event_end)
    .select('VH')
    .median()
    .clip(aoi);

  // Event imagery (minimum backscatter during event window captures maximum water surface)
  var eventImage = s1.filterDate(ev.event_start, ev.event_end)
    .select('VH')
    .min()
    .clip(aoi);

  // Change detection: backscatter drop > 2.5 dB and absolute backscatter < -16 dB
  var diff = eventImage.subtract(preEvent);
  var rawWater = diff.lt(-2.5).and(eventImage.lt(-16.0));

  // Apply terrain slope filter and mask out permanent river channel
  var candidateFlood = rawWater.updateMask(flatMask).updateMask(nonPermanentWater);

  // Connected pixel filter (>= 8 pixels at 10m = 800 m² to remove radar speckle)
  var connectedPixels = candidateFlood.connectedPixelCount(15, false);
  var filteredFlood = candidateFlood.updateMask(connectedPixels.gte(8)).unmask(0);

  // 3. Aggregate onto 500m Grid
  // flood_area_m2 = pixel count * 100 m²
  var floodAreaM2 = filteredFlood.multiply(100).reduceResolution({
    reducer: ee.Reducer.sum(),
    maxPixels: 2500
  }).reproject({
    crs: TARGET_PROJECTION,
    scale: GRID_SCALE
  });

  // Cell area = 500m * 500m = 250,000 m²
  var floodFraction = floodAreaM2.divide(250000.0).rename('flood_fraction');
  var binaryFlooded = floodFraction.gte(0.05).rename('observed_flooded'); // Documented 0.05 threshold

  var floodFeatures = floodFraction.addBands(binaryFlooded).addBands(floodAreaM2.rename('flood_area_m2'));

  // 4. Export Tasks
  Export.image.toDrive({
    image: floodFeatures,
    description: 'vishnupuri_s1_flood_' + ev.name + '_500m',
    folder: 'Vishnupuri_Flood_Spatial',
    fileNamePrefix: 'vishnupuri_s1_flood_' + ev.name + '_500m',
    region: aoi,
    scale: GRID_SCALE,
    crs: TARGET_PROJECTION,
    maxPixels: 1e9
  });

  // Visualization
  Map.centerObject(aoi, 11);
  Map.addLayer(dem, {min: 340, max: 450, palette: ['blue', 'green', 'yellow', 'brown']}, 'DEM (m)', false);
  Map.addLayer(permanentWater.updateMask(permanentWater), {palette: ['0000FF']}, 'Permanent Water (JRC)', false);
  Map.addLayer(binaryFlooded.updateMask(binaryFlooded), {palette: ['FF0000']}, 'Observed Flooded (S1 500m: ' + ev.name + ')', true);
});
