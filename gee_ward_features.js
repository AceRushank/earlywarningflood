/**
 * gee_ward_features.js
 *
 * Google Earth Engine script to compute ward-level geographic features
 * for all 24 BMC wards in Mumbai, exported as ward_geographic_features.csv.
 *
 * HOW TO RUN:
 *   1. Open https://code.earthengine.google.com/
 *   2. Paste this entire script into the Code Editor
 *   3. Upload BMC_admin_wards.geojson as a GEE Asset (Assets → New → Shape files)
 *      OR use the inline ward geometries at the bottom of this file.
 *   4. Click Run → check the Tasks tab → click RUN next to the export task.
 *   5. The CSV will land in your Google Drive as ward_geographic_features.csv
 *
 * DATA SOURCES:
 *   DEM / Elevation:
 *     HydroSHEDS 03VFDEM — Void-filled DEM at 3 arc-second (~90 m) resolution.
 *     WWF / USGS HydroSHEDS. https://www.hydrosheds.org/
 *     GEE ID: "WWF/HydroSHEDS/03VFDEM"
 *
 *   Land Cover:
 *     ESA WorldCover 10 m v200 (2021).
 *     Zanaga, D. et al. (2022). doi:10.5281/zenodo.7254221
 *     https://esa-worldcover.org/en
 *     GEE ID: "ESA/WorldCover/v200"
 *     Class codes used:
 *       10 = Tree cover, 20 = Shrubland, 30 = Grassland, 40 = Cropland
 *       50 = Built-up, 80 = Permanent water bodies
 *
 *   Permanent Water / Distance to Water:
 *     JRC Global Surface Water Seasonality dataset.
 *     Pekel, J.-F. et al. (2016). Nature, 540, 418–422.
 *     GEE ID: "JRC/GSW1_4/MonthlyHistory" (using seasonality >= 10 months/year
 *     as a proxy for permanent water).
 *
 * OUTPUT COLUMNS:
 *   bmc_ward_code      — official BMC ward identifier (from GeoJSON "name" field)
 *   elevation_mean     — mean HydroSHEDS DEM elevation (m) within ward polygon
 *   slope_mean         — mean slope (degrees) derived from the DEM via ee.Terrain.slope
 *   builtup_pct        — % of ward area classified as built-up (WorldCover class 50)
 *   water_pct          — % of ward area classified as water (WorldCover class 80)
 *   vegetation_pct     — % of ward area classified as tree/shrub/grass/cropland
 *                        (WorldCover classes 10+20+30+40 combined)
 *   dist_to_water_mean — mean Euclidean distance (m) to nearest permanent-water
 *                        pixel (JRC seasonality >= 10 months), computed via
 *                        fastDistanceTransform on a 30 m grid
 */

// ── 1. Ward boundary layer ─────────────────────────────────────────────────
// Replace the string below with your uploaded Asset path, e.g.:
//   var wards = ee.FeatureCollection("users/YOUR_USERNAME/BMC_admin_wards");
// Or use the public-repo path if shared:
var wards = ee.FeatureCollection(
  "projects/earthengine-legacy/assets/users/YOUR_USERNAME/BMC_admin_wards"
);

// Study-area bounding box (tight around Mumbai)
var mumbai = ee.Geometry.Rectangle([72.75, 18.85, 73.05, 19.32]);

// ── 2. DEM + Slope ──────────────────────────────────────────────────────────
var dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(mumbai);
var slope = ee.Terrain.slope(dem);  // degrees

// ── 3. ESA WorldCover v200 (2021) ───────────────────────────────────────────
var worldcover = ee.ImageCollection("ESA/WorldCover/v200")
  .first()
  .clip(mumbai);

// Reclassify into binary masks for each land-cover group
var builtup    = worldcover.eq(50);                             // class 50
var water      = worldcover.eq(80);                             // class 80
var vegetation = worldcover.eq(10)                              // Tree cover
  .or(worldcover.eq(20))                                        // Shrubland
  .or(worldcover.eq(30))                                        // Grassland
  .or(worldcover.eq(40));                                       // Cropland

// ── 4. Distance to permanent water (JRC) ────────────────────────────────────
// JRC Monthly History: seasonality = number of months with water per year.
// We define "permanent water" as >= 10 months/year.
// fastDistanceTransform returns pixel distance in # of pixels; multiply by
// native resolution (~30 m) to convert to metres.
var jrc_seasonality = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
  .select("seasonality")
  .clip(mumbai);

var permanent_water_mask = jrc_seasonality.gte(10);             // binary mask
var dist_to_water_pixels = permanent_water_mask
  .Not()                                                        // distance FROM non-water
  .fastDistanceTransform({neighborhood: 512, units: "pixels"})
  .sqrt();                                                      // Euclidean distance in pixels

// JRC native resolution ~30 m → multiply by 30 to get metres
var dist_to_water_m = dist_to_water_pixels.multiply(30).rename("dist_to_water_m");

// ── 5. Per-ward feature extraction ──────────────────────────────────────────
var SCALE = 30; // metres — WorldCover/JRC native; HydroSHEDS ~90 m but 30 m is fine

function extractFeatures(ward) {
  var geom = ward.geometry();
  var code = ward.get("name");   // BMC ward code field in the GeoJSON

  // Helper: mean of an image within the ward polygon
  function wardMean(img, bandName) {
    return img.rename(bandName)
      .reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geom,
        scale: SCALE,
        maxPixels: 1e8,
        bestEffort: true
      })
      .get(bandName);
  }

  // Helper: fraction of pixels where mask == 1 (gives % as 0-100)
  function wardPct(binaryImg, bandName) {
    var frac = binaryImg.rename(bandName)
      .reduceRegion({
        reducer: ee.Reducer.mean(),   // mean of 0/1 = fraction
        geometry: geom,
        scale: SCALE,
        maxPixels: 1e8,
        bestEffort: true
      })
      .get(bandName);
    return ee.Number(frac).multiply(100); // convert fraction → percentage
  }

  return ward.set({
    "bmc_ward_code":      code,
    "elevation_mean":     wardMean(dem,            "elevation"),
    "slope_mean":         wardMean(slope,           "slope"),
    "builtup_pct":        wardPct (builtup,         "builtup"),
    "water_pct":          wardPct (water,           "water"),
    "vegetation_pct":     wardPct (vegetation,      "vegetation"),
    "dist_to_water_mean": wardMean(dist_to_water_m, "dist_to_water_m")
  });
}

var ward_features = wards.map(extractFeatures);

// ── 6. Print a preview to the Console ───────────────────────────────────────
print("Ward feature sample (first 5):", ward_features.limit(5));

// ── 7. Export to Google Drive as CSV ────────────────────────────────────────
Export.table.toDrive({
  collection: ward_features,
  description: "ward_geographic_features",
  fileNamePrefix: "ward_geographic_features",
  fileFormat: "CSV",
  selectors: [
    "bmc_ward_code",
    "elevation_mean",
    "slope_mean",
    "builtup_pct",
    "water_pct",
    "vegetation_pct",
    "dist_to_water_mean"
  ]
});

// ── 8. Optional: visualise in the Map panel ─────────────────────────────────
Map.centerObject(mumbai, 11);
Map.addLayer(dem,        {min: 0,  max: 60,  palette: ["#f7fbff","#2171b5"]}, "Elevation (m)");
Map.addLayer(slope,      {min: 0,  max: 10,  palette: ["#ffffcc","#800026"]}, "Slope (degrees)");
Map.addLayer(builtup.selfMask(), {palette: ["#E2462C"]},                       "Built-up");
Map.addLayer(water.selfMask(),   {palette: ["#419BDF"]},                       "Water (WC80)");
Map.addLayer(permanent_water_mask.selfMask(), {palette: ["#08519c"]},          "Perm. Water (JRC >= 10 mo)");
Map.addLayer(wards, {color: "white", fillColor: "00000000", width: 1.5},       "BMC Wards");
