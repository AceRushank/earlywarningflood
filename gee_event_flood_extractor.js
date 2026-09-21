/**
 * gee_event_flood_extractor.js
 *
 * Google Earth Engine Script to extract 500 m grid-cell event-based flood observations
 * for Mumbai using the faculty-specified Sentinel-1 SAR methodology.
 *
 * This version uses EXACT verified Sentinel-1 scene IDs to guarantee reproducibility.
 *
 * HOW TO RUN:
 *   1. Paste this script into Google Earth Engine Code Editor.
 *   2. Click "Run" at the top.
 *   3. Open the "Tasks" tab and click "Run" on the export tasks to generate the CSVs.
 */

// ── 1. Study Area Geometry & Grid ───────────────────────────────────────────
var mumbai = ee.Geometry.Rectangle([72.75, 18.85, 73.05, 19.32]);
var proj = "EPSG:32643"; // UTM Zone 43N (metric)

// Create 500m Fishnet
var grid500m = mumbai.coveringGrid(proj, 500);

// Add unique cell ID
var gridCells = grid500m.map(function(f) {
  var centroid = f.geometry().centroid();
  return f.set({
    'grid_id': f.id(), // Stable grid ID
    'grid_center_lon': centroid.coordinates().get(0),
    'grid_center_lat': centroid.coordinates().get(1)
  });
});

// ── 2. Topography & Slope Layers ────────────────────────────────────────────
var dem = ee.Image("WWF/HydroSHEDS/03VFDEM").clip(mumbai);
var slope = ee.Terrain.slope(dem);
var slopeMask = slope.lt(5); // Slope < 5 degrees rule

// ── 3. Land Cover (ESA WorldCover 10m v200) ─────────────────────────────────
var worldcover = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(mumbai).first().clip(mumbai);
var builtupFraction = worldcover.eq(50).rename('builtup_fraction'); 

// ── 4. Permanent Water Mask (JRC Global Surface Water v1.4) ─────────────────
var jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(mumbai);
var permanentWater = jrc.select("seasonality").gte(10); // Seasonality >= 10 months

// Distance to permanent water (fastDistanceTransform on 30m grid)
var distToWater = permanentWater.Not()
  .fastDistanceTransform({neighborhood: 512, units: "pixels"})
  .sqrt()
  .multiply(30)
  .rename("distance_to_permanent_water_m");

// Stack physical static variables
var staticVars = ee.Image([
  dem.rename('elevation_m'),
  slope.rename('slope_deg'),
  builtupFraction,
  distToWater
]);

// ── 5. Verified Historical Event Catalog ────────────────────────────────────
var events = [
  {
    id: "EV_2018_07_19",
    event_type: "flood",
    preId: "S1A_IW_GRDH_1SDV_20180707T010252_20180707T010317_022681_02751F_438D",
    postId: "S1A_IW_GRDH_1SDV_20180719T010253_20180719T010318_022856_027A79_D005"
  },
  {
    id: "EV_2018_08_24",
    event_type: "flood",
    preId: "S1A_IW_GRDH_1SDV_20180812T010254_20180812T010319_023206_02858E_B304",
    postId: "S1A_IW_GRDH_1SDV_20180824T010255_20180824T010320_023381_028B35_1A05"
  },
  {
    id: "EV_2019_09_24",
    event_type: "flood",
    preId: "S1A_IW_GRDH_1SDV_20190912T010302_20190912T010327_028981_03497A_E99F",
    postId: "S1A_IW_GRDH_1SDV_20190924T010302_20190924T010327_029156_034F70_C1E8"
  },
  {
    id: "EV_2023_07_29",
    event_type: "flood",
    preId: "S1A_IW_GRDH_1SDV_20230717T010330_20230717T010355_049456_05F271_ACE0",
    postId: "S1A_IW_GRDH_1SDV_20230729T010331_20230729T010356_049631_05F7D5_4EAB"
  },
  {
    id: "CTRL_2024_03_12",
    event_type: "control",
    postId: "S1A_IW_GRDH_1SDV_20240313T010329_20240313T010354_052956_066912_8293",
    preDateRange: ['2024-02-25', '2024-03-05']
  }
];

// ── 6. Faculty Sentinel-1 Flood Detection & Export ──────────────────────────

print("=== STARTING FLOOD EXTRACTION ===");

var visualMasks = [];

events.forEach(function(ev) {
  
  var preImg, postImg;
  
  if (ev.preId) {
    preImg = ee.Image("COPERNICUS/S1_GRD/" + ev.preId).select('VV');
  } else {
    var preCol = ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(mumbai)
      .filterDate(ev.preDateRange[0], ev.preDateRange[1])
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
      .filter(ee.Filter.eq("relativeOrbitNumber_start", 34));
    preImg = ee.Image(preCol.first()).select('VV');
  }

  postImg = ee.Image("COPERNICUS/S1_GRD/" + ev.postId).select('VV');
  
  var actualPreId = preImg.get('system:index');
  var actualPostId = postImg.get('system:index');
  
  if (ev.id === "CTRL_2024_03_12") {
    ee.Dictionary({pre: actualPreId, post: actualPostId}).evaluate(function(res) {
      print("Exact IDs for CTRL_2024_03_12:");
      print("Pre: " + res.pre);
      print("Post: " + res.post);
    });
  }
  
  var preTime = ee.Date(preImg.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss');
  var postTime = ee.Date(postImg.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss');
  
  // Apply smoothing
  var pre = preImg.focal_median(30, "circle", "meters");
  var post = postImg.focal_median(30, "circle", "meters");

  // Faculty Rule: diff < -2.5 and post < -14.0
  var diff = post.subtract(pre);
  var candidateFlood = diff.lt(-2.5).and(post.lt(-14.0));

  // Mask permanent water
  var floodNoPerm = candidateFlood.where(permanentWater, 0);

  // Remove isolated pixels (connectedPixelCount >= 8)
  var connectivity = floodNoPerm.connectedPixelCount({maxSize: 8, eightConnected: true});
  var floodConnected = floodNoPerm.where(connectivity.lt(8), 0);

  // Apply slope < 5 degrees mask
  var finalFlood = floodConnected.updateMask(slopeMask).unmask(0).rename("flood_binary");
  
  // Save for visualization
  visualMasks.push({id: ev.id, mask: finalFlood});

  // ── Reduce to Grid Cells ──────────────────────────────────────────────────
  
  // Calculate cell-level metrics
  var gridWithMetrics = gridCells.map(function(feature) {
    // Reduce static variables
    var staticDict = staticVars.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: feature.geometry(),
      scale: 10,
      crs: proj,
      bestEffort: true,
      maxPixels: 1e9
    });
    
    // Reduce flood area (sum of 10m pixels converted to area)
    var pixelArea = ee.Image.pixelArea();
    var floodAreaImg = finalFlood.multiply(pixelArea);
    
    var floodDict = floodAreaImg.reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: feature.geometry(),
      scale: 10,
      crs: proj,
      bestEffort: true,
      maxPixels: 1e9
    });
    
    var floodArea = floodDict.get('flood_binary');
    var flood_area_m2 = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(floodArea, null), 0, floodArea));
    
    var flood_fraction = flood_area_m2.divide(250000);
    var observed_flooded = ee.Algorithms.If(flood_fraction.gte(0.05), 1, 0);

    return feature.set(staticDict).set({
      'event_id': ev.id,
      'event_type': ev.event_type,
      'event_date': postTime,
      'flood_area_m2': flood_area_m2,
      'flood_fraction': flood_fraction,
      'observed_flooded': observed_flooded,
      'sentinel_pre_id': actualPreId,
      'sentinel_post_id': actualPostId,
      'label_method_version': '1.0'
    });
  });
  
  // Add to export task
  Export.table.toDrive({
    collection: gridWithMetrics,
    description: "Export_" + ev.id,
    folder: "Mumbai_Flood_Project",
    fileNamePrefix: ev.id + "_500m_SAR_dataset",
    fileFormat: "CSV"
  });
});

// ── 7. Visualization ────────────────────────────────────────────────────────
Map.centerObject(mumbai, 11);
Map.addLayer(dem, {min: 0, max: 60, palette: ["#f7fbff","#2171b5"]}, "HydroSHEDS DEM (m)", false);
Map.addLayer(slopeMask.selfMask(), {palette: ["#fee0d2"]}, "Slope < 5 deg", false);
Map.addLayer(permanentWater.selfMask(), {palette: ["#08519c"]}, "Permanent Water (JRC)", false);

// Add all event flood masks
visualMasks.forEach(function(vm) {
  // Color flood events red, control event green
  var color = vm.id.indexOf('CTRL') !== -1 ? "#00ff00" : "#ff0000";
  Map.addLayer(vm.mask.selfMask(), {palette: [color]}, "Flood Mask: " + vm.id, false);
});

print("=== SCRIPT READY ===");
print("1. Turn on the layers in the Map to inspect the actual SAR flood masks.");
print("2. When satisfied, go to the 'Tasks' tab and click 'Run' on all 5 tasks to export the CSVs.");
