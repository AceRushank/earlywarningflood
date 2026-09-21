import sys
sys.path.insert(0, ".")
from app.data_loader import load_ward_data

df = load_ward_data()
print()
print("=== WARD LOADER RESULT ===")
print(df[["ward_name", "bmc_ward_code", "ward_area_km2", "lat", "lon", "historical_flood_score", "inundation_ratio"]].to_string(index=False))
print()
print("Total wards:", len(df))
nonzero = (df["historical_flood_score"] > 0).sum()
zero = (df["historical_flood_score"] == 0).sum()
print("Wards with historical score > 0:", nonzero)
print("Wards with historical score = 0 (Ward G/S):", zero)
assert len(df) == 24, f"Expected 24 wards, got {len(df)}"
assert (df["ward_area_km2"] > 0).all(), "All wards must have positive area"
assert "inundation_ratio" in df.columns, "inundation_ratio column missing"
print("test_loader.py PASSED.")
