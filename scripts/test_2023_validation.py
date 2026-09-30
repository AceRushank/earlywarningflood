import xarray as xr
import numpy as np
import pandas as pd

nc_path = 'data/historical/raw_nc/ind2023_rfp25.nc'
ds = xr.open_dataset(nc_path)

print('=== 1. FILE & DATASET INFO ===')
print('Path:', nc_path)
print('Dimensions:', dict(ds.sizes))
print('Data variables:', list(ds.data_vars.keys()))

print('\n=== 2. COORDINATE ORIENTATION ===')
lats = ds['LATITUDE'].values
lons = ds['LONGITUDE'].values
times = ds['TIME'].values

print(f'LATITUDE: length={len(lats)}, min={lats.min()}, max={lats.max()}, step={lats[1]-lats[0]:.4f}')
print(f'Is LATITUDE strictly increasing (South to North)? {bool(np.all(np.diff(lats) > 0))}')
print(f'LONGITUDE: length={len(lons)}, min={lons.min()}, max={lons.max()}, step={lons[1]-lons[0]:.4f}')
print(f'Is LONGITUDE strictly increasing (West to East)? {bool(np.all(np.diff(lons) > 0))}')
t0_str = pd.to_datetime(times[0]).strftime('%Y-%m-%d')
t1_str = pd.to_datetime(times[-1]).strftime('%Y-%m-%d')
print(f'TIME: length={len(times)}, start={t0_str}, end={t1_str}')
print(f'Is TIME strictly daily increasing? {bool(np.all(np.diff(times).astype("timedelta64[D]").astype(int) == 1))}')

print('\n=== 3. SELECTED FOUR GRID COORDINATES ===')
selected_coords = [
    (19.00, 72.75),
    (19.00, 73.00),
    (19.25, 72.75),
    (19.25, 73.00)
]
for lat, lon in selected_coords:
    assert lat in lats, f'Lat {lat} not in dataset!'
    assert lon in lons, f'Lon {lon} not in dataset!'
    lat_idx = int(np.where(lats == lat)[0][0])
    lon_idx = int(np.where(lons == lon)[0][0])
    print(f'Cell ({lat:.2f}°N, {lon:.2f}°E) -> lat_index={lat_idx}, lon_index={lon_idx}')

print('\n=== 4 & 5. EXTRACTION FOR KNOWN EVENT DATE: 2023-07-20 ===')
target_date_str = '2023-07-20'
target_time = pd.Timestamp(target_date_str)
time_idx = int(np.where(pd.to_datetime(times).normalize() == target_time)[0][0])
print(f'Target date: {target_date_str} (time_index={time_idx})')

raw_rf = ds['RAINFALL'].values # shape: (TIME, LATITUDE, LONGITUDE)

cell_values = {}
for lat, lon in selected_coords:
    lat_idx = int(np.where(lats == lat)[0][0])
    lon_idx = int(np.where(lons == lon)[0][0])
    val = float(raw_rf[time_idx, lat_idx, lon_idx])
    col_name = f'rainfall_{lat:.2f}_{lon:.2f}_mm'
    cell_values[col_name] = val
    print(f'  {col_name}: {val:.4f} mm')

mean_val = float(np.mean(list(cell_values.values())))
print(f'  rainfall_mumbai_mean_mm: {mean_val:.4f} mm')

print('\n=== 6. UNITS & METADATA VERIFICATION ===')
rf_attrs = ds['RAINFALL'].attrs
print('RAINFALL variable attributes:', rf_attrs)
print('Reported units:', rf_attrs.get('units', 'UNSPECIFIED'))
print('Long name:', rf_attrs.get('long_name', 'UNSPECIFIED'))

print('\n=== 7. DIRECT VERIFICATION AGAINST RAW NETCDF ===')
# Cross-check using xarray coordinate selection
sel_ds = ds['RAINFALL'].sel(TIME=target_date_str)
for lat, lon in selected_coords:
    val_sel = float(sel_ds.sel(LATITUDE=lat, LONGITUDE=lon).values)
    col_name = f'rainfall_{lat:.2f}_{lon:.2f}_mm'
    expected = cell_values[col_name]
    diff = abs(val_sel - expected)
    print(f'  Cross-check {col_name}: sel={val_sel:.4f}, indexed={expected:.4f}, diff={diff:.1e}')
    assert diff < 1e-5, f'Discrepancy at {lat}, {lon}!'

print('\nAll 2023 validation checks PASSED successfully!')
