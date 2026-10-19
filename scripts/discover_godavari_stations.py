import pandas as pd
import numpy as np

TELEMETRY_CSV = r"c:\Users\aceru\floodsystem\data\rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
print("Loading telemetry metadata...")
df = pd.read_csv(TELEMETRY_CSV, usecols=[
    'Station', 'Agency', 'River', 'Basin', 'Tributary', 'District', 'Latitude', 'Longitude', 
    'Is_DischargeDataAvailable', 'RL_of_zeroGauge', 'MeanSeaLevel'
])
print(f"Total rows: {len(df)}")
godavari_stations = df[df['Basin'].str.contains('Godavari', case=False, na=False)].drop_duplicates(subset=['Station']).copy()
print(f"Total Godavari basin stations in 2026: {len(godavari_stations)}")

# Vishnupuri coordinates: Lat 19.1269, Lon 77.2808
v_lat = 19.1269
v_lon = 77.2808

for idx, r in godavari_stations.iterrows():
    try:
        lat = float(r['Latitude'])
        lon = float(r['Longitude'])
        # Upstream of Vishnupuri on Godavari / tributaries is typically west / north-west (lon < 77.3)
        print(f"STATION: {r['Station']:30s} | River: {str(r['River']):15s} | Trib: {str(r['Tributary']):15s} | Dist: {str(r['District']):12s} | Lat: {lat:7.4f} | Lon: {lon:7.4f} | Q: {r['Is_DischargeDataAvailable']}")
    except Exception as e:
        print(f"STATION: {r['Station']:30s} | Error: {e}")
