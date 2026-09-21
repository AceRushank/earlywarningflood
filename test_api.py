import sys, urllib.request, json

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

r = urllib.request.urlopen('http://localhost:8000/wards/risk')
data = json.loads(r.read())

print(f"Total wards: {len(data)}")
alerted = [w for w in data if w['alert']]
print(f"Alerted wards: {len(alerted)}")

# Check all required fields are present
required = ['ward_name','bmc_ward_code','lat','lon','historical_flood_score',
            'predicted_susceptibility','next_6h_rainfall_mm','peak_hourly_mm',
            'risk_score','alert','alert_channels']
for i, w in enumerate(data):
    for field in required:
        assert field in w, f'Ward {i} ({w.get("bmc_ward_code","?")}) missing: {field}'

print("Schema validation PASSED - all", len(required), "fields present in all", len(data), "wards")

print("\nTop 5 by risk_score:")
print(f"  {'Ward':<5}  {'risk':>6}  {'hist':>6}  {'susc':>6}  {'6h_rain':>8}  alert")
for w in data[:5]:
    print(f"  {w['bmc_ward_code']:<5}  {w['risk_score']:>6.4f}  "
          f"{w['historical_flood_score']:>6.3f}  "
          f"{w['predicted_susceptibility']:>6.3f}  "
          f"{w['next_6h_rainfall_mm']:>8.1f}  {w['alert']}")

print("\nFirst alerted ward's alert_channels:")
for aw in data:
    if aw['alert_channels']:
        ac = aw['alert_channels']
        print(f"  Ward: {aw['bmc_ward_code']}")
        print(f"  SMS    : {ac['sms_draft']}")
        print(f"  Push   : {ac['push_notification']['title']} | {ac['push_notification']['body'][:70]}")
        print(f"  Dashboard: {ac['dashboard_entry'][:100]}...")
        break

print("\nG/S (no hist data) entry:")
gs = next(w for w in data if w['bmc_ward_code'] == 'G/S')
print(f"  hist_score: {gs['historical_flood_score']}")
print(f"  pred_susc : {gs['predicted_susceptibility']}")
print(f"  risk_score: {gs['risk_score']}")
print(f"  alert     : {gs['alert']}")
