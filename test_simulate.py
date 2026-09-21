import sys
import urllib.request
import json
import urllib.parse

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def run_test(simulate_rain_mm):
    url = f'http://localhost:8000/wards/risk?simulate_rain_mm={simulate_rain_mm}'
    r = urllib.request.urlopen(url)
    data = json.loads(r.read())
    
    alerted = [w for w in data if w['alert']]
    return len(alerted), alerted

if __name__ == "__main__":
    rainfall_levels = [5, 40, 150, 250]
    
    print("=" * 60)
    print(f"{'Rainfall (mm)':<15} | {'Old Alerts (Count)':<20} | {'New Alerts (Count)'}")
    print("-" * 60)
    
    results = {}
    for val in rainfall_levels:
        new_count, alerted_wards = run_test(val)
        # Old logic was top 75th percentile, which means top 25% of 24 wards = exactly 6 wards alerted, ALWAYS.
        old_count = 6 
        print(f"{val:<15} | {old_count:<20} | {new_count}")
        results[val] = alerted_wards
        
    print("=" * 60)
    
    print("\n--- Validation at 150mm (Very Heavy) ---")
    alerted_150 = results[150]
    kurla_ward = next((w for w in alerted_150 if w['bmc_ward_code'] == 'L'), None)
    if kurla_ward:
        print("PASS: Ward L (Kurla) is correctly alerted at 150mm!")
        print(f"Risk Score: {kurla_ward['risk_score']}, Historical Score: {kurla_ward['historical_flood_score']}")
    else:
        print("FAIL: Ward L (Kurla) is NOT alerted at 150mm!")

    print("\n--- Validation at 5mm (Light) ---")
    alerted_5 = results[5]
    if len(alerted_5) == 0:
        print("PASS: No wards alerted at 5mm.")
    else:
        print(f"Alerted wards at 5mm: {len(alerted_5)}")
        
    print("\nSample Alert from Ward L (Kurla) at 150mm:")
    if kurla_ward:
        ac = kurla_ward['alert_channels']
        print(f"  SMS: {ac['sms_draft']}")
        print(f"  Push: {ac['push_notification']['title']} | {ac['push_notification']['body'][:70]}")
