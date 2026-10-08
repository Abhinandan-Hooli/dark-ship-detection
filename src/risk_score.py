def calculate_risk(status, distance_km, matching_score):
    if status == "Dark Ship":
        return 90
    elif status == "AIS Blackout":
        return 60
    elif distance_km <= 5:
        return 20
    else:
        return 75