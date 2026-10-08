def calculate_severity(vehicle_count, stopped_count, collision_detected):
    risk = 0
    reasons = []

    if vehicle_count >= 4:
        risk += 20
        reasons.append("High vehicle density")

    if vehicle_count >= 6:
        risk += 15
        reasons.append("Heavy traffic")

    if stopped_count > 0:
        risk += stopped_count * 10
        reasons.append("Stopped vehicle detected")

    if collision_detected:
        risk += 40
        reasons.append("Persistent vehicle overlap")

    risk = min(risk, 100)

    if risk >= 70:
        severity = "HIGH"
    elif risk >= 40:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    if not reasons:
        reasons.append("No abnormal traffic behaviour detected")

    return risk, severity, reasons


# Simple test
if __name__ == "__main__":
    risk, severity, reasons = calculate_severity(
        vehicle_count=5,
        stopped_count=1,
        collision_detected=True
    )

    print("AEGIS-X SEVERITY REPORT")
    print("-----------------------")
    print(f"Risk Score : {risk}/100")
    print(f"Severity   : {severity}")
    print("Reasons:")

    for reason in reasons:
        print("-", reason)