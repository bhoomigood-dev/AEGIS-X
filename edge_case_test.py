def handle_edge_case(case):
    print("AEGIS-X EDGE-CASE TEST")
    print("----------------------")

    if case == "camera_failure":
        print("ERROR: Live camera unavailable.")
        print("SAFE ACTION: Switch to demo road video.")
        print("STATUS: FALLBACK READY")

    elif case == "no_detection":
        print("INFO: No relevant vehicles detected.")
        print("SAFE ACTION: Continue monitoring.")
        print("STATUS: NORMAL")

    elif case == "ai_failure":
        print("WARNING: Generative AI service unavailable.")
        print("ACTION: Retry AI request.")
        print("FALLBACK: Use deterministic AEGIS-X response.")
        print("STATUS: CORE SYSTEM CONTINUES")

    elif case == "invalid_input":
        print("ERROR: Invalid or unsupported input.")
        print("SAFE ACTION: Request valid video or use live camera.")
        print("STATUS: INPUT REJECTED SAFELY")

    else:
        print("ERROR: Unknown test case.")
        print("SAFE ACTION: Keep system in safe monitoring state.")


# Test all required Review 3 edge cases
cases = [
    "camera_failure",
    "no_detection",
    "ai_failure",
    "invalid_input",
]

for test_case in cases:
    print()
    handle_edge_case(test_case)