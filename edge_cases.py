def run_edge_case(case):

    if case == "Camera Failure":

        return {
            "title": "Camera Failure",
            "level": "ERROR",
            "message": "Live camera input is unavailable.",
            "action": "Switch to demo road video.",
            "status": "FALLBACK READY",
        }

    elif case == "No Relevant Detection":

        return {
            "title": "No Relevant Detection",
            "level": "INFO",
            "message": "No relevant vehicles or incident indicators detected.",
            "action": "Continue monitoring.",
            "status": "SAFE / NORMAL",
        }

    elif case == "Gemini Unavailable":

        return {
            "title": "Generative AI Service Failure",
            "level": "WARNING",
            "message": "Gemini service is temporarily unavailable.",
            "action": "Retry request and continue with deterministic AEGIS-X response.",
            "status": "CORE SYSTEM CONTINUES",
        }

    elif case == "Invalid Input":

        return {
            "title": "Invalid Input",
            "level": "ERROR",
            "message": "The supplied input is invalid or unsupported.",
            "action": "Reject input and request a valid camera/video source.",
            "status": "SAFE REJECTION",
        }

    return {
        "title": "Unknown Condition",
        "level": "ERROR",
        "message": "Unknown system condition.",
        "action": "Keep AEGIS-X in safe monitoring mode.",
        "status": "SAFE STATE",
    }