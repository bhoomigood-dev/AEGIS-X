import os
from dotenv import load_dotenv
from google import genai

# Load the .env file
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_incident_response(
    incident="Possible road incident",
    severity="HIGH",
    risk_score=70,
    vehicle_count=5,
    stopped_count=1,
    blocked_road="B-C",
    alternative_route="A-E-D",
):
    prompt = f"""
You are the AEGIS-X Incident Commander.

Generate a concise emergency intelligence response using the
following system-generated information.

Incident: {incident}
Severity: {severity}
Risk Score: {risk_score}/100
Vehicles detected: {vehicle_count}
Stopped vehicles: {stopped_count}
Blocked road: {blocked_road}
Alternative route: {alternative_route}

Return:
1. Incident summary
2. Why it was flagged
3. Recommended immediate action
4. Route recommendation

Do not invent sensor readings, locations, casualties, or facts
that are not provided.
Do not make medical or safety-critical decisions.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":

    print("AEGIS-X AI INCIDENT COMMANDER")
    print("-----------------------------")

    result = generate_incident_response()

    print(result)