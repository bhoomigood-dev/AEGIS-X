import os
import threading
import time
from collections import defaultdict, deque

import av
import cv2
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

from edge_cases import run_edge_case


# ============================================================
# AEGIS-X
# AI Emergency & Geographic Intelligence System
# Detect. Understand. Respond.
# ============================================================

st.set_page_config(
    page_title="AEGIS-X",
    page_icon="🚨",
    layout="wide",
)

load_dotenv()

MODEL_PATH = "yolo26n.pt"

VEHICLES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
}


# ============================================================
# GEMINI API KEY
# ============================================================

def get_gemini_key():
    """
    Read Gemini key from Streamlit Cloud Secrets first.
    Fall back to local .env for local development.
    """

    try:
        key = st.secrets.get("GEMINI_API_KEY")

        if key:
            return key

    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


GEMINI_API_KEY = get_gemini_key()


# ============================================================
# LOAD YOLO26n
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SHARED AEGIS STATE
# ============================================================

class AEGISState:

    def __init__(self):

        self.lock = threading.Lock()

        # Recent positions for tracked vehicles
        self.history = defaultdict(
            lambda: deque(maxlen=8)
        )

        # Persistent overlap between tracked vehicles
        self.overlap_frames = defaultdict(int)

        # Live values
        self.vehicle_count = 0
        self.stopped_count = 0
        self.risk = 0
        self.severity = "LOW"
        self.status = "NORMAL"

        self.reason = (
            "No abnormal traffic behaviour detected"
        )


if "aegis_state" not in st.session_state:

    st.session_state.aegis_state = AEGISState()


state = st.session_state.aegis_state


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def center_of(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def distance(p1, p2):

    return (
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    ) ** 0.5


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    intersection = width * height

    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LIVE CAMERA CALLBACK
# ============================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    # --------------------------------------------------------
    # YOLO + ByteTrack
    # --------------------------------------------------------

    try:

        results = model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            verbose=False,
        )

    except Exception as error:

        with state.lock:

            state.status = "PROCESSING ERROR"

            state.reason = (
                f"Vision processing failed: {error}"
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    result = results[0]

    output = result.plot()

    vehicles = []

    # --------------------------------------------------------
    # READ TRACKED VEHICLES
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        for track_id, class_id, box in zip(
            ids,
            classes,
            boxes,
        ):

            class_name = result.names[class_id]

            if class_name not in VEHICLES:
                continue

            centre = center_of(box)

            state.history[track_id].append(
                centre
            )

            vehicles.append(
                {
                    "id": track_id,
                    "name": class_name,
                    "box": box,
                }
            )

    vehicle_count = len(vehicles)

    # --------------------------------------------------------
    # STOPPED VEHICLE DETECTION
    # --------------------------------------------------------

    stopped_count = 0

    for vehicle in vehicles:

        positions = state.history[
            vehicle["id"]
        ]

        if len(positions) >= 6:

            movement = 0

            for i in range(
                1,
                len(positions)
            ):

                movement += distance(
                    positions[i - 1],
                    positions[i],
                )

            if movement < 25:

                stopped_count += 1

    # --------------------------------------------------------
    # POSSIBLE COLLISION
    # --------------------------------------------------------

    collision_detected = False

    for i in range(len(vehicles)):

        for j in range(
            i + 1,
            len(vehicles)
        ):

            id1 = vehicles[i]["id"]
            id2 = vehicles[j]["id"]

            pair = tuple(
                sorted(
                    (id1, id2)
                )
            )

            overlap = calculate_iou(
                vehicles[i]["box"],
                vehicles[j]["box"],
            )

            if overlap > 0.15:

                state.overlap_frames[pair] += 1

            else:

                state.overlap_frames[pair] = 0

            if state.overlap_frames[pair] >= 5:

                collision_detected = True

    # --------------------------------------------------------
    # RISK ENGINE
    # --------------------------------------------------------

    risk = 0

    reasons = []

    if vehicle_count >= 4:

        risk += 20

        reasons.append(
            "High vehicle density"
        )

    if vehicle_count >= 6:

        risk += 15

        reasons.append(
            "Heavy traffic"
        )

    if stopped_count > 0:

        risk += stopped_count * 10

        reasons.append(
            "Stopped vehicle detected"
        )

    if collision_detected:

        risk += 40

        reasons.append(
            "Persistent vehicle overlap"
        )

    risk = min(
        risk,
        100
    )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    if risk >= 70:

        severity = "HIGH"

    elif risk >= 40:

        severity = "MEDIUM"

    else:

        severity = "LOW"

    # --------------------------------------------------------
    # INCIDENT STATUS
    # --------------------------------------------------------

    if collision_detected:

        status = "POSSIBLE INCIDENT"

    elif risk >= 40:

        status = "WARNING"

    else:

        status = "NORMAL"

    if reasons:

        reason = "; ".join(reasons)

    else:

        reason = (
            "No abnormal traffic behaviour detected"
        )

    # --------------------------------------------------------
    # UPDATE LIVE STATE
    # --------------------------------------------------------

    with state.lock:

        state.vehicle_count = vehicle_count
        state.stopped_count = stopped_count
        state.risk = risk
        state.severity = severity
        state.status = status
        state.reason = reason

    # --------------------------------------------------------
    # DRAW INFORMATION ON VIDEO
    # --------------------------------------------------------

    lines = [
        "AEGIS-X",
        f"Vehicles: {vehicle_count}",
        f"Stopped: {stopped_count}",
        f"Risk: {risk}/100",
        f"Severity: {severity}",
        f"Status: {status}",
    ]

    y = 35

    for index, line in enumerate(lines):

        cv2.putText(
            output,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9 if index == 0 else 0.7,
            (255, 255, 255),
            2,
        )

        y += 32

    return av.VideoFrame.from_ndarray(
        output,
        format="bgr24",
    )


# ============================================================
# ROUTING ENGINE
# ============================================================

def build_route_map():

    graph = nx.Graph()

    roads = [
        ("A", "B", 1),
        ("B", "C", 1),
        ("C", "D", 1),
        ("A", "E", 3),
        ("E", "D", 3),
        ("B", "E", 4),
    ]

    graph.add_weighted_edges_from(
        roads
    )

    positions = {
        "A": (0, 1),
        "B": (1, 2),
        "C": (2, 2),
        "D": (3, 1),
        "E": (1.5, 0),
    }

    source = "A"
    destination = "D"

    # Normal route
    normal_route = nx.shortest_path(
        graph,
        source,
        destination,
        weight="weight",
    )

    normal_distance = nx.shortest_path_length(
        graph,
        source,
        destination,
        weight="weight",
    )

    # Block B-C
    blocked_road = ("B", "C")

    if graph.has_edge(
        blocked_road[0],
        blocked_road[1]
    ):

        graph.remove_edge(
            blocked_road[0],
            blocked_road[1]
        )

    # Alternative route
    alternative_route = nx.shortest_path(
        graph,
        source,
        destination,
        weight="weight",
    )

    alternative_distance = nx.shortest_path_length(
        graph,
        source,
        destination,
        weight="weight",
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # Base roads
    # --------------------------------------------------------

    for u, v, _ in roads:

        if {u, v} == set(blocked_road):
            continue

        fig.add_trace(
            go.Scatter(
                x=[
                    positions[u][0],
                    positions[v][0],
                ],
                y=[
                    positions[u][1],
                    positions[v][1],
                ],
                mode="lines",
                line=dict(width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # Normal route
    # --------------------------------------------------------

    for i in range(
        len(normal_route) - 1
    ):

        u = normal_route[i]
        v = normal_route[i + 1]

        fig.add_trace(
            go.Scatter(
                x=[
                    positions[u][0],
                    positions[v][0],
                ],
                y=[
                    positions[u][1],
                    positions[v][1],
                ],
                mode="lines",
                line=dict(width=8),
                name="Normal Route",
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # Blocked road
    # --------------------------------------------------------

    u, v = blocked_road

    fig.add_trace(
        go.Scatter(
            x=[
                positions[u][0],
                positions[v][0],
            ],
            y=[
                positions[u][1],
                positions[v][1],
            ],
            mode="lines",
            line=dict(
                width=10,
                dash="dash",
            ),
            name="Incident / Blocked Road",
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # Alternative route
    # --------------------------------------------------------

    for i in range(
        len(alternative_route) - 1
    ):

        u = alternative_route[i]
        v = alternative_route[i + 1]

        fig.add_trace(
            go.Scatter(
                x=[
                    positions[u][0],
                    positions[v][0],
                ],
                y=[
                    positions[u][1],
                    positions[v][1],
                ],
                mode="lines",
                line=dict(width=8),
                name="Alternative Route",
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                positions[n][0]
                for n in positions
            ],
            y=[
                positions[n][1]
                for n in positions
            ],
            mode="markers+text",
            text=list(positions.keys()),
            textposition="top center",
            marker=dict(size=22),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="AEGIS-X Geographic Intelligence",
        xaxis=dict(
            showgrid=False,
            visible=False,
        ),
        yaxis=dict(
            showgrid=False,
            visible=False,
        ),
        template="plotly_white",
        height=560,
    )

    return (
        fig,
        normal_route,
        normal_distance,
        alternative_route,
        alternative_distance,
        blocked_road,
    )


# ============================================================
# GEMINI AI INCIDENT REPORT
# ============================================================

def generate_ai_report(
    status,
    risk,
    severity,
    vehicle_count,
    stopped_count,
    reason,
    blocked_road,
    alternative_route,
):

    api_key = get_gemini_key()

    if not api_key:

        return (
            None,
            "GEMINI_API_KEY not found."
        )

    prompt = f"""
You are the AEGIS-X Incident Commander.

The following information was generated by the
AEGIS-X deterministic processing system:

Incident Status: {status}
Risk Score: {risk}/100
Severity: {severity}
Vehicles Detected: {vehicle_count}
Stopped Vehicles: {stopped_count}
Reason: {reason}
Blocked Road: {blocked_road}
Alternative Route: {alternative_route}

Generate a concise incident intelligence report.

Include:

1. Incident Summary
2. Why It Was Flagged
3. Recommended Immediate Action
4. Route Recommendation

Rules:

- Use only the supplied information.
- Do not invent casualties.
- Do not invent locations.
- Do not invent sensor values.
- Do not invent facts.
- Do not make medical decisions.
- Do not make safety-critical decisions.
- Treat the response as decision support.
"""

    try:

        client = genai.Client(
            api_key=api_key
        )

        models = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
        ]

        last_error = None

        for model_name in models:

            for attempt in range(3):

                try:

                    response = (
                        client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                    )

                    return (
                        response.text,
                        None,
                    )

                except Exception as error:

                    last_error = error

                    error_text = str(error)

                    # Retry temporary service errors
                    if (
                        "503" in error_text
                        or
                        "UNAVAILABLE" in error_text
                    ):

                        time.sleep(
                            2 ** attempt
                        )

                    else:

                        break

            # Try next model if first model failed

        return (
            None,
            f"Gemini temporarily unavailable: {last_error}"
        )

    except Exception as error:

        return (
            None,
            str(error)
        )


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🚨 AEGIS-X")

st.subheader(
    "AI Emergency & Geographic Intelligence System"
)

st.caption(
    "Detect. Understand. Respond."
)

st.write(
    "AEGIS-X combines real-time computer vision, "
    "incident intelligence, risk assessment, "
    "geographic routing and Generative AI response."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "AEGIS-X System Status"
)

st.sidebar.success(
    "YOLO26n — READY"
)

st.sidebar.success(
    "ByteTrack — READY"
)

st.sidebar.success(
    "Risk Engine — READY"
)

st.sidebar.success(
    "Severity Engine — READY"
)

st.sidebar.success(
    "Route Engine — READY"
)

if get_gemini_key():

    st.sidebar.success(
        "Gemini AI — READY"
    )

else:

    st.sidebar.error(
        "Gemini AI — API KEY MISSING"
    )

st.sidebar.markdown("---")

st.sidebar.write(
    "Primary Input"
)

st.sidebar.write(
    "🎥 Live Browser Camera"
)

st.sidebar.write(
    "Backup Input"
)

st.sidebar.write(
    "🎞️ demo/road.mp4"
)


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


# ============================================================
# REVIEW 3 — TASK 5
# ============================================================

st.sidebar.markdown("---")

st.sidebar.header(
    "🛡️ Review 3 — Task 5"
)

edge_case = st.sidebar.selectbox(
    "Select Edge-Case",
    [
        "Select test",
        "Camera Failure",
        "No Relevant Detection",
        "Gemini Unavailable",
        "Invalid Input",
    ],
)

if st.sidebar.button(
    "Run Edge-Case Test"
):

    if edge_case == "Select test":

        st.sidebar.warning(
            "Please select an edge case."
        )

    else:

        edge_result = run_edge_case(
            edge_case
        )

        st.sidebar.markdown(
            f"**Test:** {edge_result['title']}"
        )

        if edge_result["level"] == "ERROR":

            st.sidebar.error(
                edge_result["message"]
            )

        elif edge_result["level"] == "WARNING":

            st.sidebar.warning(
                edge_result["message"]
            )

        else:

            st.sidebar.info(
                edge_result["message"]
            )

        st.sidebar.write(
            f"**Safe Action:** "
            f"{edge_result['action']}"
        )

        st.sidebar.success(
            f"**System Status:** "
            f"{edge_result['status']}"
        )


# ============================================================
# MAIN TABS
# ============================================================

tab_live, tab_map, tab_ai = st.tabs(
    [
        "🎥 Live Detection",
        "🗺️ Geographic Intelligence",
        "🤖 AI Commander",
    ]
)


# ============================================================
# TAB 1 — LIVE DETECTION
# ============================================================

with tab_live:

    st.header(
        "Live AI Incident Detection"
    )

    st.write(
        "Start the browser camera. "
        "AEGIS-X detects and tracks vehicles "
        "and calculates a prototype incident risk."
    )

    # --------------------------------------------------------
    # Browser camera
    # --------------------------------------------------------

    webrtc_streamer(
        key="aegis-camera",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Live metrics
    # --------------------------------------------------------

    @st.fragment(run_every="1s")
    def show_live_metrics():

        with state.lock:

            vehicle_count = state.vehicle_count
            stopped_count = state.stopped_count
            risk = state.risk
            severity = state.severity
            status = state.status
            reason = state.reason

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Vehicles",
                vehicle_count
            )

        with col2:

            st.metric(
                "Stopped",
                stopped_count
            )

        with col3:

            st.metric(
                "Risk Score",
                f"{risk}/100"
            )

        with col4:

            st.metric(
                "Severity",
                severity
            )

        if status == "POSSIBLE INCIDENT":

            st.error(
                f"🚨 AEGIS-X STATUS: {status}"
            )

        elif status == "WARNING":

            st.warning(
                f"⚠️ AEGIS-X STATUS: {status}"
            )

        elif status == "PROCESSING ERROR":

            st.error(
                "❌ AEGIS-X PROCESSING ERROR"
            )

        else:

            st.success(
                f"✅ AEGIS-X STATUS: {status}"
            )

        st.write(
            f"**Reason:** {reason}"
        )

    show_live_metrics()

    # --------------------------------------------------------
    # Current decision
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Current Decision"
    )

    with state.lock:

        current_status = state.status
        current_risk = state.risk
        current_severity = state.severity

    if current_status in (
        "POSSIBLE INCIDENT",
        "WARNING",
    ):

        st.warning(
            "Incident intelligence activated."
        )

        st.write(
            "**Affected Road:** B-C"
        )

        st.write(
            "**Recommended Alternative:** "
            "A → E → D"
        )

        st.write(
            f"**Decision:** {current_status} | "
            f"Risk {current_risk}/100 | "
            f"Severity {current_severity}"
        )

    else:

        st.info(
            "No active incident decision. "
            "System continues monitoring."
        )


# ============================================================
# TAB 2 — GEOGRAPHIC INTELLIGENCE
# ============================================================

with tab_map:

    st.header(
        "🗺️ Geographic Intelligence"
    )

    st.write(
        "AEGIS-X identifies an affected road segment "
        "and demonstrates alternative route selection."
    )

    (
        route_figure,
        normal_route,
        normal_distance,
        alternative_route,
        alternative_distance,
        blocked_road,
    ) = build_route_map()

    st.plotly_chart(
        route_figure,
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Affected Road",
            "B-C"
        )

    with col2:

        st.metric(
            "Normal Distance",
            str(normal_distance)
        )

    with col3:

        st.metric(
            "Alternative Distance",
            str(alternative_distance)
        )

    st.write(
        "**Normal Route:** "
        + " → ".join(normal_route)
    )

    st.write(
        "**Alternative Route:** "
        + " → ".join(alternative_route)
    )

    st.success(
        "Incident scenario: B-C is blocked. "
        "Traffic is diverted through A → E → D."
    )

    st.caption(
        "Prototype digital road graph; "
        "not live GPS navigation."
    )


# ============================================================
# TAB 3 — AI INCIDENT COMMANDER
# ============================================================

with tab_ai:

    st.header(
        "🤖 Generative AI Incident Commander"
    )

    st.write(
        "Gemini converts the deterministic AEGIS-X "
        "decision into an explainable response."
    )

    # --------------------------------------------------------
    # Current live state
    # --------------------------------------------------------

    with state.lock:

        live_status = state.status
        live_risk = state.risk
        live_severity = state.severity
        live_vehicle_count = state.vehicle_count
        live_stopped_count = state.stopped_count
        live_reason = state.reason

    st.info(
        f"Live State: {live_status} | "
        f"Risk: {live_risk}/100 | "
        f"Severity: {live_severity}"
    )

    # --------------------------------------------------------
    # System-generated incident data
    # --------------------------------------------------------

    incident_text = (
        f"Incident Status: {live_status}\n"
        f"Risk Score: {live_risk}/100\n"
        f"Severity: {live_severity}\n"
        f"Vehicles Detected: {live_vehicle_count}\n"
        f"Stopped Vehicles: {live_stopped_count}\n"
        f"Reason: {live_reason}\n"
        f"Blocked Road: B-C\n"
        f"Alternative Route: A-E-D"
    )

    st.text_area(
        "System-generated Incident Data",
        value=incident_text,
        height=220,
        disabled=True,
    )

    # --------------------------------------------------------
    # Generate AI report
    # --------------------------------------------------------

    if st.button(
        "🤖 Generate AI Incident Report",
        type="primary",
    ):

        with st.spinner(
            "Generating incident intelligence..."
        ):

            report, error = generate_ai_report(
                status=live_status,
                risk=live_risk,
                severity=live_severity,
                vehicle_count=live_vehicle_count,
                stopped_count=live_stopped_count,
                reason=live_reason,
                blocked_road="B-C",
                alternative_route="A-E-D",
            )

        if report:

            st.success(
                "AI Incident Report Generated"
            )

            st.markdown(
                report
            )

        else:

            st.error(
                f"Gemini unavailable: {error}"
            )

    # --------------------------------------------------------
    # End-to-end architecture
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "AEGIS-X End-to-End Flow"
    )

    st.markdown(
        """
        **INPUT**

        🎥 Live Browser Camera

        ↓

        **PROCESSING**

        YOLO26n + ByteTrack

        ↓

        **DECISION**

        Incident Analysis + Risk + Severity

        ↓

        **GEOGRAPHIC ACTION**

        Affected Road → Alternative Route

        ↓

        **GENERATIVE RESPONSE**

        Gemini Incident Commander

        ↓

        **OUTPUT**

        Explainable Emergency Intelligence Report
        """
    )


# ============================================================
# REVIEW 3 — TASK 5 SUMMARY
# ============================================================

st.markdown("---")

st.subheader(
    "🛡️ Review 3 — Task 5: Edge-Case Handling"
)

st.write(
    "AEGIS-X is designed to respond to abnormal "
    "conditions using fallback, safe-state, retry "
    "and input-validation behaviour."
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.info(
        "**Camera Failure**\n\n"
        "Fallback to demo road video."
    )

with col2:

    st.info(
        "**No Detection**\n\n"
        "Continue monitoring in NORMAL state."
    )

with col3:

    st.info(
        "**AI Failure**\n\n"
        "Retry Gemini and continue core processing."
    )

with col4:

    st.info(
        "**Invalid Input**\n\n"
        "Reject safely and request valid input."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

with state.lock:

    final_status = state.status
    final_risk = state.risk
    final_severity = state.severity

st.caption(
    f"AEGIS-X | Status: {final_status} | "
    f"Risk: {final_risk}/100 | "
    f"Severity: {final_severity}"
)

st.caption(
    "AI Emergency & Geographic Intelligence System | "
    "Detect. Understand. Respond."
)

st.caption(
    "Prototype note: incident and routing decisions "
    "are deterministic prototype logic. Gemini is used "
    "for explanation and response generation."
)