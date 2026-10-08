import os
import time
import threading
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
DEMO_VIDEO = os.path.join("demo", "road.mp4")

VEHICLES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
}


# ============================================================
# GEMINI KEY
# ============================================================

def get_gemini_key():
    """Read Gemini key from Streamlit Cloud Secrets or .env."""

    try:
        key = st.secrets.get("GEMINI_API_KEY")

        if key:
            return key

    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


# ============================================================
# YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SHARED STATE
# ============================================================

class AEGISState:

    def __init__(self):

        self.lock = threading.Lock()

        self.history = defaultdict(
            lambda: deque(maxlen=8)
        )

        self.overlap_frames = defaultdict(int)

        # Live camera values
        self.vehicle_count = 0
        self.stopped_count = 0
        self.risk = 0
        self.severity = "LOW"
        self.status = "NORMAL"
        self.reason = (
            "No abnormal traffic behaviour detected"
        )

        # Demo-video result
        self.demo_result = None
        self.demo_frame = None


if "aegis_state" not in st.session_state:

    st.session_state.aegis_state = AEGISState()


state = st.session_state.aegis_state


# ============================================================
# GEOMETRY
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

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    intersection = width * height

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    vehicle_count,
    stopped_count,
    collision_detected=False,
):

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

    if risk >= 70:

        severity = "HIGH"

    elif risk >= 40:

        severity = "MEDIUM"

    else:

        severity = "LOW"

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

    return (
        risk,
        severity,
        status,
        reason,
    )


# ============================================================
# LIVE CAMERA CALLBACK
# ============================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

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
    # Read tracked vehicles
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
    # Stopped vehicles
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
                    positions[i]
                )

            if movement < 25:

                stopped_count += 1

    # --------------------------------------------------------
    # Persistent overlap
    # --------------------------------------------------------

    collision_detected = False

    for i in range(
        len(vehicles)
    ):

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
                vehicles[j]["box"]
            )

            if overlap > 0.15:

                state.overlap_frames[pair] += 1

            else:

                state.overlap_frames[pair] = 0

            if state.overlap_frames[pair] >= 5:

                collision_detected = True

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    (
        risk,
        severity,
        status,
        reason,
    ) = calculate_risk(
        vehicle_count,
        stopped_count,
        collision_detected,
    )

    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    with state.lock:

        state.vehicle_count = vehicle_count
        state.stopped_count = stopped_count
        state.risk = risk
        state.severity = severity
        state.status = status
        state.reason = reason

    # --------------------------------------------------------
    # Overlay
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
# DEMO VIDEO ANALYSIS
# ============================================================

def run_demo_analysis(
    controlled_incident=False,
):

    if not os.path.exists(DEMO_VIDEO):

        return {
            "success": False,
            "error": f"Video not found: {DEMO_VIDEO}",
        }

    cap = cv2.VideoCapture(
        DEMO_VIDEO
    )

    if not cap.isOpened():

        return {
            "success": False,
            "error": "Could not open demo video.",
        }

    # Local tracking state for this run
    history = defaultdict(
        lambda: deque(maxlen=8)
    )

    overlap_frames = defaultdict(int)

    unique_ids = set()

    max_vehicle_count = 0
    max_stopped_count = 0

    collision_detected = False

    frames_processed = 0

    best_risk = -1
    best_frame = None
    best_reason = ""

    max_frames = 300

    preview = st.empty()
    progress = st.progress(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames_processed += 1

        if frames_processed > max_frames:
            break

        try:

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.35,
                verbose=False,
            )

        except Exception as error:

            cap.release()

            return {
                "success": False,
                "error": str(error),
            }

        result = results[0]

        output = result.plot()

        vehicles = []

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

                unique_ids.add(
                    track_id
                )

                centre = center_of(box)

                history[track_id].append(
                    centre
                )

                vehicles.append(
                    {
                        "id": track_id,
                        "box": box,
                    }
                )

        vehicle_count = len(vehicles)

        max_vehicle_count = max(
            max_vehicle_count,
            vehicle_count
        )

        # ----------------------------------------------------
        # Stopped vehicles
        # ----------------------------------------------------

        stopped_count = 0

        for vehicle in vehicles:

            positions = history[
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
                        positions[i]
                    )

                if movement < 25:

                    stopped_count += 1

        max_stopped_count = max(
            max_stopped_count,
            stopped_count
        )

        # ----------------------------------------------------
        # Overlap
        # ----------------------------------------------------

        for i in range(
            len(vehicles)
        ):

            for j in range(
                i + 1,
                len(vehicles)
            ):

                pair = tuple(
                    sorted(
                        (
                            vehicles[i]["id"],
                            vehicles[j]["id"],
                        )
                    )
                )

                overlap = calculate_iou(
                    vehicles[i]["box"],
                    vehicles[j]["box"]
                )

                if overlap > 0.15:

                    overlap_frames[pair] += 1

                else:

                    overlap_frames[pair] = 0

                if overlap_frames[pair] >= 5:

                    collision_detected = True

        # ----------------------------------------------------
        # Risk for current frame
        # ----------------------------------------------------

        (
            frame_risk,
            frame_severity,
            frame_status,
            frame_reason,
        ) = calculate_risk(
            vehicle_count,
            stopped_count,
            collision_detected,
        )

        if frame_risk > best_risk:

            best_risk = frame_risk
            best_frame = output.copy()
            best_reason = frame_reason

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        if frames_processed % 5 == 0:

            preview.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB,
                ),
                caption=(
                    f"Processing frame "
                    f"{frames_processed}/{max_frames}"
                ),
                use_container_width=True,
            )

        progress.progress(
            min(
                frames_processed / max_frames,
                1.0,
            )
        )

    cap.release()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    (
        risk,
        severity,
        status,
        reason,
    ) = calculate_risk(
        max_vehicle_count,
        max_stopped_count,
        collision_detected,
    )

    # --------------------------------------------------------
    # Controlled incident demonstration
    # --------------------------------------------------------

    if controlled_incident:

        risk = max(
            risk,
            70,
        )

        severity = "HIGH"

        status = "CONTROLLED INCIDENT DEMO"

        if reason:
            reason += "; "

        reason += (
            "Controlled prototype incident scenario"
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result_data = {
        "success": True,
        "frames": frames_processed,
        "unique_vehicles": len(unique_ids),
        "max_vehicles": max_vehicle_count,
        "max_stopped": max_stopped_count,
        "collision": collision_detected,
        "risk": risk,
        "severity": severity,
        "status": status,
        "reason": reason,
        "controlled_incident": controlled_incident,
    }

    with state.lock:

        state.demo_result = result_data
        state.vehicle_count = max_vehicle_count
        state.stopped_count = max_stopped_count
        state.risk = risk
        state.severity = severity
        state.status = status
        state.reason = reason
        state.demo_frame = best_frame

    return result_data


# ============================================================
# ROUTE MAP
# ============================================================

def build_route_map(
    incident_active=True,
):

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

    if incident_active:

        blocked_road = ("B", "C")

        graph.remove_edge(
            blocked_road[0],
            blocked_road[1],
        )

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

    else:

        blocked_road = None
        alternative_route = normal_route
        alternative_distance = normal_distance

    fig = go.Figure()

    # Base roads
    for u, v, _ in roads:

        if (
            incident_active
            and
            {u, v} == set(blocked_road)
        ):
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

    # Normal route
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
                line=dict(width=7),
                name="Normal Route",
                hoverinfo="skip",
            )
        )

    # Incident road
    if incident_active:

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

        # Alternative route
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

    # Nodes
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
            visible=False,
            showgrid=False,
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
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
    )


# ============================================================
# GEMINI INCIDENT REPORT
# ============================================================

def generate_ai_report():

    api_key = get_gemini_key()

    if not api_key:

        return (
            None,
            "GEMINI_API_KEY not found."
        )

    with state.lock:

        status = state.status
        risk = state.risk
        severity = state.severity
        vehicle_count = state.vehicle_count
        stopped_count = state.stopped_count
        reason = state.reason
        demo_result = state.demo_result

    incident_active = (
        status in (
            "POSSIBLE INCIDENT",
            "WARNING",
            "CONTROLLED INCIDENT DEMO",
        )
    )

    if incident_active:

        blocked_road = "B-C"
        alternative_route = "A-E-D"

    else:

        blocked_road = "None"
        alternative_route = "No rerouting required"

    scenario_note = ""

    if demo_result and demo_result.get(
        "controlled_incident"
    ):

        scenario_note = (
            "This is a controlled prototype incident "
            "demonstration, not a verified real accident."
        )

    prompt = f"""
You are the AEGIS-X Incident Commander.

System-generated information:

Status: {status}
Risk Score: {risk}/100
Severity: {severity}
Vehicles Detected: {vehicle_count}
Stopped Vehicles: {stopped_count}
Reason: {reason}
Blocked Road: {blocked_road}
Alternative Route: {alternative_route}

{scenario_note}

Generate a concise incident intelligence report with:

1. Incident Summary
2. Why It Was Flagged
3. Recommended Immediate Action
4. Route Recommendation

Rules:
- Use only the supplied information.
- Do not invent casualties.
- Do not invent locations.
- Do not invent sensor readings.
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

                    if (
                        "503" in str(error)
                        or
                        "UNAVAILABLE" in str(error)
                    ):

                        time.sleep(
                            2 ** attempt
                        )

                    else:

                        break

        return (
            None,
            f"Gemini unavailable: {last_error}",
        )

    except Exception as error:

        return (
            None,
            str(error),
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
    "AEGIS-X combines computer vision, incident "
    "intelligence, risk assessment, geographic "
    "routing and Generative AI response."
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

        result = run_edge_case(
            edge_case
        )

        st.sidebar.markdown(
            f"**Test:** {result['title']}"
        )

        if result["level"] == "ERROR":

            st.sidebar.error(
                result["message"]
            )

        elif result["level"] == "WARNING":

            st.sidebar.warning(
                result["message"]
            )

        else:

            st.sidebar.info(
                result["message"]
            )

        st.sidebar.write(
            f"**Safe Action:** "
            f"{result['action']}"
        )

        st.sidebar.success(
            f"**System Status:** "
            f"{result['status']}"
        )


# ============================================================
# MAIN TABS
# ============================================================

tab_live, tab_demo, tab_map, tab_ai = st.tabs(
    [
        "🎥 Live Detection",
        "🎞️ Demo Road Video",
        "🗺️ Geographic Intelligence",
        "🤖 AI Commander",
    ]
)


# ============================================================
# TAB 1 — LIVE CAMERA
# ============================================================

with tab_live:

    st.header(
        "Live AI Incident Detection"
    )

    st.write(
        "Use the browser camera for live YOLO26n "
        "and ByteTrack processing."
    )

    st.warning(
        "For the public deployment, camera connectivity "
        "depends on the browser/network WebRTC connection. "
        "Use Demo Road Video as the reliable backup."
    )

    RTC_CONFIGURATION = {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }

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
                vehicle_count,
            )

        with col2:

            st.metric(
                "Stopped",
                stopped_count,
            )

        with col3:

            st.metric(
                "Risk Score",
                f"{risk}/100",
            )

        with col4:

            st.metric(
                "Severity",
                severity,
            )

        if status == "POSSIBLE INCIDENT":

            st.error(
                f"🚨 {status}"
            )

        elif status == "WARNING":

            st.warning(
                f"⚠️ {status}"
            )

        elif status == "PROCESSING ERROR":

            st.error(
                f"❌ {status}"
            )

        else:

            st.success(
                f"✅ AEGIS-X STATUS: {status}"
            )

        st.write(
            f"**Reason:** {reason}"
        )

    show_live_metrics()


# ============================================================
# TAB 2 — DEMO ROAD VIDEO
# ============================================================

with tab_demo:

    st.header(
        "🎞️ Demo Road Video"
    )

    st.write(
        "Reliable backup input for the public prototype. "
        "The video is processed on the AEGIS-X server using "
        "YOLO26n and ByteTrack."
    )

    if os.path.exists(DEMO_VIDEO):

        st.video(
            DEMO_VIDEO
        )

        demo_mode = st.radio(
            "Analysis Mode",
            [
                "Automatic Analysis",
                "Controlled Incident Demo",
            ],
            horizontal=True,
        )

        st.caption(
            "Controlled Incident Demo is a simulated "
            "scenario for demonstrating the downstream "
            "route and AI response. It is not claimed "
            "as a real accident detection."
        )

        if st.button(
            "▶ Run AEGIS-X Video Analysis",
            type="primary",
        ):

            controlled = (
                demo_mode
                == "Controlled Incident Demo"
            )

            with st.spinner(
                "Running YOLO26n + ByteTrack analysis..."
            ):

                result = run_demo_analysis(
                    controlled_incident=controlled
                )

            if result["success"]:

                st.success(
                    "AEGIS-X video analysis completed."
                )

            else:

                st.error(
                    result["error"]
                )

        # ----------------------------------------------------
        # Show last result
        # ----------------------------------------------------

        with state.lock:

            result = state.demo_result
            demo_frame = state.demo_frame

        if result:

            st.markdown("---")

            st.subheader(
                "Analysis Result"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Frames",
                    result["frames"],
                )

            with col2:

                st.metric(
                    "Unique Vehicles",
                    result["unique_vehicles"],
                )

            with col3:

                st.metric(
                    "Risk Score",
                    f"{result['risk']}/100",
                )

            with col4:

                st.metric(
                    "Severity",
                    result["severity"],
                )

            if result["controlled_incident"]:

                st.warning(
                    "Controlled incident scenario active."
                )

            elif result["status"] == "POSSIBLE INCIDENT":

                st.error(
                    "Possible incident indicated by "
                    "the prototype rule engine."
                )

            elif result["status"] == "WARNING":

                st.warning(
                    "Traffic warning indicated."
                )

            else:

                st.success(
                    "No abnormal incident condition "
                    "indicated by the prototype rules."
                )

            st.write(
                f"**Reason:** {result['reason']}"
            )

            if demo_frame is not None:

                st.subheader(
                    "AI Detection Snapshot"
                )

                st.image(
                    cv2.cvtColor(
                        demo_frame,
                        cv2.COLOR_BGR2RGB,
                    ),
                    use_container_width=True,
                )

    else:

        st.error(
            f"Demo video not found: {DEMO_VIDEO}"
        )


# ============================================================
# TAB 3 — GEOGRAPHIC INTELLIGENCE
# ============================================================

with tab_map:

    st.header(
        "🗺️ Geographic Intelligence"
    )

    with state.lock:

        current_status = state.status

    incident_active = (
        current_status in (
            "POSSIBLE INCIDENT",
            "WARNING",
            "CONTROLLED INCIDENT DEMO",
        )
    )

    (
        figure,
        normal_route,
        normal_distance,
        alternative_route,
        alternative_distance,
    ) = build_route_map(
        incident_active=incident_active
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    if incident_active:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Affected Road",
                "B-C",
            )

        with col2:

            st.metric(
                "Normal Route",
                "A-B-C-D",
            )

        with col3:

            st.metric(
                "Alternative Route",
                "A-E-D",
            )

        st.success(
            "Incident condition → B-C blocked → "
            "traffic diverted through A → E → D."
        )

    else:

        st.info(
            "No active incident. Normal route remains available."
        )

        st.write(
            "**Normal Route:** "
            + " → ".join(normal_route)
        )

    st.caption(
        "Prototype digital road graph; not live GPS navigation."
    )


# ============================================================
# TAB 4 — AI COMMANDER
# ============================================================

with tab_ai:

    st.header(
        "🤖 Generative AI Incident Commander"
    )

    st.write(
        "Gemini converts AEGIS-X system-generated "
        "information into an explainable response."
    )

    with state.lock:

        current_status = state.status
        current_risk = state.risk
        current_severity = state.severity
        current_vehicle_count = state.vehicle_count
        current_stopped_count = state.stopped_count
        current_reason = state.reason

    st.info(
        f"Current State: {current_status} | "
        f"Risk: {current_risk}/100 | "
        f"Severity: {current_severity}"
    )

    incident_data = (
        f"Status: {current_status}\n"
        f"Risk Score: {current_risk}/100\n"
        f"Severity: {current_severity}\n"
        f"Vehicles Detected: {current_vehicle_count}\n"
        f"Stopped Vehicles: {current_stopped_count}\n"
        f"Reason: {current_reason}\n"
        f"Blocked Road: "
        f"{'B-C' if current_status in ('POSSIBLE INCIDENT', 'WARNING', 'CONTROLLED INCIDENT DEMO') else 'None'}\n"
        f"Alternative Route: "
        f"{'A-E-D' if current_status in ('POSSIBLE INCIDENT', 'WARNING', 'CONTROLLED INCIDENT DEMO') else 'No rerouting required'}"
    )

    st.text_area(
        "System-generated Incident Data",
        value=incident_data,
        height=220,
        disabled=True,
    )

    if st.button(
        "🤖 Generate AI Incident Report",
        type="primary",
    ):

        with st.spinner(
            "Generating incident intelligence..."
        ):

            report, error = generate_ai_report()

        if report:

            st.success(
                "AI Incident Report Generated"
            )

            st.markdown(
                report
            )

        else:

            st.error(
                error
            )


# ============================================================
# TASK 6 FLOW
# ============================================================

st.markdown("---")

st.subheader(
    "🔄 Review 3 — Task 6 End-to-End Flow"
)

st.markdown(
    """
**INPUT**

🎥 Live Camera **or** 🎞️ `road.mp4`

↓

**PROCESSING**

YOLO26n + ByteTrack

↓

**DECISION**

Vehicle behaviour → Risk Score → Severity

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
# TASK 5 SUMMARY
# ============================================================

st.markdown("---")

st.subheader(
    "🛡️ Review 3 — Task 5: Edge-Case Handling"
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
        "Continue monitoring normally."
    )

with col3:

    st.info(
        "**Gemini Failure**\n\n"
        "Retry and preserve core processing."
    )

with col4:

    st.info(
        "**Invalid Input**\n\n"
        "Reject safely."
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
    "AEGIS-X | AI Emergency & Geographic Intelligence System | "
    "Detect. Understand. Respond."
)

st.caption(
    "Prototype note: risk and routing rules are deterministic "
    "prototype logic. Gemini provides explanation and response "
    "text. Controlled incident mode is explicitly simulated."
)