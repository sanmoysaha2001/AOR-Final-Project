"""
Plotly charts and visual components for FVTRP.
Includes:
- Stowage deck utilization diagrams (Upper & Lower decks car allocation)
- GA Convergence curves (Figure 6 in paper: Optimal vs Average fitness)
- Route timetable Gantt chart
- Vehicle type demand breakdown
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from core.model import Route, FVTRPSolution

VEHICLE_COLORS = {
    "Type 1 (Sedan)": "#3B82F6",       # Blue
    "Type 2 (SUV)": "#10B981",         # Green
    "Type 3 (MPV/Van)": "#F59E0B",     # Amber
    "Empty Slot": "#E5E7EB"            # Gray
}

def plot_convergence_curve(history: List[Dict[str, float]]) -> go.Figure:
    """
    Renders the Evolutionary Algorithm convergence curve
    matching Figure 6 in Hu et al. (2015).
    """
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No evolution history available", showarrow=False)
        return fig

    df = pd.DataFrame(history)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["generation"],
        y=df["best_fitness"],
        mode="lines",
        name="Optimal (Best)",
        line=dict(color="#2563EB", width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=df["generation"],
        y=df["average_fitness"],
        mode="lines",
        name="Population Average",
        line=dict(color="#9CA3AF", width=1.5, dash="dash")
    ))

    fig.update_layout(
        title="<b>Algorithm 1 Convergence (Generations vs Fitness Value)</b>",
        xaxis_title="Generation Iteration (Pg)",
        yaxis_title="Objective Cost (f / g)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_stowage_breakdown(routes: List[Route]) -> go.Figure:
    """
    Visualizes vehicle loading vs capacity per route (Sedans, SUVs, MPVs, and Empty slots).
    Matches Table 5 stowage analysis in the paper.
    """
    if not routes:
        fig = go.Figure()
        return fig

    route_labels = [f"Route {r.route_id} ({r.loading_pattern.name[:14]})" for r in routes]
    t1_loaded = [r.loaded_demands[0] for r in routes]
    t2_loaded = [r.loaded_demands[1] for r in routes]
    t3_loaded = [r.loaded_demands[2] for r in routes]
    empty_slots = [r.empty_slots for r in routes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Type 1 (Sedans)",
        x=route_labels,
        y=t1_loaded,
        marker_color=VEHICLE_COLORS["Type 1 (Sedan)"]
    ))
    fig.add_trace(go.Bar(
        name="Type 2 (SUVs)",
        x=route_labels,
        y=t2_loaded,
        marker_color=VEHICLE_COLORS["Type 2 (SUV)"]
    ))
    fig.add_trace(go.Bar(
        name="Type 3 (MPV/Van)",
        x=route_labels,
        y=t3_loaded,
        marker_color=VEHICLE_COLORS["Type 3 (MPV/Van)"]
    ))
    fig.add_trace(go.Bar(
        name="Empty / Unused Capacity",
        x=route_labels,
        y=empty_slots,
        marker_color=VEHICLE_COLORS["Empty Slot"],
        marker_pattern_shape="/"
    ))

    fig.update_layout(
        barmode="stack",
        title="<b>Transporter Stowage & Vehicle Allocation per Route (Paper Table 5)</b>",
        xaxis_title="Scheduled Transporter Route",
        yaxis_title="Loaded Finished Vehicles (Units)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_transporter_deck_schematic(route: Route) -> go.Figure:
    """
    Renders an intuitive schematic representation of a 2-deck car transporter
    showing which vehicles occupy the upper deck vs lower deck.
    """
    fig = go.Figure()

    pat = route.loading_pattern
    demands = route.loaded_demands
    total_cap = pat.total_capacity
    
    # Generate list of slots
    slots = []
    for _ in range(demands[0]):
        slots.append(("Type 1 (Sedan)", VEHICLE_COLORS["Type 1 (Sedan)"]))
    for _ in range(demands[1]):
        slots.append(("Type 2 (SUV)", VEHICLE_COLORS["Type 2 (SUV)"]))
    for _ in range(demands[2]):
        slots.append(("Type 3 (MPV/Van)", VEHICLE_COLORS["Type 3 (MPV/Van)"]))
    while len(slots) < total_cap:
        slots.append(("Empty Slot", "#E5E7EB"))

    # Divide slots between Upper Deck and Lower Deck
    half = (total_cap + 1) // 2
    upper_deck = slots[:half]
    lower_deck = slots[half:]

    # Plot upper deck
    for idx, (vtype, color) in enumerate(upper_deck):
        fig.add_shape(
            type="rect",
            x0=idx, y0=1.2, x1=idx + 0.85, y1=1.9,
            fillcolor=color,
            line=dict(color="#374151", width=1.5)
        )
        fig.add_annotation(
            x=idx + 0.42, y=1.55,
            text=vtype.split()[0] if "Empty" not in vtype else "EMPTY",
            showarrow=False,
            font=dict(color="white" if "Empty" not in vtype else "#6B7280", size=10, family="Arial Black")
        )

    # Plot lower deck
    for idx, (vtype, color) in enumerate(lower_deck):
        fig.add_shape(
            type="rect",
            x0=idx, y0=0.1, x1=idx + 0.85, y1=0.8,
            fillcolor=color,
            line=dict(color="#374151", width=1.5)
        )
        fig.add_annotation(
            x=idx + 0.42, y=0.45,
            text=vtype.split()[0] if "Empty" not in vtype else "EMPTY",
            showarrow=False,
            font=dict(color="white" if "Empty" not in vtype else "#6B7280", size=10, family="Arial Black")
        )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, half + 0.5])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=True,
                     tickvals=[0.45, 1.55], ticktext=["<b>Lower Deck</b>", "<b>Upper Deck</b>"], range=[-0.2, 2.2])

    fig.update_layout(
        title=f"<b>Transporter #{route.transporter_id} Deck Stowage Plan ({pat.name}) — Loading Ratio: {route.loading_ratio:.1%}</b>",
        template="plotly_white",
        height=260,
        margin=dict(l=80, r=20, t=50, b=20)
    )
    return fig


def plot_timetable_gantt(routes: List[Route]) -> go.Figure:
    """
    Renders an operational dispatch Gantt timeline showing transit and service times.
    Converts minute offsets into datetime timestamps for clean Plotly rendering.
    """
    if not routes:
        return go.Figure()

    base_time = datetime(2026, 1, 1, 8, 0, 0)
    df_records = []

    for r in routes:
        r_name = f"Transporter {r.transporter_id}"
        for s_idx in range(len(r.steps) - 1):
            s_curr = r.steps[s_idx]
            s_next = r.steps[s_idx + 1]

            # Travel leg
            start_dt = base_time + timedelta(minutes=float(s_curr.departure_time))
            end_dt = base_time + timedelta(minutes=float(s_next.arrival_time))
            if end_dt > start_dt:
                df_records.append({
                    "Transporter": r_name,
                    "Activity": f"Transit to {s_next.node_name}",
                    "Start": start_dt,
                    "Finish": end_dt,
                    "Type": "Transit"
                })

            # Handling / Service
            if s_next.node_name != r.steps[-1].node_name:
                serv_start = base_time + timedelta(minutes=float(s_next.service_start))
                serv_end = base_time + timedelta(minutes=float(s_next.departure_time))
                if serv_end > serv_start:
                    df_records.append({
                        "Transporter": r_name,
                        "Activity": f"Unloading at {s_next.node_name}",
                        "Start": serv_start,
                        "Finish": serv_end,
                        "Type": "Unloading"
                    })

    if not df_records:
        return go.Figure()

    df = pd.DataFrame(df_records)
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Transporter",
        color="Type",
        color_discrete_map={"Transit": "#2563EB", "Unloading": "#10B981"},
        hover_data=["Activity"],
        title="<b>Transporter Dispatch Schedule & Service Timetable</b>"
    )

    fig.update_layout(
        xaxis_title="Time of Day (HH:MM)",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
def plot_heuristic_progress(history: List[Dict]) -> go.Figure:
    """
    Plots the step-by-step cost reduction achieved by heuristics.
    """
    if not history or len(history) <= 1:
        fig = go.Figure()
        fig.add_annotation(text="Initial solution already at local optimum", showarrow=False)
        return fig

    steps = [h["step"] for h in history]
    costs = [h["travel_time"] for h in history]
    actions = [f"Step {h['step']}: {h['action']}" for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=costs,
        mode="lines+markers",
        name="Travel Time",
        line=dict(color="#10B981", width=3),
        marker=dict(size=9, color="#059669"),
        text=actions,
        hoverinfo="text+y"
    ))

    initial_c = costs[0]
    final_c = costs[-1]
    saved = initial_c - final_c

    fig.update_layout(
        title=f"<b>Heuristic Optimization Trajectory (Total Time Saved: {saved:.1f} mins)</b>",
        xaxis_title="Heuristic Iteration / Local Search Step",
        yaxis_title="Total Travel Time (Minutes)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_before_after_comparison(
    init_routes: List[Route],
    ref_routes: List[Route]
) -> go.Figure:
    """
    Renders a side-by-side bar chart comparing initial vs heuristic-refined metrics per transporter.
    """
    fig = go.Figure()

    max_len = max(len(init_routes), len(ref_routes))
    labels = [f"Route #{i+1}" for i in range(max_len)]

    init_times = [init_routes[i].travel_time if i < len(init_routes) else 0 for i in range(max_len)]
    ref_times = [ref_routes[i].travel_time if i < len(ref_routes) else 0 for i in range(max_len)]

    fig.add_trace(go.Bar(
        name="Initial Solution",
        x=labels,
        y=init_times,
        marker_color="#94A3B8"
    ))

    fig.add_trace(go.Bar(
        name="After Heuristic Optimization",
        x=labels,
        y=ref_times,
        marker_color="#2563EB"
    ))

    fig.update_layout(
        barmode="group",
        title="<b>Travel Time Comparison per Transporter (Initial vs After Heuristics)</b>",
        xaxis_title="Transporter Route",
        yaxis_title="Travel Time (Minutes)",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
