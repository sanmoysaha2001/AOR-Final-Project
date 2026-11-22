"""
Map rendering components for FVTRP.
Supports:
1. Folium Map with 100% Free, No-API-Key Tile Providers (OpenStreetMap, Esri World Street, Esri Satellite, Carto Free CDN)
2. Plotly Interactive Map (open-street-map style, zero external tokens, ultra fast)
3. 2D Euclidean Coordinate Grid Map (exact reproduction of Figures 4 & 5 in the research paper)
"""

import folium
from typing import List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from core.model import FVTRPInstance, Route, FVTRPSolution

ROUTE_COLORS = [
    "#2563EB",  # Blue
    "#DC2626",  # Red
    "#059669",  # Emerald Green
    "#D97706",  # Amber
    "#7C3AED",  # Purple
    "#DB2777",  # Pink
    "#0891B2",  # Cyan
    "#4F46E5",  # Indigo
    "#EA580C",  # Orange
    "#16A34A",  # Green
]

def render_folium_map(
    instance: FVTRPInstance,
    routes: Optional[List[Route]] = None,
    active_route_filter: Optional[int] = None,
    tile_choice: str = "OpenStreetMap"
) -> folium.Map:
    """
    Renders an interactive Folium Map using completely FREE, No-API-Key tile providers.
    """
    depot = instance.depot
    all_lats = [depot.lat] + [d.lat for d in instance.dealers]
    all_lons = [depot.lon] + [d.lon for d in instance.dealers]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Base map with standard OpenStreetMap (NEVER asks for API key)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=None,
        control_scale=True
    )

    # Free public tile layers without any API keys
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OpenStreetMap (Default)",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ",
        name="Esri World Street Map",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Esri Satellite Imagery",
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community",
        name="Esri Topographic Map",
        overlay=False,
        control=True
    ).add_to(m)

    # 1. Depot Marker
    depot_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 200px;">
        <h4 style="margin: 0 0 6px 0; color: #1E3A8A; border-bottom: 2px solid #3B82F6; padding-bottom: 4px;">
            🏢 {depot.name}
        </h4>
        <p style="margin: 3px 0; font-size: 13px;"><b>Role:</b> Regional Vehicle Distribution Center (RVDC)</p>
        <p style="margin: 3px 0; font-size: 13px;"><b>Location:</b> {depot.lat:.4f}, {depot.lon:.4f}</p>
        <p style="margin: 3px 0; font-size: 13px;"><b>Fleet Dispatched:</b> {len(routes) if routes else 0} Transporters</p>
    </div>
    """
    folium.Marker(
        location=[depot.lat, depot.lon],
        popup=folium.Popup(depot_html, max_width=300),
        tooltip=f"RVDC Depot: {depot.name}",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)

    # Map of dealer assignments to routes
    dealer_route_map = {}
    if routes:
        for r_idx, r in enumerate(routes):
            color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
            for stop_idx, did in enumerate(r.dealer_ids):
                dealer_route_map[did] = {
                    "route_id": r.route_id,
                    "sequence": stop_idx + 1,
                    "color": color,
                    "pattern": r.loading_pattern.name,
                    "ratio": r.loading_ratio
                }

    # 2. Plot Dealers
    dealers_fg = folium.FeatureGroup(name="Automobile Dealers")
    for d in instance.dealers:
        assigned = dealer_route_map.get(d.id, None)
        color = assigned["color"] if assigned else "#6B7280"
        seq_text = f"Stop #{assigned['sequence']}" if assigned else "Unassigned"
        route_text = f"Route {assigned['route_id']}" if assigned else "None"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 210px;">
            <h4 style="margin: 0 0 5px 0; color: #111827; border-bottom: 2px solid {color};">
                🚗 {d.name} (ID: {d.id})
            </h4>
            <p style="margin: 3px 0; font-size: 12px;"><b>Assigned:</b> {route_text} ({seq_text})</p>
            <p style="margin: 3px 0; font-size: 12px;"><b>Time Window:</b> [{d.time_window[0]:.0f}, {d.time_window[1]:.0f}] min</p>
            <p style="margin: 3px 0; font-size: 12px;"><b>Handling Time:</b> {d.handling_time:.0f} min</p>
            <hr style="margin: 4px 0; border: 0; border-top: 1px solid #E5E7EB;">
            <p style="margin: 2px 0; font-size: 12px;"><b>Orders:</b> {d.demands[0]} Sedans, {d.demands[1]} SUVs, {d.demands[2]} MPVs</p>
            <p style="margin: 2px 0; font-size: 12px; font-weight: bold; color: {color};">Total: {d.total_demand} vehicles</p>
        </div>
        """

        badge_html = f"""
        <div style="background-color: {color}; color: white; border-radius: 50%; width: 24px; height: 24px;
                    display: flex; align-items: center; justify-content: center; font-weight: bold;
                    font-size: 11px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.4);">
            {assigned['sequence'] if assigned else d.id}
        </div>
        """

        folium.Marker(
            location=[d.lat, d.lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{d.name} | Total: {d.total_demand} cars | {route_text}",
            icon=folium.DivIcon(html=badge_html, icon_size=(24, 24), icon_anchor=(12, 12))
        ).add_to(dealers_fg)
    dealers_fg.add_to(m)

    # 3. Plot Transporter Routes
    if routes:
        for r_idx, r in enumerate(routes):
            if active_route_filter is not None and r.route_id != active_route_filter:
                continue

            color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
            coords = [[step.lat, step.lon] for step in r.steps]

            route_fg = folium.FeatureGroup(name=f"Carrier #{r.route_id} ({r.loading_pattern.name[:16]})")

            # Route polyline
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=5,
                opacity=0.85,
                tooltip=f"Route {r.route_id}: {len(r.dealer_ids)} dealers | {r.travel_time:.0f} min | Loading Ratio: {r.loading_ratio:.1%}"
            ).add_to(route_fg)

            route_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_plotly_map(
    instance: FVTRPInstance,
    routes: Optional[List[Route]] = None,
    active_route_filter: Optional[int] = None
) -> go.Figure:
    """
    Renders an interactive Plotly OpenStreetMap (100% token-free, no API key).
    """
    fig = go.Figure()

    # Map routes
    if routes:
        for r_idx, r in enumerate(routes):
            if active_route_filter is not None and r.route_id != active_route_filter:
                continue

            color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
            lats = [step.lat for step in r.steps]
            lons = [step.lon for step in r.steps]
            hover_texts = [f"Stop #{idx}: {step.node_name} (Arr: {step.arrival_time:.0f}m)" for idx, step in enumerate(r.steps)]

            # Route Line
            fig.add_trace(go.Scattermap(
                lat=lats,
                lon=lons,
                mode="lines+markers",
                name=f"Carrier #{r.route_id} ({r.loading_pattern.name.split('(')[0].strip()})",
                line=dict(width=4, color=color),
                marker=dict(size=8, color=color),
                text=hover_texts,
                hoverinfo="text"
            ))

    # Dealers markers
    dealer_lats = [d.lat for d in instance.dealers]
    dealer_lons = [d.lon for d in instance.dealers]
    dealer_texts = [
        f"<b>{d.name}</b> (ID: {d.id})<br>"
        f"Demand: {d.demands[0]} Sedans, {d.demands[1]} SUVs, {d.demands[2]} MPVs (Total: {d.total_demand})<br>"
        f"Time Window: [{d.time_window[0]:.0f}, {d.time_window[1]:.0f}] min"
        for d in instance.dealers
    ]

    fig.add_trace(go.Scattermap(
        lat=dealer_lats,
        lon=dealer_lons,
        mode="markers+text",
        name="Automobile Dealers",
        marker=dict(size=14, color="#1E3A8A"),
        text=[str(d.id) for d in instance.dealers],
        textposition="top center",
        hovertext=dealer_texts,
        hoverinfo="text"
    ))

    # Depot marker
    fig.add_trace(go.Scattermap(
        lat=[instance.depot.lat],
        lon=[instance.depot.lon],
        mode="markers+text",
        name="RVDC Depot (Hub)",
        marker=dict(size=20, color="#DC2626", symbol="warehouse"),
        text=["RVDC DEPOT"],
        textposition="bottom center",
        hovertext=[f"<b>{instance.depot.name}</b><br>Role: Distribution Center Hub"],
        hoverinfo="text"
    ))

    center_lat = np.mean([instance.depot.lat] + dealer_lats)
    center_lon = np.mean([instance.depot.lon] + dealer_lons)

    fig.update_layout(
        map=dict(
            style="open-street-map",  # Zero API key required
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10.5
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.85)")
    )
    return fig


def render_euclidean_grid(
    instance: FVTRPInstance,
    routes: Optional[List[Route]] = None,
    active_route_filter: Optional[int] = None
) -> go.Figure:
    """
    Renders 2D Planar Coordinate Grid matching Figures 4 and 5 in Hu et al. (2015).
    """
    fig = go.Figure()

    # If instance has grid coordinates, use them; otherwise scale lat/lon
    has_grid = (instance.depot.grid_x is not None and instance.depot.grid_y is not None)

    def get_xy(node):
        if has_grid and getattr(node, "grid_x", None) is not None:
            return node.grid_x, node.grid_y
        # Convert lat/lon to local planar km from depot
        dx = (node.lon - instance.depot.lon) * 111.0 * np.cos(np.radians(instance.depot.lat))
        dy = (node.lat - instance.depot.lat) * 111.0
        return dx, dy

    depot_x, depot_y = get_xy(instance.depot)

    # Plot routes as directed paths
    if routes:
        for r_idx, r in enumerate(routes):
            if active_route_filter is not None and r.route_id != active_route_filter:
                continue

            color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
            xs = [depot_x]
            ys = [depot_y]
            texts = ["Depot Start"]

            for did in r.dealer_ids:
                d = instance.get_dealer_by_id(did)
                x, y = get_xy(d)
                xs.append(x)
                ys.append(y)
                texts.append(f"Dealer {did} ({d.name})")

            xs.append(depot_x)
            ys.append(depot_y)
            texts.append("Depot Return")

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"Carrier #{r.route_id} ({r.loading_pattern.name.split('(')[0].strip()})",
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color),
                text=texts,
                hoverinfo="text"
            ))

    # Dealers scatter
    dealer_xs = [get_xy(d)[0] for d in instance.dealers]
    dealer_ys = [get_xy(d)[1] for d in instance.dealers]
    dealer_labels = [f"<b>D{d.id}</b><br>{d.demands}" for d in instance.dealers]

    fig.add_trace(go.Scatter(
        x=dealer_xs,
        y=dealer_ys,
        mode="markers+text",
        name="Dealers",
        marker=dict(size=14, color="#1E3A8A"),
        text=[f"D{d.id}" for d in instance.dealers],
        textposition="top center",
        hovertext=[f"Dealer {d.id}: {d.name}<br>Demands: {d.demands}" for d in instance.dealers],
        hoverinfo="text"
    ))

    # Depot
    fig.add_trace(go.Scatter(
        x=[depot_x],
        y=[depot_y],
        mode="markers+text",
        name="Depot Hub",
        marker=dict(size=18, color="#DC2626", symbol="square"),
        text=["DEPOT"],
        textposition="bottom center",
        hoverinfo="name"
    ))

    axis_title = "Grid X (500x500 Space)" if has_grid else "Relative Distance X (km)"
    axis_y_title = "Grid Y (500x500 Space)" if has_grid else "Relative Distance Y (km)"

    fig.update_layout(
        title="<b>Planar Coordinate System View (Paper Fig. 4 & Fig. 5)</b>",
        xaxis_title=axis_title,
        yaxis_title=axis_y_title,
        template="plotly_white",
        height=540,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
