"""
Domain models and data structures for Finished-Vehicle Transporter Routing Problem (FVTRP).
Based on: "Finished-vehicle transporter routing problem solved by loading pattern discovery"
(Hu et al., Annals of Operations Research, 2015).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

VEHICLE_TYPE_NAMES = ["Type 1 (Sedan)", "Type 2 (SUV)", "Type 3 (Commercial / MPV)"]

@dataclass
class Dealer:
    id: int
    name: str
    lat: float
    lon: float
    demands: List[int]  # [d_o1, d_o2, d_o3]
    time_window: Tuple[float, float] = (1.0, 10000.0)  # [L_i, U_i] in minutes
    handling_time: float = 2.0  # P_i in minutes
    grid_x: Optional[float] = None  # Original 500x500 grid x if applicable
    grid_y: Optional[float] = None  # Original 500x500 grid y if applicable

    @property
    def total_demand(self) -> int:
        return sum(self.demands)


@dataclass
class Depot:
    id: int = 1
    name: str = "Regional VDC (Anting Auto Hub)"
    lat: float = 31.2989
    lon: float = 121.1685
    time_window: Tuple[float, float] = (0.0, 10000.0)
    handling_time: float = 0.0
    grid_x: Optional[float] = 250.0
    grid_y: Optional[float] = 250.0


@dataclass
class LoadingPattern:
    id: int
    name: str
    capacities: List[int]  # [Theta_e1, Theta_e2, Theta_e3]
    description: str = ""

    @property
    def total_capacity(self) -> int:
        return sum(self.capacities)


@dataclass
class RouteStep:
    node_id: int
    node_name: str
    lat: float
    lon: float
    arrival_time: float
    wait_time: float
    service_start: float
    departure_time: float
    delivered_demands: List[int]


@dataclass
class Route:
    route_id: int
    transporter_id: int
    loading_pattern: LoadingPattern
    dealer_ids: List[int]
    steps: List[RouteStep]
    loaded_demands: List[int]  # [sum_d1, sum_d2, sum_d3]
    loading_ratio: float
    travel_time: float
    total_duration: float
    empty_slots: int
    is_feasible: bool = True
    violation_notes: str = ""


@dataclass
class FVTRPInstance:
    name: str
    city_name: str
    depot: Depot
    dealers: List[Dealer]
    loading_patterns: List[LoadingPattern]
    time_matrix: Optional[np.ndarray] = None  # Shape (N_nodes, N_nodes)
    max_route_duration: float = 1440.0  # P_c parameter (e.g. 24 hours / 1440 mins)
    speed_kmh: float = 50.0

    def get_dealer_by_id(self, d_id: int) -> Optional[Dealer]:
        for d in self.dealers:
            if d.id == d_id:
                return d
        return None

    def compute_distance_matrix_if_needed(self):
        """Computes Euclidean or Haversine distance matrix if not explicitly supplied."""
        nodes = [self.depot] + self.dealers
        n = len(nodes)
        
        # Check if we should use grid coordinates or geographic coordinates
        has_grid = all(node.grid_x is not None and node.grid_y is not None for node in nodes)
        mat = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = 0.0
                elif has_grid:
                    dx = nodes[i].grid_x - nodes[j].grid_x
                    dy = nodes[i].grid_y - nodes[j].grid_y
                    dist = np.hypot(dx, dy)
                    # Convert distance to minutes
                    mat[i, j] = round(dist, 1)
                else:
                    # Haversine formula
                    lat1, lon1 = np.radians(nodes[i].lat), np.radians(nodes[i].lon)
                    lat2, lon2 = np.radians(nodes[j].lat), np.radians(nodes[j].lon)
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    km = 6371.0 * c
                    time_min = (km / self.speed_kmh) * 60.0
                    mat[i, j] = round(time_min, 1)
        self.time_matrix = mat


@dataclass
class FVTRPSolution:
    instance_name: str
    routes: List[Route]
    total_travel_time: float
    total_vehicles_delivered: int
    total_capacity_deployed: int
    overall_loading_ratio: float
    computation_time_sec: float
    algorithm_name: str
    convergence_history: List[Dict[str, float]] = field(default_factory=list)
