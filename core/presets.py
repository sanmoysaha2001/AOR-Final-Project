"""
Preset benchmark datasets and scenario generators for FVTRP.
Includes:
- Exact Table 1, 2, 3 Paper Benchmark (Hu et al. 2015)
- Shanghai Real-world Dealership Logistics Instance
- Metropolitan 20-Dealer Fleet Distribution Instance
"""

import numpy as np
from typing import List, Tuple
from core.model import Dealer, Depot, LoadingPattern, FVTRPInstance

def get_paper_benchmark_patterns() -> List[LoadingPattern]:
    """Loading patterns from Table 3 in the paper."""
    patterns_data = [
        (1, "Pattern 1 (Balanced Deck)", [6, 5, 4], "Standard 2-deck car carrier"),
        (2, "Pattern 2 (Compact Carrier)", [5, 4, 4], "Short chassis transporter"),
        (3, "Pattern 3 (SUV Dominant)", [4, 6, 5], "High-roof upper deck"),
        (4, "Pattern 4 (SUV/Sedan Mix)", [5, 6, 3], "Hydraulic ramp layout A"),
        (5, "Pattern 5 (MPV/Van Deck)", [4, 5, 6], "Heavy duty axle carrier"),
        (6, "Pattern 6 (Multi-SUV Deck)", [3, 7, 4], "Extended upper deck"),
        (7, "Pattern 7 (High Capacity Sedan)", [7, 4, 5], "High capacity 16-car transporter"),
        (8, "Pattern 8 (Sedan/MPV Balanced)", [6, 4, 5], "Variable angle deck"),
        (9, "Pattern 9 (Sedan Intensive)", [5, 7, 2], "Low clearance car carrier")
    ]
    return [
        LoadingPattern(id=pid, name=name, capacities=cap, description=desc)
        for pid, name, cap, desc in patterns_data
    ]

def get_paper_benchmark_time_matrix() -> np.ndarray:
    """Exact Table 2 time matrix from paper (12x12).
    Index 0 is Depot Start (Node 1), Indices 1-10 are Dealers 2-11, Index 11 is Depot Return (Node 12).
    """
    return np.array([
        [0, 141, 195, 206, 94, 176, 156, 160, 243, 211, 103, 0],
        [141, 0, 188, 342, 74, 138, 262, 267, 339, 350, 233, 141],
        [195, 188, 0, 370, 132, 54, 174, 353, 438, 362, 288, 195],
        [206, 342, 370, 0, 300, 370, 227, 118, 153, 30, 109, 206],
        [94, 74, 132, 300, 0, 95, 189, 248, 328, 302, 197, 94],
        [176, 138, 54, 370, 95, 0, 200, 336, 419, 367, 277, 176],
        [156, 262, 174, 227, 189, 200, 0, 260, 340, 211, 187, 156],
        [160, 267, 353, 118, 248, 336, 260, 0, 85, 144, 73, 160],
        [243, 339, 438, 153, 328, 419, 340, 85, 0, 183, 156, 243],
        [211, 350, 362, 30, 302, 367, 211, 144, 183, 0, 122, 211],
        [103, 233, 288, 109, 197, 277, 187, 73, 156, 122, 0, 103],
        [0, 141, 195, 206, 94, 176, 156, 160, 243, 211, 103, 0]
    ], dtype=float)

def get_paper_benchmark_instance() -> FVTRPInstance:
    """Returns the exact 10-dealer instance from Table 1, 2, 3 of Hu et al. (2015).
    Depot is mapped to Anting Auto Logistics Hub, Shanghai.
    The (x, y) 500x500 coordinates from the paper are mapped to geographic coordinates
    around Shanghai Metropolitan Area.
    """
    depot_lat, depot_lon = 31.2989, 121.1685  # Anting Regional VDC
    
    # Scale 500x500 box into ~0.6 deg lat/lon around Shanghai
    def grid_to_geo(gx: float, gy: float) -> Tuple[float, float]:
        # y corresponds to North (lat), x corresponds to East (lon)
        lat = depot_lat + (gy - 250.0) * 0.0016
        lon = depot_lon + (gx - 250.0) * 0.0022
        return round(lat, 5), round(lon, 5)

    depot = Depot(
        id=1,
        name="Anji RVDC Hub (Anting Depot)",
        lat=depot_lat,
        lon=depot_lon,
        time_window=(0.0, 10000.0),
        handling_time=2.0,
        grid_x=250.0,
        grid_y=250.0
    )

    dealers_meta = [
        # id, name, gx, gy, L, U, P, demands [Type 1, Type 2, Type 3]
        (2, "Dealer 2 (Pudong Jinqiao Center)", 388, 222, 1, 10000, 3, [1, 1, 1]),
        (3, "Dealer 3 (Baoshan Terminal)", 364, 408, 1, 10000, 2, [2, 2, 0]),
        (4, "Dealer 4 (Qingpu Industrial Park)", 46, 219, 1, 10000, 2, [2, 0, 0]),
        (5, "Dealer 5 (Minhang Qibao Auto Plaza)", 340, 278, 1, 10000, 3, [2, 0, 0]),
        (6, "Dealer 6 (Yangpu Wujiaochang Auto)", 388, 360, 1, 10000, 2, [1, 1, 1]),
        (7, "Dealer 7 (Jiading North Hub)", 191, 394, 1, 10000, 3, [0, 1, 2]),
        (8, "Dealer 8 (Songjiang New Town)", 134, 140, 1, 10000, 3, [2, 0, 0]),
        (9, "Dealer 9 (Jinshan Chemical Industry Bay)", 85, 71, 1, 10000, 2, [0, 2, 1]),
        (10, "Dealer 10 (Qingpu Dianshan Lake)", 39, 248, 1, 10000, 2, [2, 1, 1]),
        (11, "Dealer 11 (Hongqiao Transport Hub)", 155, 210, 1, 10000, 2, [1, 1, 0])
    ]

    dealers = []
    for did, name, gx, gy, L, U, P, d in dealers_meta:
        lat, lon = grid_to_geo(gx, gy)
        dealers.append(Dealer(
            id=did,
            name=name,
            lat=lat,
            lon=lon,
            demands=d,
            time_window=(float(L), float(U)),
            handling_time=float(P),
            grid_x=float(gx),
            grid_y=float(gy)
        ))

    patterns = get_paper_benchmark_patterns()
    time_mat = get_paper_benchmark_time_matrix()

    return FVTRPInstance(
        name="Paper Benchmark 10-Dealers (Hu et al. 2015)",
        city_name="Shanghai, China (Anji Logistics RVDC)",
        depot=depot,
        dealers=dealers,
        loading_patterns=patterns,
        time_matrix=time_mat,
        max_route_duration=1440.0
    )


def get_shanghai_fleet_instance() -> FVTRPInstance:
    """A realistic 15-dealer automotive distribution scenario across Shanghai."""
    depot = Depot(
        id=1,
        name="Shanghai Anji Logistics RVDC (Anting)",
        lat=31.2950,
        lon=121.1620,
        time_window=(0.0, 1440.0),
        handling_time=0.0
    )

    real_dealers = [
        Dealer(2, "SAIC Volkswagen 4S (Jiading)", 31.3780, 121.2500, [2, 1, 1], (60.0, 300.0), 15.0),
        Dealer(3, "Tesla Giga Center (Pudong Lingang)", 30.8750, 121.9280, [3, 2, 2], (120.0, 480.0), 20.0),
        Dealer(4, "BYD Flagship Showroom (Minhang)", 31.1120, 121.3810, [1, 2, 1], (90.0, 360.0), 15.0),
        Dealer(5, "NIO Delivery Center (Songjiang)", 31.0320, 121.2280, [2, 0, 1], (60.0, 240.0), 10.0),
        Dealer(6, "BMW Shanghai Mall (Xuhui)", 31.1880, 121.4360, [1, 1, 2], (90.0, 420.0), 15.0),
        Dealer(7, "Mercedes Auto Park (Baoshan)", 31.3950, 121.4880, [2, 2, 0], (120.0, 480.0), 15.0),
        Dealer(8, "Audi Distribution Terminal (Yangpu)", 31.2980, 121.5450, [1, 2, 2], (150.0, 540.0), 20.0),
        Dealer(9, "Geely Zeekr Center (Pudong Jinqiao)", 31.2590, 121.6150, [3, 1, 1], (180.0, 600.0), 15.0),
        Dealer(10, "XPeng Auto Delivery (Qingpu)", 31.1490, 121.1250, [2, 1, 0], (60.0, 300.0), 10.0),
        Dealer(11, "Changan Auto Hub (Fengxian)", 30.9180, 121.4680, [2, 0, 2], (120.0, 480.0), 15.0),
        Dealer(12, "Toyota Auto Plaza (Putuo)", 31.2480, 121.3920, [1, 1, 1], (90.0, 360.0), 12.0),
        Dealer(13, "Honda East Terminal (Jinshan)", 30.7420, 121.3390, [0, 2, 2], (180.0, 600.0), 20.0),
        Dealer(14, "Li Auto Superstore (Jing'an)", 31.2290, 121.4480, [2, 2, 1], (120.0, 420.0), 15.0),
        Dealer(15, "Volvo Car Mall (Hongqiao)", 31.1960, 121.3250, [1, 1, 0], (60.0, 240.0), 10.0),
        Dealer(16, "GAC Aion Terminal (Chongming)", 31.6210, 121.4020, [1, 2, 1], (180.0, 720.0), 25.0),
    ]

    patterns = get_paper_benchmark_patterns()
    inst = FVTRPInstance(
        name="Shanghai Auto Dealership Fleet (15 Dealers)",
        city_name="Shanghai, China",
        depot=depot,
        dealers=real_dealers,
        loading_patterns=patterns,
        max_route_duration=720.0,  # 12 hours shift
        speed_kmh=45.0
    )
    inst.compute_distance_matrix_if_needed()
    return inst

def get_metro_large_instance(num_dealers: int = 25, random_seed: int = 42) -> FVTRPInstance:
    """Generates an extended metropolitan multi-type vehicle distribution instance."""
    np.random.seed(random_seed)
    depot = Depot(
        id=1,
        name="Regional Auto Distribution Center (RVDC)",
        lat=31.2989,
        lon=121.1685,
        time_window=(0.0, 1440.0),
        handling_time=0.0
    )

    dealers = []
    for i in range(2, num_dealers + 2):
        # Generate around depot within radius ~45km
        r_km = np.random.uniform(5.0, 45.0)
        theta = np.random.uniform(0, 2 * np.pi)
        d_lat = (r_km / 111.0) * np.cos(theta)
        d_lon = (r_km / (111.0 * np.cos(np.radians(depot.lat)))) * np.sin(theta)
        
        # Demands for Type 1, 2, 3 sampled uniformly from U[0, 3] as in paper Sect 6
        d1 = int(np.random.randint(0, 3))
        d2 = int(np.random.randint(0, 3))
        d3 = int(np.random.randint(0, 3))
        if d1 + d2 + d3 == 0:
            d1 = 1  # ensure at least 1 car demanded
        
        L = float(np.random.choice([60.0, 90.0, 120.0, 150.0, 180.0]))
        U = L + float(np.random.choice([240.0, 300.0, 360.0, 480.0]))
        P = float(np.random.choice([10.0, 15.0, 20.0]))
        
        dealers.append(Dealer(
            id=i,
            name=f"Automotive Dealer #{i}",
            lat=round(depot.lat + d_lat, 5),
            lon=round(depot.lon + d_lon, 5),
            demands=[d1, d2, d3],
            time_window=(L, U),
            handling_time=P
        ))

    patterns = get_paper_benchmark_patterns()
    inst = FVTRPInstance(
        name=f"Metropolitan Regional Fleet ({num_dealers} Dealers)",
        city_name="Shanghai & Yangtze Delta",
        depot=depot,
        dealers=dealers,
        loading_patterns=patterns,
        max_route_duration=900.0,
        speed_kmh=50.0
    )
    inst.compute_distance_matrix_if_needed()
    return inst
