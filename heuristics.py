"""
Heuristics module for Finished-Vehicle Transporter Routing Problem (FVTRP).
Implements:
1. Initial Solution Construction:
   - Nearest Neighbor Heuristic (with Split)
   - Sweep Heuristic (with Split)
2. Four Distinct Routing & Loading Heuristics:
   - Heuristic 1: 2-Opt Intra-Route Edge Uncrossing (Croes / Lin)
   - Heuristic 2: Clarke & Wright Savings Merge with Pattern Compatibility
   - Heuristic 3: Inter-Route Relocate / Shift (Savelsbergh)
   - Heuristic 4: Cross-Route Customer Swap (Osman)
   - Heuristic 5: Comprehensive Multi-Heuristic Pipeline (Sequential Local Search)
"""

import copy
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from core.model import (
    Dealer, Depot, LoadingPattern, FVTRPInstance,
    Route, FVTRPSolution
)
from core.solver import evaluate_route_segment, split_chromosome

def build_initial_solution(
    instance: FVTRPInstance,
    method: str = "nearest_neighbor",
    alpha: float = 0.8
) -> Tuple[List[Route], float, str]:
    """
    Constructs an initial feasible solution before heuristic improvement.
    Methods:
    - 'nearest_neighbor': Successive closest unvisited dealer
    - 'sweep': Angular radial sweep from depot
    - 'random': Natural dealer ordering
    """
    dealers = [d.id for d in instance.dealers]
    depot = instance.depot

    if method == "nearest_neighbor":
        unvisited = set(dealers)
        tour = []
        curr_nid = depot.id

        def get_dist(n1_id: int, n2_id: int) -> float:
            def to_idx(nid):
                if nid == depot.id:
                    return 0
                if instance.time_matrix is not None and instance.time_matrix.shape[0] == len(instance.dealers) + 2:
                    return nid - 1
                for k, d in enumerate(instance.dealers):
                    if d.id == nid:
                        return k + 1
                return 0
            return instance.time_matrix[to_idx(n1_id), to_idx(n2_id)]

        while unvisited:
            next_nid = min(unvisited, key=lambda nid: get_dist(curr_nid, nid))
            tour.append(next_nid)
            unvisited.remove(next_nid)
            curr_nid = next_nid

        perm = tour
        desc = "Greedy Nearest Neighbor (Initial)"

    elif method == "sweep":
        sweep_dealers = sorted(
            instance.dealers,
            key=lambda d: np.arctan2(d.lat - depot.lat, d.lon - depot.lon)
        )
        perm = [d.id for d in sweep_dealers]
        desc = "Radial Angular Sweep (Initial)"

    else:
        perm = list(dealers)
        desc = "Sequential Natural Order (Initial)"

    routes, cost = split_chromosome(perm, instance, alpha=alpha)
    if routes is None:
        # Fallback if strict time windows violated
        perm = list(dealers)
        routes, cost = split_chromosome(perm, instance, alpha=alpha)

    return routes if routes else [], cost, desc


# =========================================================================
# HEURISTIC 1: 2-OPT INTRA-ROUTE EDGE EXCHANGE (Croes 1958)
# =========================================================================
def heuristic_2opt(
    routes: List[Route],
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Tuple[List[Route], float, List[Dict]]:
    """
    Heuristic 1: 2-Opt Edge Uncrossing.
    Reverses sub-segments within each transporter route to remove crossing lines and reduce travel distance.
    """
    improved_routes = []
    history = []
    step = 1

    total_initial_time = sum(r.travel_time for r in routes)
    history.append({
        "step": step,
        "name": "Start 2-Opt",
        "travel_time": total_initial_time,
        "action": "Baseline initial routes"
    })

    for r_idx, r in enumerate(routes):
        best_dealers = list(r.dealer_ids)
        best_eval = evaluate_route_segment([instance.get_dealer_by_id(did) for did in best_dealers], instance, alpha=alpha)
        best_cost = best_eval["edge_cost"] if best_eval else float('inf')

        improved = True
        iterations = 0
        while improved and iterations < 30:
            improved = False
            iterations += 1
            n_d = len(best_dealers)
            if n_d <= 2:
                break

            for i in range(n_d - 1):
                for j in range(i + 1, n_d):
                    cand_dealers = best_dealers[:i] + best_dealers[i:j + 1][::-1] + best_dealers[j + 1:]
                    cand_eval = evaluate_route_segment(
                        [instance.get_dealer_by_id(did) for did in cand_dealers],
                        instance,
                        alpha=alpha
                    )
                    if cand_eval is not None and cand_eval["edge_cost"] < best_cost - 1e-4:
                        best_dealers = cand_dealers
                        best_cost = cand_eval["edge_cost"]
                        best_eval = cand_eval
                        improved = True
                        step += 1
                        history.append({
                            "step": step,
                            "name": "2-Opt Edge Inversion",
                            "travel_time": cand_eval["travel_time"],
                            "action": f"Inverted segment [{i}:{j}] on Transporter {r.route_id}"
                        })
                        break
                if improved:
                    break

        if best_eval is not None:
            improved_routes.append(Route(
                route_id=r.route_id,
                transporter_id=r.transporter_id,
                loading_pattern=best_eval["pattern"],
                dealer_ids=best_dealers,
                steps=best_eval["steps"],
                loaded_demands=best_eval["loaded_demands"],
                loading_ratio=best_eval["ratio"],
                travel_time=best_eval["travel_time"],
                total_duration=best_eval["total_duration"],
                empty_slots=best_eval["empty_slots"],
                is_feasible=True
            ))
        else:
            improved_routes.append(r)

    total_cost = sum(r.travel_time for r in improved_routes)
    return improved_routes, total_cost, history


# =========================================================================
# HEURISTIC 2: CLARKE & WRIGHT SAVINGS HEURISTIC (Clarke & Wright 1964)
# =========================================================================
def heuristic_clarke_wright_savings(
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Tuple[List[Route], float, List[Dict]]:
    """
    Heuristic 2: Clarke & Wright Savings Algorithm adapted for Loading Patterns.
    Computes savings S_ij = T_0i + T_0j - T_ij and merges routes where loading patterns are loadable.
    """
    depot_id = instance.depot.id
    dealers = instance.dealers

    def get_time(n1_id: int, n2_id: int) -> float:
        def to_idx(nid):
            if nid == depot_id:
                return 0
            if instance.time_matrix is not None and instance.time_matrix.shape[0] == len(instance.dealers) + 2:
                return nid - 1
            for k, d in enumerate(instance.dealers):
                if d.id == nid:
                    return k + 1
            return 0
        return instance.time_matrix[to_idx(n1_id), to_idx(n2_id)]

    # Start with individual routes: Depot -> Dealer i -> Depot
    active_routes: List[List[int]] = [[d.id] for d in dealers]

    # Calculate savings matrix for all pairs (i, j)
    savings = []
    for i in range(len(dealers)):
        for j in range(i + 1, len(dealers)):
            di = dealers[i].id
            dj = dealers[j].id
            s = get_time(depot_id, di) + get_time(depot_id, dj) - get_time(di, dj)
            savings.append((s, di, dj))

    # Sort savings descending
    savings.sort(key=lambda x: x[0], reverse=True)

    history = []
    step = 1

    # Attempt to merge based on savings
    for s_val, di, dj in savings:
        if s_val <= 0:
            continue

        # Find routes containing di and dj
        r_i = None
        r_j = None
        for r in active_routes:
            if di in r:
                r_i = r
            if dj in r:
                r_j = r

        if r_i is not None and r_j is not None and r_i != r_j:
            # Check merge orientations
            candidates = []
            if r_i[-1] == di and r_j[0] == dj:
                candidates.append(r_i + r_j)
            elif r_j[-1] == dj and r_i[0] == di:
                candidates.append(r_j + r_i)
            elif r_i[0] == di and r_j[0] == dj:
                candidates.append(r_i[::-1] + r_j)
            elif r_i[-1] == di and r_j[-1] == dj:
                candidates.append(r_i + r_j[::-1])

            merged_success = False
            for cand in candidates:
                cand_eval = evaluate_route_segment(
                    [instance.get_dealer_by_id(did) for did in cand],
                    instance,
                    alpha=alpha
                )
                if cand_eval is not None:
                    # Merge is feasible under loading patterns and time windows!
                    active_routes.remove(r_i)
                    active_routes.remove(r_j)
                    active_routes.append(cand)
                    merged_success = True
                    step += 1
                    history.append({
                        "step": step,
                        "name": "Savings Merge",
                        "travel_time": cand_eval["travel_time"],
                        "action": f"Merged route with {di} and {dj} (Saving: {s_val:.1f} min)"
                    })
                    break

    # Build final Route objects
    final_routes = []
    for idx, r_dealers in enumerate(active_routes):
        eval_res = evaluate_route_segment([instance.get_dealer_by_id(did) for did in r_dealers], instance, alpha=alpha)
        if eval_res is not None:
            final_routes.append(Route(
                route_id=idx + 1,
                transporter_id=idx + 1,
                loading_pattern=eval_res["pattern"],
                dealer_ids=r_dealers,
                steps=eval_res["steps"],
                loaded_demands=eval_res["loaded_demands"],
                loading_ratio=eval_res["ratio"],
                travel_time=eval_res["travel_time"],
                total_duration=eval_res["total_duration"],
                empty_slots=eval_res["empty_slots"],
                is_feasible=True
            ))

    total_cost = sum(r.travel_time for r in final_routes)
    return final_routes, total_cost, history


# =========================================================================
# HEURISTIC 3: INTER-ROUTE RELOCATE / INSERTION (Savelsbergh 1992)
# =========================================================================
def heuristic_relocate(
    routes: List[Route],
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Tuple[List[Route], float, List[Dict]]:
    """
    Heuristic 3: Inter-Route Customer Relocate (Shift).
    Moves a dealer from one transporter route to a more cost-effective position in another route.
    """
    current_routes = [list(r.dealer_ids) for r in routes]
    history = []
    step = 1

    def total_system_cost(rts_dealers):
        c = 0.0
        for r_d in rts_dealers:
            eval_res = evaluate_route_segment([instance.get_dealer_by_id(did) for did in r_d], instance, alpha=alpha)
            if eval_res is None:
                return float('inf')
            c += eval_res["travel_time"]
        return c

    best_cost = total_system_cost(current_routes)
    improved = True
    passes = 0

    while improved and passes < 25:
        improved = False
        passes += 1

        for r_from_idx in range(len(current_routes)):
            if len(current_routes[r_from_idx]) <= 1:
                continue

            for d_idx, dealer_id in enumerate(current_routes[r_from_idx]):
                # Candidate dealer to relocate
                for r_to_idx in range(len(current_routes)):
                    if r_from_idx == r_to_idx:
                        continue

                    # Try inserting dealer_id into every position in r_to
                    for ins_pos in range(len(current_routes[r_to_idx]) + 1):
                        cand_from = current_routes[r_from_idx][:d_idx] + current_routes[r_from_idx][d_idx + 1:]
                        cand_to = current_routes[r_to_idx][:ins_pos] + [dealer_id] + current_routes[r_to_idx][ins_pos:]

                        eval_from = evaluate_route_segment([instance.get_dealer_by_id(did) for did in cand_from], instance, alpha=alpha)
                        eval_to = evaluate_route_segment([instance.get_dealer_by_id(did) for did in cand_to], instance, alpha=alpha)

                        if eval_from is not None and eval_to is not None:
                            old_pair_cost = (
                                evaluate_route_segment([instance.get_dealer_by_id(did) for did in current_routes[r_from_idx]], instance, alpha=alpha)["travel_time"] +
                                evaluate_route_segment([instance.get_dealer_by_id(did) for did in current_routes[r_to_idx]], instance, alpha=alpha)["travel_time"]
                            )
                            new_pair_cost = eval_from["travel_time"] + eval_to["travel_time"]

                            if new_pair_cost < old_pair_cost - 1e-4:
                                current_routes[r_from_idx] = cand_from
                                current_routes[r_to_idx] = cand_to
                                best_cost = total_system_cost(current_routes)
                                improved = True
                                step += 1
                                history.append({
                                    "step": step,
                                    "name": "Customer Relocate",
                                    "travel_time": best_cost,
                                    "action": f"Relocated Dealer {dealer_id} from Transporter {r_from_idx + 1} to {r_to_idx + 1}"
                                })
                                break
                    if improved:
                        break
                if improved:
                    break

    # Build final Route objects
    final_routes = []
    for idx, r_dealers in enumerate(current_routes):
        eval_res = evaluate_route_segment([instance.get_dealer_by_id(did) for did in r_dealers], instance, alpha=alpha)
        if eval_res is not None:
            final_routes.append(Route(
                route_id=idx + 1,
                transporter_id=idx + 1,
                loading_pattern=eval_res["pattern"],
                dealer_ids=r_dealers,
                steps=eval_res["steps"],
                loaded_demands=eval_res["loaded_demands"],
                loading_ratio=eval_res["ratio"],
                travel_time=eval_res["travel_time"],
                total_duration=eval_res["total_duration"],
                empty_slots=eval_res["empty_slots"],
                is_feasible=True
            ))

    total_cost = sum(r.travel_time for r in final_routes)
    return final_routes, total_cost, history


# =========================================================================
# HEURISTIC 4: CROSS-ROUTE CUSTOMER SWAP (Osman 1993)
# =========================================================================
def heuristic_swap(
    routes: List[Route],
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Tuple[List[Route], float, List[Dict]]:
    """
    Heuristic 4: Cross-Route Customer Swap.
    Swaps two dealers between two distinct transporter routes and tests for pattern compatibility.
    """
    current_routes = [list(r.dealer_ids) for r in routes]
    history = []
    step = 1

    def pair_cost(r1, r2):
        e1 = evaluate_route_segment([instance.get_dealer_by_id(did) for did in r1], instance, alpha=alpha)
        e2 = evaluate_route_segment([instance.get_dealer_by_id(did) for did in r2], instance, alpha=alpha)
        if e1 is None or e2 is None:
            return float('inf')
        return e1["travel_time"] + e2["travel_time"]

    improved = True
    passes = 0

    while improved and passes < 25:
        improved = False
        passes += 1

        for r1_idx in range(len(current_routes) - 1):
            for r2_idx in range(r1_idx + 1, len(current_routes)):
                r1 = current_routes[r1_idx]
                r2 = current_routes[r2_idx]
                base_c = pair_cost(r1, r2)

                for i_idx, d1 in enumerate(r1):
                    for j_idx, d2 in enumerate(r2):
                        cand_r1 = list(r1)
                        cand_r2 = list(r2)
                        cand_r1[i_idx] = d2
                        cand_r2[j_idx] = d1

                        c_cand = pair_cost(cand_r1, cand_r2)
                        if c_cand < base_c - 1e-4:
                            current_routes[r1_idx] = cand_r1
                            current_routes[r2_idx] = cand_r2
                            base_c = c_cand
                            improved = True
                            step += 1
                            history.append({
                                "step": step,
                                "name": "Customer Swap",
                                "travel_time": c_cand,
                                "action": f"Swapped Dealer {d1} and {d2} between Transporters {r1_idx + 1} and {r2_idx + 1}"
                            })
                            break
                    if improved:
                        break
                if improved:
                    break

    # Build final Route objects
    final_routes = []
    for idx, r_dealers in enumerate(current_routes):
        eval_res = evaluate_route_segment([instance.get_dealer_by_id(did) for did in r_dealers], instance, alpha=alpha)
        if eval_res is not None:
            final_routes.append(Route(
                route_id=idx + 1,
                transporter_id=idx + 1,
                loading_pattern=eval_res["pattern"],
                dealer_ids=r_dealers,
                steps=eval_res["steps"],
                loaded_demands=eval_res["loaded_demands"],
                loading_ratio=eval_res["ratio"],
                travel_time=eval_res["travel_time"],
                total_duration=eval_res["total_duration"],
                empty_slots=eval_res["empty_slots"],
                is_feasible=True
            ))

    total_cost = sum(r.travel_time for r in final_routes)
    return final_routes, total_cost, history


# =========================================================================
# HEURISTIC 5: COMPREHENSIVE PIPELINE (Sequential Local Search)
# =========================================================================
def heuristic_pipeline(
    routes: List[Route],
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Tuple[List[Route], float, List[Dict]]:
    """
    Heuristic 5: Multi-Heuristic Pipeline.
    Sequentially runs: Relocate -> Swap -> 2-Opt until local optimum.
    """
    curr_routes = routes
    all_history = []

    # 1. Relocate
    curr_routes, c1, h1 = heuristic_relocate(curr_routes, instance, alpha=alpha)
    all_history.extend(h1)

    # 2. Swap
    curr_routes, c2, h2 = heuristic_swap(curr_routes, instance, alpha=alpha)
    all_history.extend(h2)

    # 3. 2-Opt
    curr_routes, c3, h3 = heuristic_2opt(curr_routes, instance, alpha=alpha)
    all_history.extend(h3)

    return curr_routes, c3, all_history
