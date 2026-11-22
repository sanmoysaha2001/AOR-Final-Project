"""
Solvers for Finished-Vehicle Transporter Routing Problem (FVTRP).
Implements:
1. Prins' Split Algorithm adapted for Loading Patterns & Time Windows (Liu et al. 2008, Hu et al. 2015)
2. Evolutionary Algorithm (Algorithm 1 in paper: PMX, Tournament, Swap/Inversion Mutation, Elitism)
3. 2-Opt Route Local Search Refinement
4. Multi-objective balancing (Eq 2: travel time vs Eq 17: empty slots / loading ratio)
"""

import time
import random
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from core.model import (
    Dealer, Depot, LoadingPattern, FVTRPInstance,
    Route, RouteStep, FVTRPSolution
)

def evaluate_route_segment(
    dealers_subseq: List[Dealer],
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Optional[Dict]:
    """
    Evaluates whether a sequence of dealers can form a feasible route.
    Checks:
    1. Vehicle type demand vs loading patterns (Eq. 10, 14)
    2. Time windows & handling times (Eq. 8, 9, 12)
    3. Maximum route duration Pc
    Returns dict with cost, pattern, ratio, schedule if feasible, else None.
    """
    if not dealers_subseq:
        return None

    num_dealers = len(dealers_subseq)
    
    # Check total demand per vehicle type
    cum_demand = np.zeros(3, dtype=int)
    for d in dealers_subseq:
        cum_demand += np.array(d.demands)

    # 1. Compatibility check with loading patterns
    # Find all loading patterns e such that cum_demand <= pattern.capacities
    feasible_patterns = []
    for pat in instance.loading_patterns:
        cap = np.array(pat.capacities)
        if np.all(cum_demand <= cap):
            tot_dem = np.sum(cum_demand)
            tot_cap = np.sum(cap)
            ratio = tot_dem / tot_cap if tot_cap > 0 else 0.0
            empty = tot_cap - tot_dem
            feasible_patterns.append({
                "pattern": pat,
                "ratio": ratio,
                "empty_slots": empty
            })

    if not feasible_patterns:
        return None  # No transporter loading pattern can carry this combination

    # Rank patterns:
    # If user wants to maximize loading ratio / minimize empty space, pick highest ratio
    feasible_patterns.sort(key=lambda x: (x["ratio"], -x["empty_slots"]), reverse=True)
    best_pat_info = feasible_patterns[0]
    best_pattern = best_pat_info["pattern"]
    best_ratio = best_pat_info["ratio"]
    empty_slots = best_pat_info["empty_slots"]

    # 2. Time Window & Travel Time simulation
    # Nodes in instance time_matrix:
    # Index 0 is Depot.
    # If instance has explicit time_matrix, map dealer ID to index in time_matrix.
    def get_matrix_index(node_id: int, is_depot_end: bool = False) -> int:
        if node_id == instance.depot.id:
            if is_depot_end and instance.time_matrix is not None and instance.time_matrix.shape[0] == len(instance.dealers) + 2:
                return instance.time_matrix.shape[0] - 1  # Node 12 in 12x12 paper matrix
            return 0
        # If paper benchmark 10 dealers with IDs 2..11, matrix index is node_id - 1
        if instance.time_matrix is not None and instance.time_matrix.shape[0] == len(instance.dealers) + 2:
            return node_id - 1
        # General case: index in [depot] + dealers
        for idx, d in enumerate(instance.dealers):
            if d.id == node_id:
                return idx + 1
        return 0

    curr_time = instance.depot.time_window[0]
    travel_time = 0.0
    prev_idx = get_matrix_index(instance.depot.id)
    steps: List[RouteStep] = []

    # Step at Depot start
    steps.append(RouteStep(
        node_id=instance.depot.id,
        node_name=instance.depot.name,
        lat=instance.depot.lat,
        lon=instance.depot.lon,
        arrival_time=0.0,
        wait_time=0.0,
        service_start=0.0,
        departure_time=curr_time,
        delivered_demands=[0, 0, 0]
    ))

    for d in dealers_subseq:
        d_idx = get_matrix_index(d.id)
        leg_time = instance.time_matrix[prev_idx, d_idx]
        travel_time += leg_time
        arrival = curr_time + leg_time

        # Check latest arrival
        if arrival > d.time_window[1]:
            return None  # Late arrival violation

        # Earliest arrival & wait
        wait = max(0.0, d.time_window[0] - arrival)
        start_service = arrival + wait
        departure = start_service + d.handling_time

        steps.append(RouteStep(
            node_id=d.id,
            node_name=d.name,
            lat=d.lat,
            lon=d.lon,
            arrival_time=round(arrival, 1),
            wait_time=round(wait, 1),
            service_start=round(start_service, 1),
            departure_time=round(departure, 1),
            delivered_demands=list(d.demands)
        ))

        curr_time = departure
        prev_idx = d_idx

    # Return to Depot
    depot_end_idx = get_matrix_index(instance.depot.id, is_depot_end=True)
    return_leg_time = instance.time_matrix[prev_idx, depot_end_idx]
    travel_time += return_leg_time
    final_arrival = curr_time + return_leg_time

    if final_arrival > instance.depot.time_window[1]:
        return None  # Depot return time window violation

    total_route_duration = final_arrival - steps[0].departure_time
    if total_route_duration > instance.max_route_duration:
        return None  # Max route duration Pc exceeded

    steps.append(RouteStep(
        node_id=instance.depot.id,
        node_name=instance.depot.name + " (Return)",
        lat=instance.depot.lat,
        lon=instance.depot.lon,
        arrival_time=round(final_arrival, 1),
        wait_time=0.0,
        service_start=round(final_arrival, 1),
        departure_time=round(final_arrival, 1),
        delivered_demands=[0, 0, 0]
    ))

    # Multi-objective edge cost:
    # alpha * travel_time + (1 - alpha) * empty_penalty
    # Empty penalty scaled to typical travel times (e.g., 20 mins per empty car slot)
    edge_cost = alpha * travel_time + (1.0 - alpha) * (empty_slots * 25.0)

    return {
        "travel_time": round(travel_time, 1),
        "total_duration": round(total_route_duration, 1),
        "pattern": best_pattern,
        "ratio": round(best_ratio, 4),
        "empty_slots": int(empty_slots),
        "loaded_demands": cum_demand.tolist(),
        "steps": steps,
        "edge_cost": edge_cost
    }


def split_chromosome(
    chromosome: List[int],
    instance: FVTRPInstance,
    alpha: float = 0.8
) -> Tuple[Optional[List[Route]], float]:
    """
    Applies Prins' Split Algorithm on a DAG to find the optimal partitioning
    of the dealer permutation into transporter routes satisfying all FVTRP constraints.
    Returns: (list_of_routes, total_cost)
    """
    n = len(chromosome)
    dealer_map = {d.id: d for d in instance.dealers}
    
    # DAG with n+1 nodes: 0, 1, ..., n
    dist = [float('inf')] * (n + 1)
    parent = [-1] * (n + 1)
    route_details = {}
    dist[0] = 0.0

    # Max capacity across all patterns
    cap_max = max(pat.total_capacity for pat in instance.loading_patterns)

    for i in range(n):
        if dist[i] == float('inf'):
            continue

        cum_dem = 0
        subseq_dealers = []

        for j in range(i + 1, n + 1):
            dealer_id = chromosome[j - 1]
            dealer = dealer_map[dealer_id]
            subseq_dealers.append(dealer)
            cum_dem += dealer.total_demand

            if cum_dem > cap_max:
                # Adding more dealers cannot be feasible under any single loading pattern
                break

            seg_res = evaluate_route_segment(subseq_dealers, instance, alpha=alpha)
            if seg_res is None:
                # If capacity violated, break early because demands are non-negative
                dem_vec = np.sum([d.demands for d in subseq_dealers], axis=0)
                if not any(np.all(dem_vec <= pat.capacities) for pat in instance.loading_patterns):
                    break
                continue

            cost = seg_res["edge_cost"]
            if dist[i] + cost < dist[j]:
                dist[j] = dist[i] + cost
                parent[j] = i
                route_details[(i, j)] = seg_res

    if dist[n] == float('inf'):
        return None, float('inf')

    # Backtrack shortest path
    routes = []
    curr = n
    route_counter = 1
    while curr > 0:
        p = parent[curr]
        info = route_details[(p, curr)]
        dealer_ids = [chromosome[k - 1] for k in range(p + 1, curr + 1)]
        routes.append(Route(
            route_id=route_counter,
            transporter_id=route_counter,
            loading_pattern=info["pattern"],
            dealer_ids=dealer_ids,
            steps=info["steps"],
            loaded_demands=info["loaded_demands"],
            loading_ratio=info["ratio"],
            travel_time=info["travel_time"],
            total_duration=info["total_duration"],
            empty_slots=info["empty_slots"],
            is_feasible=True
        ))
        route_counter += 1
        curr = p

    routes.reverse()
    # Re-index route_ids 1, 2, ...
    for idx, r in enumerate(routes):
        r.route_id = idx + 1
        r.transporter_id = idx + 1

    return routes, dist[n]


def pmx_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Partially Mapped Crossover (PMX) for permutations."""
    size = len(p1)
    if size < 3:
        return list(p1), list(p2)
    cx1 = random.randint(0, size - 2)
    cx2 = random.randint(cx1 + 1, size - 1)

    c1 = [-1] * size
    c2 = [-1] * size
    c1[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
    c2[cx1:cx2 + 1] = p2[cx1:cx2 + 1]

    for c, parent, other in [(c1, p2, p1), (c2, p1, p2)]:
        for i in range(cx1, cx2 + 1):
            val = parent[i]
            if val not in c:
                pos = i
                while cx1 <= pos <= cx2:
                    val_in_other = other[pos]
                    pos = parent.index(val_in_other)
                c[pos] = val
        for i in range(size):
            if c[i] == -1:
                c[i] = parent[i]

    return c1, c2


def mutate_permutation(ind: List[int], prob_swap: float = 0.5):
    """Random swap or 2-opt inversion mutation."""
    if len(ind) < 2:
        return
    if random.random() < prob_swap:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    else:
        i, j = sorted(random.sample(range(len(ind)), 2))
        ind[i:j + 1] = reversed(ind[i:j + 1])


def two_opt_route(route_dealers: List[int], instance: FVTRPInstance) -> List[int]:
    """2-Opt local search refinement to eliminate crosses in a route."""
    if len(route_dealers) <= 2:
        return route_dealers

    best = list(route_dealers)
    improved = True

    dealer_map = {d.id: d for d in instance.dealers}
    depot_id = instance.depot.id

    def get_dist(n1: int, n2: int) -> float:
        def to_idx(nid):
            if nid == depot_id:
                return 0
            if instance.time_matrix is not None and instance.time_matrix.shape[0] == len(instance.dealers) + 2:
                return nid - 1
            for k, d in enumerate(instance.dealers):
                if d.id == nid:
                    return k + 1
            return 0
        return instance.time_matrix[to_idx(n1), to_idx(n2)]

    def route_dist(seq: List[int]) -> float:
        total = get_dist(depot_id, seq[0])
        for k in range(len(seq) - 1):
            total += get_dist(seq[k], seq[k + 1])
        total += get_dist(seq[-1], depot_id)
        return total

    best_dist = route_dist(best)
    attempts = 0
    while improved and attempts < 25:
        improved = False
        attempts += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                d = route_dist(cand)
                if d < best_dist - 1e-4:
                    best = cand
                    best_dist = d
                    improved = True
                    break
            if improved:
                break
    return best


class EvolutionaryFvtrpSolver:
    """
    Evolutionary Algorithm for FVTRP following Algorithm 1 in Hu et al. (2015).
    Incorporates:
    - Tournament selection
    - PMX Crossover
    - Swap & Inversion Mutation
    - Prins' Split DAG Evaluation
    - Post-optimization 2-Opt local search
    """
    def __init__(
        self,
        instance: FVTRPInstance,
        alpha: float = 0.8,
        pop_size: int = 40,
        max_generations: int = 120,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.35,
        random_seed: Optional[int] = 42
    ):
        self.instance = instance
        self.alpha = alpha
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def initialize_population(self) -> List[List[int]]:
        dealers = [d.id for d in self.instance.dealers]
        pop = []

        # 1. Angle sweep heuristic from depot
        sweep_order = sorted(
            self.instance.dealers,
            key=lambda d: np.arctan2(d.lat - self.instance.depot.lat, d.lon - self.instance.depot.lon)
        )
        pop.append([d.id for d in sweep_order])
        pop.append([d.id for d in reversed(sweep_order)])

        # 2. Nearest neighbor tours starting from random seeds
        depot = self.instance.depot
        for _ in range(min(5, len(dealers))):
            unvisited = set(dealers)
            tour = []
            curr = depot.id
            while unvisited:
                # Pick closest
                best_next = min(
                    unvisited,
                    key=lambda nid: (
                        (self.instance.get_dealer_by_id(nid).lat - depot.lat)**2 +
                        (self.instance.get_dealer_by_id(nid).lon - depot.lon)**2
                    ) if curr == depot.id else (
                        (self.instance.get_dealer_by_id(nid).lat - self.instance.get_dealer_by_id(curr).lat)**2 +
                        (self.instance.get_dealer_by_id(nid).lon - self.instance.get_dealer_by_id(curr).lon)**2
                    )
                )
                tour.append(best_next)
                unvisited.remove(best_next)
                curr = best_next
            pop.append(tour)

        # 3. Random permutations for the rest
        while len(pop) < self.pop_size:
            p = list(dealers)
            random.shuffle(p)
            pop.append(p)

        return pop

    def solve(
        self,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> FVTRPSolution:
        start_time = time.time()
        pop = self.initialize_population()
        
        # Evaluate initial population
        pop_scores = []
        pop_routes = []
        for ind in pop:
            rts, cost = split_chromosome(ind, self.instance, alpha=self.alpha)
            pop_scores.append(cost)
            pop_routes.append(rts)

        best_cost = min(pop_scores)
        best_idx = pop_scores.index(best_cost)
        best_ind = list(pop[best_idx])
        best_routes = pop_routes[best_idx]

        history: List[Dict[str, float]] = []

        for gen in range(self.max_generations):
            avg_fitness = float(np.mean([s for s in pop_scores if s < float('inf')])) if any(s < float('inf') for s in pop_scores) else best_cost
            history.append({
                "generation": gen + 1,
                "best_fitness": best_cost,
                "average_fitness": avg_fitness
            })

            if progress_callback:
                progress_callback(gen + 1, self.max_generations, best_cost, avg_fitness)

            # Elitism: retain top 2 individuals
            sorted_indices = np.argsort(pop_scores)
            new_pop = [list(pop[sorted_indices[0]]), list(pop[sorted_indices[1]])]

            while len(new_pop) < self.pop_size:
                # Tournament selection (size 3)
                def tournament():
                    cand = random.sample(range(self.pop_size), 3)
                    return pop[min(cand, key=lambda idx: pop_scores[idx])]

                p1 = tournament()
                p2 = tournament()

                if random.random() < self.crossover_prob:
                    c1, c2 = pmx_crossover(p1, p2)
                else:
                    c1, c2 = list(p1), list(p2)

                if random.random() < self.mutation_prob:
                    mutate_permutation(c1)
                if random.random() < self.mutation_prob:
                    mutate_permutation(c2)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            pop_scores = []
            pop_routes = []
            for ind in pop:
                rts, cost = split_chromosome(ind, self.instance, alpha=self.alpha)
                pop_scores.append(cost)
                pop_routes.append(rts)

            curr_best = min(pop_scores)
            if curr_best < best_cost:
                best_cost = curr_best
                b_idx = pop_scores.index(best_cost)
                best_ind = list(pop[b_idx])
                best_routes = pop_routes[b_idx]

        # Post-optimization refinement: 2-Opt on each individual route
        refined_routes = []
        if best_routes is not None:
            # Reconstruct chromosome from refined routes and re-split
            new_chrom = []
            for r in best_routes:
                refined_dealers = two_opt_route(r.dealer_ids, self.instance)
                new_chrom.extend(refined_dealers)
            
            ref_rts, ref_cost = split_chromosome(new_chrom, self.instance, alpha=self.alpha)
            if ref_rts is not None and ref_cost <= best_cost:
                best_routes = ref_rts
                best_cost = ref_cost

        elapsed = time.time() - start_time

        # Calculate metrics
        total_time = sum(r.travel_time for r in best_routes) if best_routes else 0.0
        total_delivered = sum(sum(r.loaded_demands) for r in best_routes) if best_routes else 0
        total_cap = sum(r.loading_pattern.total_capacity for r in best_routes) if best_routes else 0
        avg_ratio = (total_delivered / total_cap) if total_cap > 0 else 0.0

        return FVTRPSolution(
            instance_name=self.instance.name,
            routes=best_routes if best_routes else [],
            total_travel_time=round(total_time, 1),
            total_vehicles_delivered=int(total_delivered),
            total_capacity_deployed=int(total_cap),
            overall_loading_ratio=round(avg_ratio, 4),
            computation_time_sec=round(elapsed, 3),
            algorithm_name="Evolutionary Algorithm with Prins-Split & 2-Opt (Hu et al. 2015)",
            convergence_history=history
        )
