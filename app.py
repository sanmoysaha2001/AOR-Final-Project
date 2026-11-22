"""
Finished-Vehicle Transporter Routing Problem (FVTRP) Decision Support System.
Streamlit Web Application implementing Step 3 of the project overview based on:
"Finished-vehicle transporter routing problem solved by loading pattern discovery"
(Hu et al., Annals of Operations Research, 2015).
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import streamlit.components.v1 as components

from core.model import (
    Dealer, Depot, LoadingPattern, FVTRPInstance,
    Route, FVTRPSolution, VEHICLE_TYPE_NAMES
)
from core.presets import (
    get_paper_benchmark_instance,
    get_shanghai_fleet_instance,
    get_metro_large_instance,
    get_paper_benchmark_patterns
)
from core.solver import (
    EvolutionaryFvtrpSolver,
    evaluate_route_segment,
    split_chromosome,
    two_opt_route
)
from core.heuristics import (
    build_initial_solution,
    heuristic_2opt,
    heuristic_clarke_wright_savings,
    heuristic_relocate,
    heuristic_swap,
    heuristic_pipeline
)
from ui.map_view import render_folium_map, render_plotly_map, render_euclidean_grid
from ui.charts import (
    plot_convergence_curve,
    plot_stowage_breakdown,
    plot_transporter_deck_schematic,
    plot_timetable_gantt,
    plot_heuristic_progress,
    plot_before_after_comparison
)

# Page configuration
st.set_page_config(
    page_title="FVTRP Decision Support System | Auto Logistics",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-size: 15px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
        padding: 0 18px;
    }
    .savings-badge {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if "instance" not in st.session_state:
    st.session_state.instance = get_paper_benchmark_instance()
    st.session_state.solution = None
    st.session_state.initial_routes = None
    st.session_state.heuristic_routes = None
    st.session_state.heuristic_history = []
    st.session_state.heuristic_name = "2-Opt Edge Uncrossing"
    st.session_state.initial_method_name = "Natural Unoptimized Sequence"
    st.session_state.active_scenario_name = "Paper Benchmark (10 Dealers)"
    st.session_state.capacity_multiplier = 1.0
    st.session_state.test_results = None

if st.session_state.get("test_results_version") != 4:
    st.session_state.test_results = None
    st.session_state.test_results_version = 4

# Helper function to solve with EA
def run_ea_solver(instance: FVTRPInstance, alpha: float, pop_size: int, generations: int, cx_prob: float, mut_prob: float) -> FVTRPSolution:
    solver = EvolutionaryFvtrpSolver(
        instance=instance,
        alpha=alpha,
        pop_size=pop_size,
        max_generations=generations,
        crossover_prob=cx_prob,
        mutation_prob=mut_prob,
        random_seed=42
    )
    return solver.solve()


def run_dashboard_test_cases():
    """Execute 18 FVTRP acceptance tests and return dashboard-ready results."""
    results = []

    def record(test_id, name, setup, acceptance, passed, observed):
        results.append({
            "ID": test_id,
            "Test Case": name,
            "Setup": setup,
            "Acceptance Criteria": acceptance,
            "Observed": observed,
            "Status": "SUCCESS" if passed else "FAILURE"
        })

    benchmark = get_paper_benchmark_instance()
    benchmark_perm = [5, 2, 6, 3, 7, 8, 9, 4, 10, 11]
    benchmark_routes, benchmark_cost = split_chromosome(
        benchmark_perm, benchmark, alpha=1.0
    )
    expected_routes = [[5, 2, 6, 3, 7], [8, 9, 4, 10, 11]]
    expected_route_facts = [(690.0, 1, 1.0), (653.0, 7, 0.8125)]
    actual_route_facts = [
        (float(route.travel_time), route.loading_pattern.id, float(route.loading_ratio))
        for route in benchmark_routes or []
    ]
    benchmark_passed = (
        benchmark_routes is not None
        and abs(benchmark_cost - 1343.0) < 0.001
        and [route.dealer_ids for route in benchmark_routes] == expected_routes
        and all(
            abs(actual[0] - expected[0]) < 0.001
            and actual[1] == expected[1]
            and abs(actual[2] - expected[2]) < 0.001
            for actual, expected in zip(actual_route_facts, expected_route_facts)
        )
        and len(actual_route_facts) == len(expected_route_facts)
    )
    record(
        "TC-01",
        "Paper benchmark reference solution",
        "Run the documented 10-dealer benchmark permutation with travel-time objective.",
        "Expected two reference routes, 690 and 653 minutes, patterns 1 and 7, total 1,343 minutes.",
        benchmark_passed,
        f"Routes={[route.dealer_ids for route in benchmark_routes or []]}; "
        f"route facts={actual_route_facts}; total={benchmark_cost:.1f} min."
    )

    served_ids = [
        dealer_id
        for route in benchmark_routes or []
        for dealer_id in route.dealer_ids
    ]
    expected_demand = [
        sum(dealer.demands[type_index] for dealer in benchmark.dealers)
        for type_index in range(3)
    ]
    actual_demand = [
        sum(route.loaded_demands[type_index] for route in benchmark_routes or [])
        for type_index in range(3)
    ]
    accounting_passed = (
        benchmark_routes is not None
        and sorted(served_ids) == sorted(dealer.id for dealer in benchmark.dealers)
        and len(served_ids) == len(set(served_ids))
        and actual_demand == expected_demand
    )
    record(
        "TC-02",
        "Dealer coverage and vehicle-demand accounting",
        "Reconcile every dealer and vehicle type across the benchmark routes.",
        "Each dealer appears once; delivered vehicle totals equal total dealer demand.",
        accounting_passed,
        f"Dealers served={len(served_ids)}/{len(benchmark.dealers)} "
        f"(unique={len(set(served_ids))}); vehicles by type "
        f"expected={expected_demand}, routed={actual_demand}."
    )

    route_constraints_passed = benchmark_routes is not None
    if benchmark_routes is not None:
        for route in benchmark_routes:
            route_capacity_ok = all(
                route.loaded_demands[index] <= route.loading_pattern.capacities[index]
                for index in range(len(route.loaded_demands))
            )
            dealer_windows_ok = all(
                benchmark.get_dealer_by_id(step.node_id) is not None
                and step.arrival_time
                <= benchmark.get_dealer_by_id(step.node_id).time_window[1]
                for step in route.steps
                if step.node_id != benchmark.depot.id
            )
            depot_return_ok = (
                route.steps[-1].arrival_time <= benchmark.depot.time_window[1]
            )
            duration_ok = route.total_duration <= benchmark.max_route_duration
            route_constraints_passed = route_constraints_passed and (
                route.is_feasible
                and route_capacity_ok
                and dealer_windows_ok
                and depot_return_ok
                and duration_ok
            )
    route_observations = [
        f"Route {route.route_id}: demand={route.loaded_demands}, "
        f"capacity={route.loading_pattern.capacities}, "
        f"duration={route.total_duration:.1f} min"
        for route in benchmark_routes or []
    ]
    record(
        "TC-03",
        "Route capacity, time-window, and shift constraints",
        "Inspect each route returned for the benchmark solution.",
        "Every route fits its per-type deck limits, dealer/depot time windows, "
        "and the maximum route duration.",
        route_constraints_passed,
        "; ".join(route_observations) or "No routes were returned."
    )

    over_capacity_dealer = Dealer(
        id=999,
        name="Over-capacity test dealer",
        lat=benchmark.depot.lat,
        lon=benchmark.depot.lon,
        demands=[99, 99, 99]
    )
    over_capacity_rejected = evaluate_route_segment(
        [over_capacity_dealer], benchmark, alpha=1.0
    ) is None
    record(
        "TC-04",
        "Over-capacity load is rejected",
        "Evaluate one dealer with demand [99, 99, 99] against the benchmark fleet.",
        "No loading pattern can carry the demand, so the route must be rejected.",
        over_capacity_rejected,
        "Route rejected as infeasible." if over_capacity_rejected
        else "Route was unexpectedly accepted."
    )

    window_instance = get_paper_benchmark_instance()
    window_dealer = window_instance.get_dealer_by_id(2)
    if window_dealer is not None:
        window_dealer.time_window = (1.0, 100.0)
    window_rejected = (
        window_dealer is not None
        and evaluate_route_segment([window_dealer], window_instance, alpha=1.0) is None
    )
    record(
        "TC-05",
        "Latest-arrival time window is enforced",
        "Set dealer 2's latest arrival to minute 100; depot-to-dealer travel is 141 minutes.",
        "The route must be rejected because it cannot meet the dealer's latest arrival.",
        window_rejected,
        "Late-arrival route rejected." if window_rejected
        else "Late-arrival route was unexpectedly accepted."
    )

    duration_instance = get_paper_benchmark_instance()
    duration_instance.max_route_duration = 100.0
    duration_dealer = duration_instance.get_dealer_by_id(2)
    duration_rejected = (
        duration_dealer is not None
        and evaluate_route_segment([duration_dealer], duration_instance, alpha=1.0) is None
    )
    record(
        "TC-06",
        "Maximum route duration is enforced",
        "Set the shift limit to 100 minutes for a depot-to-dealer-2 round trip.",
        "The route must be rejected when its duration exceeds the shift limit.",
        duration_rejected,
        "Route exceeding the 100-minute limit rejected." if duration_rejected
        else "Over-duration route was unexpectedly accepted."
    )

    scenario_results = []
    for scenario_name, scenario in (
        ("Shanghai", get_shanghai_fleet_instance()),
        ("Metropolitan", get_metro_large_instance(20))
    ):
        scenario_routes, scenario_cost = split_chromosome(
            [dealer.id for dealer in scenario.dealers],
            scenario,
            alpha=1.0
        )
        scenario_served = [
            dealer_id
            for route in scenario_routes or []
            for dealer_id in route.dealer_ids
        ]
        scenario_ok = (
            scenario_routes is not None
            and sorted(scenario_served) == sorted(
                dealer.id for dealer in scenario.dealers
            )
            and len(scenario_served) == len(set(scenario_served))
            and all(route.is_feasible for route in scenario_routes)
        )
        scenario_results.append(
            (scenario_name, scenario_ok, len(scenario_routes or []),
             len(scenario_served), len(scenario.dealers), scenario_cost)
        )
    scenarios_passed = all(result[1] for result in scenario_results)
    record(
        "TC-07",
        "Expanded-scenario route coverage",
        "Solve the supplied 15-dealer Shanghai and 20-dealer Metropolitan scenarios.",
        "Both scenarios produce feasible routes that serve every dealer exactly once.",
        scenarios_passed,
        "; ".join(
            f"{name}: {served}/{total} dealers, {route_count} routes, "
            f"cost={cost:.1f} min"
            for name, _, route_count, served, total, cost in scenario_results
        )
    )

    objective_instance = get_paper_benchmark_instance()
    objective_dealer = objective_instance.get_dealer_by_id(2)
    travel_objective = (
        evaluate_route_segment([objective_dealer], objective_instance, alpha=1.0)
        if objective_dealer is not None else None
    )
    loading_objective = (
        evaluate_route_segment([objective_dealer], objective_instance, alpha=0.0)
        if objective_dealer is not None else None
    )
    objective_passed = (
        travel_objective is not None
        and loading_objective is not None
        and abs(travel_objective["edge_cost"] - travel_objective["travel_time"]) < 0.001
        and abs(
            loading_objective["edge_cost"]
            - loading_objective["empty_slots"] * 25.0
        ) < 0.001
    )
    objective_observed = (
        f"alpha=1 cost/time={travel_objective['edge_cost']:.1f}/"
        f"{travel_objective['travel_time']:.1f}; "
        f"alpha=0 cost/empty penalty={loading_objective['edge_cost']:.1f}/"
        f"{loading_objective['empty_slots'] * 25.0:.1f}."
        if travel_objective is not None and loading_objective is not None
        else "Could not evaluate the objective for the benchmark dealer."
    )
    record(
        "TC-08",
        "Objective weighting at both extremes",
        "Evaluate the same valid route with alpha=1.0 and alpha=0.0.",
        "At alpha=1, edge cost equals travel time; at alpha=0, it equals "
        "the empty-slot penalty.",
        objective_passed,
        objective_observed
    )

    empty_routes, empty_cost = split_chromosome([], benchmark, alpha=1.0)
    empty_rejected = empty_routes is None
    record(
        "TC-09",
        "Empty chromosome is rejected",
        "Run the split solver with no dealer IDs against the 10-dealer benchmark.",
        "The solver must report no feasible solution instead of an empty, zero-cost plan.",
        empty_rejected,
        f"Returned routes={empty_routes}, cost={empty_cost:.1f} min."
    )

    duplicate_routes, duplicate_cost = split_chromosome([2, 2], benchmark, alpha=1.0)
    duplicate_ids = [
        dealer_id
        for route in duplicate_routes or []
        for dealer_id in route.dealer_ids
    ]
    duplicate_rejected = (
        duplicate_routes is None
        and len(duplicate_ids) == len(set(duplicate_ids))
    )
    record(
        "TC-10",
        "Duplicate dealer IDs are rejected",
        "Run the split solver with dealer 2 listed twice.",
        "Invalid duplicate input must be rejected; no dealer can be delivered twice.",
        duplicate_rejected,
        f"Returned dealers={duplicate_ids}, cost={duplicate_cost:.1f} min."
    )

    incomplete_perm = [5, 2, 6, 3, 7, 8, 9, 4, 10]
    incomplete_routes, incomplete_cost = split_chromosome(
        incomplete_perm, benchmark, alpha=1.0
    )
    incomplete_ids = [
        dealer_id
        for route in incomplete_routes or []
        for dealer_id in route.dealer_ids
    ]
    incomplete_rejected = (
        incomplete_routes is None
        and sorted(incomplete_ids) == sorted(dealer.id for dealer in benchmark.dealers)
    )
    missing_ids = sorted(
        set(dealer.id for dealer in benchmark.dealers) - set(incomplete_ids)
    )
    record(
        "TC-11",
        "Missing dealer IDs are detected",
        "Run the solver with a benchmark sequence that omits dealer 11.",
        "An incomplete chromosome must not be reported as a complete solution.",
        incomplete_rejected,
        f"Returned dealers={len(incomplete_ids)}/{len(benchmark.dealers)}, "
        f"missing={missing_ids}, cost={incomplete_cost:.1f} min."
    )

    unknown_routes = None
    unknown_error = None
    try:
        unknown_routes, _ = split_chromosome([999], benchmark, alpha=1.0)
    except KeyError as error:
        unknown_error = str(error)
    unknown_rejected_cleanly = unknown_routes is None and unknown_error is None
    record(
        "TC-12",
        "Unknown dealer ID is handled cleanly",
        "Pass dealer ID 999, which is not present in the benchmark instance.",
        "The solver must return a controlled infeasible result, not raise an exception.",
        unknown_rejected_cleanly,
        f"Raised KeyError for unknown dealer ID: {unknown_error}."
        if unknown_error is not None
        else f"Returned routes={unknown_routes}."
    )

    initial_instance = get_paper_benchmark_instance()
    initial_routes, initial_cost, initial_method = build_initial_solution(
        initial_instance, method="nearest_neighbor", alpha=1.0
    )
    initial_ids = [
        dealer_id
        for route in initial_routes
        for dealer_id in route.dealer_ids
    ]
    initial_passed = (
        sorted(initial_ids) == sorted(dealer.id for dealer in initial_instance.dealers)
        and len(initial_ids) == len(set(initial_ids))
        and all(route.is_feasible for route in initial_routes)
    )
    record(
        "TC-13",
        "Nearest-neighbor initial solution",
        "Generate an initial benchmark plan using the nearest-neighbor method.",
        "All dealers appear once and every generated route is marked feasible.",
        initial_passed,
        f"{initial_method}: {len(initial_routes)} routes, "
        f"{len(initial_ids)}/{len(initial_instance.dealers)} dealers, "
        f"cost={initial_cost:.1f} min."
    )

    improved_routes, improved_cost, _ = heuristic_2opt(
        initial_routes, initial_instance, alpha=1.0
    )
    improved_ids = [
        dealer_id
        for route in improved_routes
        for dealer_id in route.dealer_ids
    ]
    improvement_passed = (
        improved_cost <= initial_cost
        and sorted(improved_ids) == sorted(dealer.id for dealer in initial_instance.dealers)
        and len(improved_ids) == len(set(improved_ids))
        and all(route.is_feasible for route in improved_routes)
    )
    record(
        "TC-14",
        "2-Opt preserves coverage without worsening cost",
        "Apply 2-Opt to the nearest-neighbor benchmark solution.",
        "The result preserves all dealers and feasible routes while not increasing travel cost.",
        improvement_passed,
        f"Travel cost {initial_cost:.1f} -> {improved_cost:.1f} min; "
        f"dealers={len(improved_ids)}/{len(initial_instance.dealers)}."
    )

    wait_instance = get_paper_benchmark_instance()
    wait_dealer = wait_instance.get_dealer_by_id(2)
    if wait_dealer is not None:
        wait_dealer.time_window = (300.0, 400.0)
    wait_result = (
        evaluate_route_segment([wait_dealer], wait_instance, alpha=1.0)
        if wait_dealer is not None else None
    )
    wait_passed = (
        wait_result is not None
        and wait_result["steps"][1].arrival_time <= wait_dealer.time_window[1]
        and wait_result["steps"][1].service_start >= wait_dealer.time_window[0]
        and wait_result["steps"][1].wait_time > 0
    )
    wait_observed = (
        f"Arrival={wait_result['steps'][1].arrival_time:.1f}, "
        f"wait={wait_result['steps'][1].wait_time:.1f}, "
        f"service start={wait_result['steps'][1].service_start:.1f} min."
        if wait_result is not None
        else "Route was not feasible."
    )
    record(
        "TC-15",
        "Earliest-arrival waiting is scheduled",
        "Set dealer 2's earliest service time to minute 300.",
        "The route waits after arrival and begins service no earlier than minute 300.",
        wait_passed,
        wait_observed
    )

    matrix = benchmark.time_matrix
    matrix_passed = (
        matrix is not None
        and matrix.shape == (len(benchmark.dealers) + 2, len(benchmark.dealers) + 2)
        and bool(np.isfinite(matrix).all())
        and bool((matrix >= 0).all())
        and bool(np.allclose(matrix, matrix.T))
        and bool(np.allclose(np.diag(matrix), 0))
    )
    record(
        "TC-16",
        "Benchmark travel-time matrix integrity",
        "Inspect dimensions, finiteness, signs, symmetry, and diagonal values.",
        "The matrix is 12x12, finite, non-negative, symmetric, and zero on its diagonal.",
        matrix_passed,
        f"Shape={matrix.shape if matrix is not None else None}; "
        f"finite/non-negative/symmetric/zero-diagonal="
        f"{matrix is not None and bool(np.isfinite(matrix).all())}/"
        f"{matrix is not None and bool((matrix >= 0).all())}/"
        f"{matrix is not None and bool(np.allclose(matrix, matrix.T))}/"
        f"{matrix is not None and bool(np.allclose(np.diag(matrix), 0))}."
    )

    zero_demand_instance = get_paper_benchmark_instance()
    zero_demand_dealer = zero_demand_instance.get_dealer_by_id(2)
    if zero_demand_dealer is not None:
        zero_demand_dealer.demands = [0, 0, 0]
    zero_demand_result = (
        evaluate_route_segment(
            [zero_demand_dealer], zero_demand_instance, alpha=1.0
        )
        if zero_demand_dealer is not None else None
    )
    zero_demand_passed = (
        zero_demand_result is not None
        and zero_demand_result["loaded_demands"] == [0, 0, 0]
        and zero_demand_result["travel_time"] > 0
    )
    record(
        "TC-17",
        "Zero-demand dealer route remains feasible",
        "Set dealer 2's order to zero vehicles of all three types.",
        "The valid service stop can still be routed with zero loaded demand.",
        zero_demand_passed,
        f"Route feasible={zero_demand_result is not None}; "
        f"loaded demand={zero_demand_result['loaded_demands'] if zero_demand_result else None}."
    )

    closure_routes, closure_cost = split_chromosome(
        benchmark_perm, benchmark, alpha=1.0
    )
    depot_closure_passed = (
        closure_routes is not None
        and all(
            route.steps[0].node_id == benchmark.depot.id
            and route.steps[-1].node_id == benchmark.depot.id
            for route in closure_routes
        )
        and abs(
            sum(route.travel_time for route in closure_routes) - closure_cost
        ) < 0.001
    )
    record(
        "TC-18",
        "Route depot closure and travel-cost reconciliation",
        "Inspect the start/end nodes and travel-time totals for both reference routes.",
        "Every route starts and returns to the depot; total travel equals summed route travel.",
        depot_closure_passed,
        f"Routes={len(closure_routes or [])}; "
        f"sum travel={sum(route.travel_time for route in closure_routes or []):.1f} min; "
        f"solver cost={closure_cost:.1f} min."
    )

    return results


# ==========================================
# SIDEBAR: CONTROLS & SCENARIOS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/semi-truck.png", width=64)
    st.title("FVTRP Control Center")
    st.caption("Finished-Vehicle Transporter Routing Problem DSS")

    st.markdown("---")
    st.subheader("📁 Scenario Selection")
    scenario_choice = st.selectbox(
        "Choose Logistics Scenario",
        [
            "Paper Benchmark 10-Dealers (Hu et al. 2015)",
            "Shanghai Auto Dealership Fleet (15 Real Hubs)",
            "Metropolitan Regional Fleet (20 Dealers)",
            "Custom User-Defined Scenario"
        ],
        index=0
    )

    # Reload scenario if changed
    if scenario_choice != st.session_state.active_scenario_name:
        if "Paper Benchmark" in scenario_choice:
            st.session_state.instance = get_paper_benchmark_instance()
        elif "Shanghai Auto" in scenario_choice:
            st.session_state.instance = get_shanghai_fleet_instance()
        elif "Metropolitan" in scenario_choice:
            st.session_state.instance = get_metro_large_instance(20)
        st.session_state.active_scenario_name = scenario_choice
        st.session_state.solution = None
        st.session_state.initial_routes = None
        st.session_state.heuristic_routes = None
        st.session_state.capacity_multiplier = 1.0
        st.rerun()

    # ------------------------------------------
    # CARRIER / TRANSPORTER CAPACITY CONTROLS
    # ------------------------------------------
    st.markdown("---")
    st.subheader("🚚 Transporter / Carrier Capacity")
    st.caption("Adjust maximal vehicle slots & carrier deck sizes:")

    cap_preset = st.selectbox(
        "Carrier Capacity Fleet Preset",
        [
            "Custom / Standard (Paper Table 3)",
            "High-Capacity Transporters (18-24 Cars)",
            "Compact City Carriers (8-12 Cars)",
            "Scale by Multiplier"
        ],
        index=0
    )

    if cap_preset == "High-Capacity Transporters (18-24 Cars)":
        if st.button("Apply High-Capacity Carrier Preset", use_container_width=True):
            for p in st.session_state.instance.loading_patterns:
                p.capacities = [int(round(c * 1.4)) for c in p.capacities]
            st.session_state.solution = None
            st.session_state.initial_routes = None
            st.session_state.heuristic_routes = None
            st.success("High-capacity carrier specifications applied!")
            st.rerun()

    elif cap_preset == "Compact City Carriers (8-12 Cars)":
        if st.button("Apply Compact Carrier Preset", use_container_width=True):
            for p in st.session_state.instance.loading_patterns:
                p.capacities = [max(1, int(round(c * 0.7))) for c in p.capacities]
            st.session_state.solution = None
            st.session_state.initial_routes = None
            st.session_state.heuristic_routes = None
            st.success("Compact carrier specifications applied!")
            st.rerun()

    elif cap_preset == "Scale by Multiplier":
        mult = st.slider("Global Capacity Scale Factor", 0.5, 2.0, float(st.session_state.capacity_multiplier), step=0.1)
        if st.button("Apply Scaled Capacities", use_container_width=True):
            base_pats = get_paper_benchmark_patterns()
            for idx, p in enumerate(st.session_state.instance.loading_patterns):
                base_c = base_pats[idx % len(base_pats)].capacities
                p.capacities = [max(1, int(round(c * mult))) for c in base_c]
            st.session_state.capacity_multiplier = mult
            st.session_state.solution = None
            st.session_state.initial_routes = None
            st.session_state.heuristic_routes = None
            st.success(f"Applied {mult:.1f}x carrier capacity scaling!")
            st.rerun()

    # Current fleet capacity summary
    max_cap = max(p.total_capacity for p in st.session_state.instance.loading_patterns)
    min_cap = min(p.total_capacity for p in st.session_state.instance.loading_patterns)
    st.info(f"Active Carrier Deck Limits: **{min_cap} to {max_cap} vehicles** across {len(st.session_state.instance.loading_patterns)} loading patterns.")

    # ------------------------------------------
    # OBJECTIVE FUNCTION
    # ------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Optimization Objective")
    obj_mode = st.radio(
        "Objective Function Formulation",
        [
            "Minimize Travel Cost (Paper Eq. 2)",
            "Maximize Loading Ratio / Min Empty (Paper Eq. 17)",
            "Balanced Multi-Objective (Trade-off)"
        ],
        index=0
    )

    if "Minimize Travel" in obj_mode:
        alpha = 1.0
        st.caption("Prioritizes minimum distance & time: min f = ∑ T_ij * x_ijk")
    elif "Maximize Loading" in obj_mode:
        alpha = 0.0
        st.caption("Prioritizes filling every transporter deck slot: min g = ∑(Θ_eo - D_io)")
    else:
        alpha = st.slider("Travel Time vs Stowage Weight (α)", 0.0, 1.0, 0.7, 0.05,
                          help="α = 1.0 is pure travel time; α = 0.0 is pure empty space minimization")

    # ------------------------------------------
    # SOLVER HYPERPARAMETERS
    # ------------------------------------------
    st.markdown("---")
    st.subheader("⚙️ Evolutionary Algorithm")
    with st.expander("EA Hyperparameters", expanded=False):
        pop_size = st.slider("Population Size (Ps)", 10, 80, 40, step=5)
        generations = st.slider("Generations (Pg)", 20, 250, 70, step=10)
        crossover_p = st.slider("PMX Crossover Prob (Px)", 0.5, 1.0, 0.8, step=0.05)
        mutation_p = st.slider("Mutation Prob (Pm)", 0.1, 0.6, 0.35, step=0.05)
        max_duration = st.number_input("Max Route Shift Pc (mins)", value=1440.0, step=60.0)
        st.session_state.instance.max_route_duration = max_duration

    st.markdown("---")
    if st.button("🚀 Run Full Evolutionary Optimization", type="primary", use_container_width=True):
        with st.spinner("Executing Prins-Split Evolutionary Algorithm..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, best_fit, avg_fit):
                frac = current / total
                progress_bar.progress(frac)
                status_text.text(f"Generation {current}/{total} | Best Cost: {best_fit:.1f}")

            if st.session_state.instance.time_matrix is None or st.session_state.instance.time_matrix.shape[0] != len(st.session_state.instance.dealers) + (2 if "Benchmark" in st.session_state.instance.name else 1):
                st.session_state.instance.compute_distance_matrix_if_needed()

            sol = run_ea_solver(
                instance=st.session_state.instance,
                alpha=alpha,
                pop_size=pop_size,
                generations=generations,
                cx_prob=crossover_p,
                mut_prob=mutation_p
            )
            st.session_state.solution = sol
            progress_bar.empty()
            status_text.empty()
            st.success(f"Optimized in {sol.computation_time_sec:.2f}s! Found {len(sol.routes)} routes.")
            st.rerun()

# Ensure base solution exists
if st.session_state.solution is None:
    st.session_state.solution = run_ea_solver(
        instance=st.session_state.instance,
        alpha=1.0 if "Minimize Travel" in obj_mode else (0.0 if "Maximize Loading" in obj_mode else 0.7),
        pop_size=35,
        generations=60,
        cx_prob=0.8,
        mut_prob=0.35
    )

sol: FVTRPSolution = st.session_state.solution
inst: FVTRPInstance = st.session_state.instance


# ==========================================
# MAIN PAGE HEADER & KPIS
# ==========================================
st.markdown("<div class='main-header'>Finished-Vehicle Transporter Routing Decision Support System</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Outbound Logistics Optimization for <b>{inst.name}</b> | City: <b>{inst.city_name}</b> | RVDC Hub: <b>{inst.depot.name}</b></div>", unsafe_allow_html=True)

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        label="⏱️ Total Travel Time",
        value=f"{sol.total_travel_time:.0f} min",
        delta=f"{(sol.total_travel_time / 60):.1f} hrs"
    )
with col2:
    st.metric(
        label="🚛 Fleet Dispatched",
        value=f"{len(sol.routes)} Carriers",
        delta=f"{sum(len(r.dealer_ids) for r in sol.routes)} stops"
    )
with col3:
    st.metric(
        label="🎯 Avg Deck Loading Ratio",
        value=f"{sol.overall_loading_ratio:.1%}",
        delta=f"{(1.0 - sol.overall_loading_ratio):.1%} empty rate"
    )
with col4:
    st.metric(
        label="🚗 Finished Vehicles Delivered",
        value=f"{sol.total_vehicles_delivered} Units",
        delta=f"Cap: {sol.total_capacity_deployed} slots"
    )
with col5:
    st.metric(
        label="⚡ Solution Runtime",
        value=f"{sol.computation_time_sec:.3f} s",
        delta="Optimal (Prins Split)"
    )

st.markdown("")

# ==========================================
# MAIN TABS INTERFACE
# ==========================================
tabs = st.tabs([
    "✍️ Live Decision Variable & Capacity Studio",
    "🚚 Transporter Stowage & Decks (Table 5)",
    "🗺️ Route Map & Visualizer",
    "🔬 Heuristics & Before/After Comparison",
    "📊 Dispatch Timetable & Convergence",
    "📑 Step 3 DSS Documentation & Export"
])


# ------------------------------------------
# TAB 1: INTERACTIVE ROUTE MAP & DETAILS
# ------------------------------------------
with tabs[2]:
    st.markdown("### 🗺️ Visual Transporter Routes Laid Over City Map")
    st.caption(f"Depot and dealer distribution network across **{inst.city_name}**.")

    # Map Style / Engine Selector
    map_ctrl1, map_ctrl2, map_ctrl3 = st.columns([4, 3, 3])
    with map_ctrl1:
        map_engine = st.radio(
            "Select Map Engine (All 100% Free, Zero API Keys)",
            [
                "🌐 OpenStreetMap & Esri (Folium)",
                "⚡ Plotly Native Map (Zero API Key)",
                "📐 2D Coordinate Grid (Paper Fig 4 & 5)"
            ],
            horizontal=True
        )
    with map_ctrl2:
        route_options = ["All Active Routes"] + [f"Route #{r.route_id} ({r.loading_pattern.name[:18]})" for r in sol.routes]
        selected_route_str = st.selectbox("Filter Route Highlight", route_options, index=0)
        active_filter = None
        if selected_route_str != "All Active Routes":
            active_filter = int(selected_route_str.split("#")[1].split()[0])

    with map_ctrl3:
        if "Folium" in map_engine:
            tile_provider = st.selectbox(
                "Map Tile Layer",
                ["OpenStreetMap (Default)", "Esri World Street Map", "Esri Satellite Imagery", "Esri Topographic Map"],
                index=0
            )

    # Render selected map
    if "Folium" in map_engine:
        folium_map = render_folium_map(
            instance=inst,
            routes=sol.routes,
            active_route_filter=active_filter,
            tile_choice=tile_provider if 'tile_provider' in locals() else "OpenStreetMap"
        )
        map_html = folium_map._repr_html_()
        components.html(map_html, height=560, scrolling=False)

    elif "Plotly" in map_engine:
        plotly_fig = render_plotly_map(
            instance=inst,
            routes=sol.routes,
            active_route_filter=active_filter
        )
        st.plotly_chart(plotly_fig, use_container_width=True, key="route_map_plotly")

    else:
        grid_fig = render_euclidean_grid(
            instance=inst,
            routes=sol.routes,
            active_route_filter=active_filter
        )
        st.plotly_chart(grid_fig, use_container_width=True, key="route_map_grid")

    # Detailed Turn-by-Turn Timetable Sheet for selected or all routes
    st.markdown("---")
    st.markdown("#### 📜 Transporter Turn-by-Turn Waybill & Schedule")
    
    inspect_route_id = 1 if active_filter is None else active_filter
    matching_routes = [r for r in sol.routes if r.route_id == inspect_route_id]
    if matching_routes:
        inspected_r = matching_routes[0]
        st.markdown(f"**Viewing Transporter #{inspected_r.route_id}** | Carrier Pattern: **{inspected_r.loading_pattern.name}** | Vehicle Capacity: **{inspected_r.loading_pattern.capacities}** | Total Loaded: **{sum(inspected_r.loaded_demands)} / {inspected_r.loading_pattern.total_capacity}** ({inspected_r.loading_ratio:.1%})")

        step_records = []
        for idx, s in enumerate(inspected_r.steps):
            is_depot = (idx == 0 or idx == len(inspected_r.steps) - 1)
            step_records.append({
                "Step": f"Stop {idx}" if not is_depot else ("RVDC Dispatch" if idx == 0 else "RVDC Return"),
                "Location / Dealer Name": s.node_name,
                "Arrival (min)": f"{s.arrival_time:.1f}",
                "Wait Time (min)": f"{s.wait_time:.1f}",
                "Service Start": f"{s.service_start:.1f}",
                "Departure (min)": f"{s.departure_time:.1f}",
                "Delivered [Sedans, SUVs, MPVs]": f"{s.delivered_demands}" if not is_depot else "—"
            })
        st.dataframe(pd.DataFrame(step_records), hide_index=True, use_container_width=True)


# ------------------------------------------
# TAB 2: HEURISTICS & BEFORE/AFTER COMPARISON
# ------------------------------------------
with tabs[3]:
    st.markdown("### 🔬 Heuristics Optimization Studio & Before/After Comparison")
    st.markdown(
        """
        Compare an unoptimized **Initial Solution** against **Heuristic Optimization**. 
        You can choose from **4 distinct heuristic algorithms** (plus an integrated pipeline) 
        and inspect the exact differences in travel time, deck loading ratio, routes, and improvement trajectory.
        """
    )

    h_col1, h_col2, h_col3 = st.columns([3, 4, 3])

    with h_col1:
        initial_method = st.selectbox(
            "1. Choose Initial Solution Method",
            [
                "Natural Unoptimized Sequence",
                "Greedy Nearest Neighbor",
                "Radial Angular Sweep"
            ],
            index=0,
            help="Constructs the baseline starting point before applying heuristics"
        )

    with h_col2:
        heuristic_choice = st.selectbox(
            "2. Select Heuristic to Apply (4 Options)",
            [
                "Heuristic 1: 2-Opt Edge Uncrossing (Intra-Route)",
                "Heuristic 2: Clarke & Wright Savings Heuristic (Route Merging)",
                "Heuristic 3: Inter-Route Relocate / Shift (Customer Re-allocation)",
                "Heuristic 4: Cross-Route Customer Swap (Osman Heuristic)",
                "Heuristic 5: Full Multi-Heuristic Pipeline (Sequential Local Search)"
            ],
            index=0,
            help="Heuristics specifically adapted for FVTRP loading patterns and time windows"
        )

    with h_col3:
        st.write("")
        st.write("")
        run_h_button = st.button("⚡ Apply Heuristic & Compare", type="primary", use_container_width=True)

    # Run Heuristic when clicked
    if run_h_button or st.session_state.initial_routes is None:
        init_key = "random" if "Natural" in initial_method else ("nearest_neighbor" if "Nearest" in initial_method else "sweep")
        init_rts, init_c, init_desc = build_initial_solution(inst, method=init_key, alpha=alpha)
        st.session_state.initial_routes = init_rts
        st.session_state.initial_method_name = initial_method

        t_h0 = time.time()
        if "2-Opt" in heuristic_choice:
            ref_rts, ref_c, hist = heuristic_2opt(init_rts, inst, alpha=alpha)
        elif "Clarke & Wright" in heuristic_choice:
            ref_rts, ref_c, hist = heuristic_clarke_wright_savings(inst, alpha=alpha)
        elif "Relocate" in heuristic_choice:
            ref_rts, ref_c, hist = heuristic_relocate(init_rts, inst, alpha=alpha)
        elif "Swap" in heuristic_choice:
            ref_rts, ref_c, hist = heuristic_swap(init_rts, inst, alpha=alpha)
        else:
            ref_rts, ref_c, hist = heuristic_pipeline(init_rts, inst, alpha=alpha)

        elapsed_h = time.time() - t_h0
        st.session_state.heuristic_routes = ref_rts
        st.session_state.heuristic_history = hist
        st.session_state.heuristic_name = heuristic_choice
        st.session_state.heuristic_runtime = elapsed_h

    init_routes = st.session_state.initial_routes or []
    ref_routes = st.session_state.heuristic_routes or []
    hist = st.session_state.heuristic_history or []

    # Calculate Comparative Metrics
    init_total_time = sum(r.travel_time for r in init_routes) if init_routes else 0.0
    ref_total_time = sum(r.travel_time for r in ref_routes) if ref_routes else 0.0
    saved_time = init_total_time - ref_total_time
    pct_saved = (saved_time / init_total_time * 100.0) if init_total_time > 0 else 0.0

    init_cars = sum(sum(r.loaded_demands) for r in init_routes) if init_routes else 0
    init_cap = sum(r.loading_pattern.total_capacity for r in init_routes) if init_routes else 0
    init_ratio = (init_cars / init_cap) if init_cap > 0 else 0.0

    ref_cars = sum(sum(r.loaded_demands) for r in ref_routes) if ref_routes else 0
    ref_cap = sum(r.loading_pattern.total_capacity for r in ref_routes) if ref_routes else 0
    ref_ratio = (ref_cars / ref_cap) if ref_cap > 0 else 0.0

    st.markdown("---")
    st.markdown(f"#### 📊 Performance Comparison: Initial vs {st.session_state.heuristic_name}")

    # Comparative KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='font-size: 13px; color: #64748B;'>⏱️ Total Travel Time</div>
                <div style='font-size: 22px; font-weight: 700; color: #0F172A;'>
                    {ref_total_time:.0f} min <span style='font-size: 14px; color: #94A3B8; text-decoration: line-through;'>{init_total_time:.0f}</span>
                </div>
                <div class='savings-badge'>↓ Saved {saved_time:.0f} min ({pct_saved:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='font-size: 13px; color: #64748B;'>🚛 Transporters Deployed</div>
                <div style='font-size: 22px; font-weight: 700; color: #0F172A;'>
                    {len(ref_routes)} Carriers <span style='font-size: 14px; color: #94A3B8;'>({len(init_routes)} initial)</span>
                </div>
                <div style='font-size: 12px; color: #059669; margin-top: 4px;'><b>{'Consolidated Fleet' if len(ref_routes) < len(init_routes) else 'Maintained Fleet'}</b></div>
            </div>
            """, unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='font-size: 13px; color: #64748B;'>🎯 Fleet Deck Loading Ratio</div>
                <div style='font-size: 22px; font-weight: 700; color: #0F172A;'>
                    {ref_ratio:.1%} <span style='font-size: 14px; color: #94A3B8;'>({init_ratio:.1%})</span>
                </div>
                <div style='font-size: 12px; color: #2563EB; margin-top: 4px;'>Deck Efficiency: <b>+{((ref_ratio - init_ratio)*100):.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='font-size: 13px; color: #64748B;'>⚡ Heuristic Speed</div>
                <div style='font-size: 22px; font-weight: 700; color: #0F172A;'>
                    {st.session_state.get('heuristic_runtime', 0.05):.3f} s
                </div>
                <div style='font-size: 12px; color: #64748B; margin-top: 4px;'><b>{len(hist)} Local Search Passes</b></div>
            </div>
            """, unsafe_allow_html=True
        )

    # Visual Route Comparison Toggle & Side-by-Side Maps
    st.markdown("---")
    st.markdown("#### 🗺️ Visual Route Differences: Before vs After Heuristic")

    map_view_choice = st.radio(
        "Display Mode for Map Comparison:",
        ["Side-by-Side Dual View", "After Heuristics (Optimal)", "Initial Baseline (Unoptimized)"],
        horizontal=True
    )

    if map_view_choice == "Side-by-Side Dual View":
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.markdown(f"**Initial Solution ({st.session_state.initial_method_name})** — Cost: **{init_total_time:.0f} min**")
            p_init = render_plotly_map(inst, init_routes)
            p_init.update_layout(height=420)
            st.plotly_chart(p_init, use_container_width=True, key="heuristic_initial_map")

        with m_c2:
            st.markdown(f"**After Heuristic ({st.session_state.heuristic_name.split(':')[0]})** — Cost: **{ref_total_time:.0f} min**")
            p_ref = render_plotly_map(inst, ref_routes)
            p_ref.update_layout(height=420)
            st.plotly_chart(p_ref, use_container_width=True, key="heuristic_improved_map")

    elif map_view_choice == "After Heuristics (Optimal)":
        p_ref = render_plotly_map(inst, ref_routes)
        st.plotly_chart(p_ref, use_container_width=True, key="heuristic_after_map")

    else:
        p_init = render_plotly_map(inst, init_routes)
        st.plotly_chart(p_init, use_container_width=True, key="heuristic_before_map")

    # Heuristic Improvement Trajectory & Per-Route Comparison
    st.markdown("---")
    t_c1, t_c2 = st.columns([6, 4])

    with t_c1:
        st.markdown("#### 📉 Heuristic Step-by-Step Optimization Trajectory")
        st.caption("Detailed progression of travel cost reduction across successive local search moves.")
        h_prog_fig = plot_heuristic_progress(hist)
        st.plotly_chart(h_prog_fig, use_container_width=True, key="heuristic_progress")

    with t_c2:
        st.markdown("#### 📊 Route-by-Route Travel Time Delta")
        st.caption("Comparison of initial travel duration vs post-heuristic travel duration.")
        h_bar_fig = plot_before_after_comparison(init_routes, ref_routes)
        st.plotly_chart(h_bar_fig, use_container_width=True, key="heuristic_before_after")

    # Route Delta Table
    st.markdown("---")
    st.markdown("#### 📋 Detailed Transporter Route Delta Table")
    route_delta_rows = []
    max_r = max(len(init_routes), len(ref_routes))
    for i in range(max_r):
        r_init = init_routes[i] if i < len(init_routes) else None
        r_ref = ref_routes[i] if i < len(ref_routes) else None

        init_seq = " ➔ ".join(str(d) for d in r_init.dealer_ids) if r_init else "—"
        ref_seq = " ➔ ".join(str(d) for d in r_ref.dealer_ids) if r_ref else "—"

        t_init = f"{r_init.travel_time:.0f}" if r_init else "0"
        t_ref = f"{r_ref.travel_time:.0f}" if r_ref else "0"
        t_diff = (r_init.travel_time - r_ref.travel_time) if (r_init and r_ref) else (-(r_ref.travel_time if r_ref else 0))

        route_delta_rows.append({
            "Transporter": f"Route #{i+1}",
            "Initial Sequence": init_seq,
            "Refined Sequence (Heuristic)": ref_seq,
            "Initial Time (min)": t_init,
            "Refined Time (min)": t_ref,
            "Saved (min)": f"{t_diff:+.0f}" if t_diff != 0 else "0",
            "Initial Ratio": f"{r_init.loading_ratio:.1%}" if r_init else "—",
            "Refined Ratio": f"{r_ref.loading_ratio:.1%}" if r_ref else "—"
        })

    st.dataframe(pd.DataFrame(route_delta_rows), hide_index=True, use_container_width=True)

    # Button to set this heuristic solution as active main solution
    if st.button("🌟 Set this Heuristic Solution as Active Dashboard Plan", use_container_width=True):
        st.session_state.solution = FVTRPSolution(
            instance_name=inst.name,
            routes=ref_routes,
            total_travel_time=ref_total_time,
            total_vehicles_delivered=ref_cars,
            total_capacity_deployed=ref_cap,
            overall_loading_ratio=ref_ratio,
            computation_time_sec=st.session_state.get('heuristic_runtime', 0.05),
            algorithm_name=st.session_state.heuristic_name
        )
        st.success("Set as active dashboard solution! Check Tab 1 and Tab 2.")
        st.rerun()


# ------------------------------------------
# TAB 3: STOWAGE DECK PLAN & VEHICLE ALLOCATION (TABLE 5)
# ------------------------------------------
with tabs[1]:
    st.markdown("### 🚚 Transporter Loading Patterns & Stowage Breakdown")
    st.markdown(
        """
        In automotive logistics, finished vehicles cannot simply be treated as 1D volume. 
        Transporters feature complex hydraulic decks where vehicle combinations must match discovered 
        **Loading Patterns** $\\Theta_{k,e,o}$. The table below matches **Table 5 (The stowage result)** from the research paper.
        """
    )

    # Recreate Paper Table 5 exactly
    table5_rows = []
    for r in sol.routes:
        table5_rows.append({
            "Route": r.route_id,
            "Loaded Type 1 (Sedan)": r.loaded_demands[0],
            "Loaded Type 2 (SUV)": r.loaded_demands[1],
            "Loaded Type 3 (MPV)": r.loaded_demands[2],
            "Total Loaded": sum(r.loaded_demands),
            "Pattern No.": r.loading_pattern.id,
            "Pattern Type 1 Cap": r.loading_pattern.capacities[0],
            "Pattern Type 2 Cap": r.loading_pattern.capacities[1],
            "Pattern Type 3 Cap": r.loading_pattern.capacities[2],
            "Pattern Total Cap": r.loading_pattern.total_capacity,
            "Loading Ratio η(r,e)": f"{r.loading_ratio:.3f}",
            "Travel Cost (min)": f"{r.travel_time:.0f}",
            "No. Dealers Visited": len(r.dealer_ids),
            "Dealer Sequence": " ➔ ".join(str(did) for did in r.dealer_ids)
        })

    df_table5 = pd.DataFrame(table5_rows)
    st.dataframe(df_table5, hide_index=True, use_container_width=True)

    # Average summaries matching Table 5 bottom row
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.info(f"**Average Loading Ratio:** {sol.overall_loading_ratio:.3f} (Empty rate: {1.0 - sol.overall_loading_ratio:.1%})")
    with col_t2:
        avg_cost = sum(r.travel_time for r in sol.routes) / len(sol.routes) if sol.routes else 0
        st.info(f"**Average Cost per Route:** {avg_cost:.1f} minutes")
    with col_t3:
        st.info(f"**Total Fleet Vehicle Decks Deployed:** {len(sol.routes)} Transporters")

    st.markdown("---")
    st.subheader("📊 Vehicle Deck Stowage Diagram (Upper vs Lower Deck)")
    selected_deck_route = st.selectbox(
        "Select Transporter for Deck Schematic View:",
        [f"Route #{r.route_id} ({r.loading_pattern.name})" for r in sol.routes]
    )
    sel_route_id = int(selected_deck_route.split("#")[1].split()[0])
    sel_r = [r for r in sol.routes if r.route_id == sel_route_id][0]

    # Transporter deck schematic
    deck_fig = plot_transporter_deck_schematic(sel_r)
    st.plotly_chart(deck_fig, use_container_width=True, key="transporter_deck_schematic")

    # Stacked stowage distribution chart
    st.markdown("---")
    breakdown_fig = plot_stowage_breakdown(sol.routes)
    st.plotly_chart(breakdown_fig, use_container_width=True, key="stowage_breakdown")


# ------------------------------------------
# TAB 4: LIVE DECISION VARIABLE & CAPACITY STUDIO
# ------------------------------------------
with tabs[0]:
    st.markdown("### ✍️ Interactive Decision Variables & Capacity Studio")
    st.caption("Add dealers, change dealer orders, and adjust transporter capacities directly on the page.")

    # SECTION A: CARRIER CAPACITY CONTROLS
    st.markdown("#### 1. 🚚 Change Carrier / Transporter Capacities")
    st.markdown(
        """
        Adjust the capacity limits of each car transporter pattern. 
        Each pattern represents an arrangement of hydraulic decks that can accommodate a specific combination of Sedans, SUVs, and Vans.
        """
    )

    cap_col1, cap_col2 = st.columns([7, 3])

    with cap_col2:
        st.markdown("**Quick Capacity Adjustments**")
        scale_val = st.slider("Quick Scale All Capacities", 0.5, 2.5, 1.0, 0.1, key="quick_scale_slider")
        if st.button("Apply Quick Scale to Fleet", type="secondary", use_container_width=True):
            for p in inst.loading_patterns:
                p.capacities = [max(1, int(round(c * scale_val))) for c in p.capacities]
            st.session_state.solution = None
            st.session_state.initial_routes = None
            st.session_state.heuristic_routes = None
            st.success(f"Scaled all transporter capacities by {scale_val}x!")
            st.rerun()

    with cap_col1:
        pat_data = []
        for p in inst.loading_patterns:
            pat_data.append({
                "Pattern ID": p.id,
                "Carrier Pattern Name": p.name,
                "Type 1 Cap (Sedan)": p.capacities[0],
                "Type 2 Cap (SUV)": p.capacities[1],
                "Type 3 Cap (MPV)": p.capacities[2],
                "Total Capacity": p.total_capacity,
                "Transporter Description": p.description
            })
        df_pat_edit = pd.DataFrame(pat_data)

        edited_pat_df = st.data_editor(
            df_pat_edit,
            num_rows="dynamic",
            key="pat_editor",
            use_container_width=True,
            column_config={
                "Pattern ID": st.column_config.NumberColumn("Pattern ID", disabled=True),
                "Total Capacity": st.column_config.NumberColumn("Total Capacity", disabled=True),
                "Type 1 Cap (Sedan)": st.column_config.NumberColumn("Sedan Cap", min_value=1, max_value=25, step=1),
                "Type 2 Cap (SUV)": st.column_config.NumberColumn("SUV Cap", min_value=1, max_value=25, step=1),
                "Type 3 Cap (MPV)": st.column_config.NumberColumn("MPV Cap", min_value=1, max_value=25, step=1)
            }
        )

    st.markdown("---")
    # SECTION B: ADD DEALERS
    st.markdown("#### 2. ➕ Add a New Dealer to Network")
    with st.form("add_dealer_form", clear_on_submit=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            new_id = max(d.id for d in inst.dealers) + 1
            new_name = st.text_input("Dealer Name", f"New Auto Dealership #{new_id}")
            new_handling = st.number_input("Unload Time P (min)", min_value=1.0, max_value=60.0, value=2.0)
        with fcol2:
            new_lat = st.number_input("Latitude", value=round(inst.depot.lat + np.random.uniform(-0.15, 0.15), 5), format="%.5f")
            new_lon = st.number_input("Longitude", value=round(inst.depot.lon + np.random.uniform(-0.15, 0.15), 5), format="%.5f")
        with fcol3:
            st.markdown("**Order Quantities (Vehicles)**")
            dem_t1 = st.number_input("Type 1: Sedans", min_value=0, max_value=10, value=2)
            dem_t2 = st.number_input("Type 2: SUVs", min_value=0, max_value=10, value=1)
            dem_t3 = st.number_input("Type 3: MPV/Van", min_value=0, max_value=10, value=1)
        with fcol4:
            st.markdown("**Time Window [L, U]**")
            win_l = st.number_input("Earliest Arrival L (min)", min_value=0.0, value=1.0)
            win_u = st.number_input("Latest Arrival U (min)", min_value=10.0, value=10000.0)
            submitted = st.form_submit_button("➕ Add Dealer to Network", use_container_width=True, type="primary")

        if submitted:
            new_dealer = Dealer(
                id=new_id,
                name=new_name,
                lat=new_lat,
                lon=new_lon,
                demands=[dem_t1, dem_t2, dem_t3],
                time_window=(win_l, win_u),
                handling_time=new_handling
            )
            inst.dealers.append(new_dealer)
            inst.compute_distance_matrix_if_needed()
            st.success(f"Added dealer '{new_name}' (ID: {new_id}) successfully!")
            st.rerun()

    st.markdown("---")
    # SECTION C: EDIT DEALER ORDERS DIRECTLY
    st.markdown("#### 3. 📝 Edit Dealer Orders & Demands Directly")
    st.caption("You can edit dealer names, coordinates, vehicle demands, and time windows directly in the table below:")

    dealer_data = []
    for d in inst.dealers:
        dealer_data.append({
            "ID": d.id,
            "Name": d.name,
            "Type 1 (Sedan)": d.demands[0],
            "Type 2 (SUV)": d.demands[1],
            "Type 3 (MPV)": d.demands[2],
            "Earliest L (min)": d.time_window[0],
            "Latest U (min)": d.time_window[1],
            "Unload P (min)": d.handling_time,
            "Latitude": d.lat,
            "Longitude": d.lon
        })
    df_dealers_edit = pd.DataFrame(dealer_data)

    edited_dealers_df = st.data_editor(
        df_dealers_edit,
        num_rows="dynamic",
        key="dealers_editor",
        use_container_width=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Type 1 (Sedan)": st.column_config.NumberColumn("Type 1 (Sedan)", min_value=0, max_value=20, step=1),
            "Type 2 (SUV)": st.column_config.NumberColumn("Type 2 (SUV)", min_value=0, max_value=20, step=1),
            "Type 3 (MPV)": st.column_config.NumberColumn("Type 3 (MPV)", min_value=0, max_value=20, step=1),
        }
    )

    # Master Button to save all table edits and re-optimize
    st.markdown("---")
    if st.button("💾 Apply All Changes (Capacities + Orders) & Re-Optimize Routes", type="primary", use_container_width=True):
        updated_dealers = []
        for _, row in edited_dealers_df.iterrows():
            updated_dealers.append(Dealer(
                id=int(row["ID"]),
                name=str(row["Name"]),
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                demands=[int(row["Type 1 (Sedan)"]), int(row["Type 2 (SUV)"]), int(row["Type 3 (MPV)"])],
                time_window=(float(row["Earliest L (min)"]), float(row["Latest U (min)"])),
                handling_time=float(row["Unload P (min)"])
            ))
        inst.dealers = updated_dealers

        updated_patterns = []
        for _, row in edited_pat_df.iterrows():
            c1, c2, c3 = int(row["Type 1 Cap (Sedan)"]), int(row["Type 2 Cap (SUV)"]), int(row["Type 3 Cap (MPV)"])
            updated_patterns.append(LoadingPattern(
                id=int(row["Pattern ID"]),
                name=str(row["Carrier Pattern Name"]),
                capacities=[c1, c2, c3],
                description=str(row.get("Transporter Description", ""))
            ))
        inst.loading_patterns = updated_patterns

        inst.compute_distance_matrix_if_needed()

        with st.spinner("Re-optimizing with updated capacities and dealer demands..."):
            new_sol = run_ea_solver(
                instance=inst,
                alpha=alpha,
                pop_size=pop_size,
                generations=generations,
                cx_prob=crossover_p,
                mut_prob=mutation_p
            )
            st.session_state.solution = new_sol
            st.session_state.initial_routes = None
            st.session_state.heuristic_routes = None
            st.success("Re-optimization complete! Updated routes, deck stowage, and maps.")
            st.rerun()


# ------------------------------------------
# TAB 5: DISPATCH TIMETABLE & CONVERGENCE
# ------------------------------------------
with tabs[4]:
    st.markdown("### 📊 Operational Analytics & Algorithm Performance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Algorithm 1 Convergence History")
        st.caption("Optimal fitness (best) vs Population average fitness across evolution generations (reproducing **Figure 6**).")
        conv_fig = plot_convergence_curve(sol.convergence_history)
        st.plotly_chart(conv_fig, use_container_width=True, key="ea_convergence")

    with c2:
        st.markdown("#### ⏱️ Transporter Dispatch Schedule")
        st.caption("Gantt timeline depicting transit legs and unloading handling times at dealer locations.")
        gantt_fig = plot_timetable_gantt(sol.routes)
        st.plotly_chart(gantt_fig, use_container_width=True, key="dispatch_gantt")

    st.markdown("---")
    st.markdown("#### 📋 Transporter Fleet Capacity Utilization")
    util_records = []
    for r in sol.routes:
        util_records.append({
            "Transporter": f"Carrier #{r.transporter_id}",
            "Assigned Pattern": r.loading_pattern.name,
            "Total Cap": r.loading_pattern.total_capacity,
            "Loaded Units": sum(r.loaded_demands),
            "Empty Slots": r.empty_slots,
            "Loading Efficiency": f"{r.loading_ratio:.1%}",
            "Travel Time": f"{r.travel_time:.0f} min",
            "Route Status": "✅ Feasible & Verified" if r.is_feasible else "❌ Infeasible"
        })
    st.dataframe(pd.DataFrame(util_records), hide_index=True, use_container_width=True)


# ------------------------------------------
# TAB 6: STEP 3 DSS DOCUMENTATION & EXPORT
# ------------------------------------------
with tabs[5]:
    st.markdown("### 📑 Step 3 Prototype Decision Support System Documentation")
    st.markdown(
        """
        As requested in **Scope Step 3** of the Project Overview:
        > *"Prepare a 'prototype' decision support system (web-based) to show how you will implement it in practice:
        > Data input, Running the model, Visualizing the results."*
        """
    )

    doc_col1, doc_col2 = st.columns(2)

    with doc_col1:
        st.markdown("#### 🎯 For Business Users")
        st.markdown(
            """
            - **Real-World Case Background:** Implemented for finished-vehicle logistics (FVL), based on Anji Logistics (Shanghai RVDC).
            - **Business Value:** Replaces manual driver loading guesswork with mathematically optimal transporter routing and discovered loading patterns.
            - **Double Optimization:**
              1. **Reduces Fuel & Transportation Costs:** Generates shortest feasible routes with time windows.
              2. **Minimizes Empty Car Carrier Capacity:** Maximizes multi-deck loading ratios (up to 100% on key routes).
            - **Decision Agility:** Fleet managers can add dealers, adjust car orders, and modify carrier capacities interactively.
            """
        )

        st.markdown("#### 🧪 For Testers & Quality Assurance")
        st.caption(
            "Eighteen acceptance checks using the paper benchmark and synthetic fleet scenarios. "
            "Each result is based on an executed solver check."
        )
        if st.button("▶ Run 18 Validation Test Cases", use_container_width=True):
            with st.spinner("Executing validation tests..."):
                st.session_state.test_results = run_dashboard_test_cases()

        test_results = st.session_state.test_results
        if test_results is not None:
            test_df = pd.DataFrame(test_results)
            success_count = int((test_df["Status"] == "SUCCESS").sum())
            failure_count = int((test_df["Status"] == "FAILURE").sum())
            total_count = len(test_df)
            if failure_count == 0:
                st.success(f"Overall result: SUCCESS — {success_count}/{total_count} tests passed.")
            else:
                st.error(
                    f"Overall result: FAILURE — {success_count}/{total_count} tests passed; "
                    f"{failure_count} require attention."
                )

            success_col, failure_col = st.columns(2)
            with success_col:
                st.metric("Successful Tests", success_count)
            with failure_col:
                st.metric("Failed Tests", failure_count)

            overview = test_df[["ID", "Test Case", "Status", "Observed"]].copy()
            overview["Status"] = overview["Status"].map(
                lambda status: "✅ SUCCESS" if status == "SUCCESS" else "❌ FAILURE"
            )
            st.dataframe(overview, use_container_width=True, hide_index=True)

            failures = [
                test for test in test_results if test["Status"] == "FAILURE"
            ]
            if failures:
                st.markdown("#### ⚠️ Failure findings")
                for test in failures:
                    st.error(
                        f"{test['ID']} — {test['Test Case']}: {test['Observed']}"
                    )

            with st.expander("View all test setup and acceptance criteria", expanded=False):
                for test in test_results:
                    st.markdown(f"**{test['ID']} — {test['Test Case']}**")
                    st.markdown(f"- **Setup:** {test['Setup']}")
                    st.markdown(f"- **Acceptance criteria:** {test['Acceptance Criteria']}")
                    st.markdown(f"- **Observed:** {test['Observed']}")
                    result_label = (
                        "✅ SUCCESS" if test["Status"] == "SUCCESS" else "❌ FAILURE"
                    )
                    st.markdown(f"- **Result:** {result_label}")

    with doc_col2:
        st.markdown("#### 💻 For Developers (System Architecture)")
        st.markdown(
            """
            - **Mathematical Model:** Mixed-Integer Linear Programming (MILP) formulation (Equations 1–17 in paper):
              - Flow balance constraints (Eqs. 3–6)
              - Subtour elimination & time windows (Eqs. 7–9, 12)
              - Loading pattern capacity constraints (Eqs. 10–11)
            - **Prins' Split Algorithm on DAG (Liu et al. 2008):**
              - Chromosome: dealer sequence permutation $\\pi = (\\pi_1, \\dots, \\pi_n)$.
              - Directed Acyclic Graph: node $i$ to $j$ represents route segment.
              - Subproblem evaluator verifies pattern capacity $\\sum D_{c,o} \\le \\Theta_{e,o}$, time windows $[L, U]$, and duration $P_c$.
              - Bellman dynamic programming finds exact minimum-cost partition in $O(n^2)$ time.
            - **Genetic Operators:** Tournament selection, PMX crossover, gene swap mutation, 2-opt route polishing.
            - **Four Heuristic Local Search Operators:**
              1. *2-Opt Intra-Route Edge Uncrossing*
              2. *Clarke & Wright Savings Route Merge*
              3. *Inter-Route Customer Relocate*
              4. *Cross-Route Customer Swap*
            """
        )

    st.markdown("---")
    st.subheader("💾 Solution Export")
    export_data = {
        "scenario": inst.name,
        "city": inst.city_name,
        "total_travel_time_min": sol.total_travel_time,
        "overall_loading_ratio": sol.overall_loading_ratio,
        "total_vehicles_delivered": sol.total_vehicles_delivered,
        "routes": [
            {
                "route_id": r.route_id,
                "dealers": r.dealer_ids,
                "loading_pattern_id": r.loading_pattern.id,
                "loading_pattern_name": r.loading_pattern.name,
                "travel_time_min": r.travel_time,
                "loading_ratio": r.loading_ratio,
                "loaded_demands": r.loaded_demands
            }
            for r in sol.routes
        ]
    }
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Solution as JSON",
        data=json_str,
        file_name="fvtrp_optimized_solution.json",
        mime="application/json"
    )
