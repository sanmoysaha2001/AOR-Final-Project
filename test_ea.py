import numpy as np
import random
import time

def test_ea():
    # Use same time_matrix, demands, etc.
    from test_split import time_matrix, demands, handling_time, time_windows, loading_patterns, split_chromosome
    
    dealers = list(demands.keys())
    n_dealers = len(dealers)
    
    # EA parameters
    pop_size = 40
    n_gen = 100
    crossover_prob = 0.8
    mutation_prob = 0.3
    
    # Initialize population: random permutations of dealers
    pop = []
    for _ in range(pop_size):
        p = list(dealers)
        random.shuffle(p)
        pop.append(p)
    
    def evaluate(ind):
        routes, cost = split_chromosome(ind, loading_patterns, time_matrix, demands, handling_time, time_windows, alpha=1.0)
        return cost if routes is not None else 1e9

    pop_fitness = [evaluate(ind) for ind in pop]
    
    best_fitness = min(pop_fitness)
    best_ind = pop[pop_fitness.index(best_fitness)]
    
    print(f"Initial best cost: {best_fitness}")
    
    t0 = time.time()
    for gen in range(n_gen):
        new_pop = []
        # Elitism: keep top 2
        sorted_indices = np.argsort(pop_fitness)
        new_pop.append(list(pop[sorted_indices[0]]))
        new_pop.append(list(pop[sorted_indices[1]]))
        
        while len(new_pop) < pop_size:
            # Tournament selection (size 3)
            def tournament():
                candidates = random.sample(range(pop_size), 3)
                best_c = min(candidates, key=lambda idx: pop_fitness[idx])
                return pop[best_c]
            
            p1 = tournament()
            p2 = tournament()
            
            # PMX Crossover
            if random.random() < crossover_prob:
                c1, c2 = pmx_crossover(p1, p2)
            else:
                c1, c2 = list(p1), list(p2)
            
            # Mutation: swap or 2-opt inversion
            if random.random() < mutation_prob:
                mutate(c1)
            if random.random() < mutation_prob:
                mutate(c2)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        pop = new_pop
        pop_fitness = [evaluate(ind) for ind in pop]
        
        current_best = min(pop_fitness)
        if current_best < best_fitness:
            best_fitness = current_best
            best_ind = pop[pop_fitness.index(best_fitness)]
            print(f"Gen {gen}: New Best Cost = {best_fitness}")
    
    elapsed = time.time() - t0
    print(f"EA finished in {elapsed:.2f}s. Final Best Cost: {best_fitness}")
    routes, _ = split_chromosome(best_ind, loading_patterns, time_matrix, demands, handling_time, time_windows, alpha=1.0)
    for idx, r in enumerate(routes):
        print(f"  Route {idx+1}: Dealers {r['dealers']}, Cost: {r['travel_cost']}, Pattern: {r['pattern']['id']}, Demands: {r['demands']}")

def pmx_crossover(p1, p2):
    size = len(p1)
    cxpoint1 = random.randint(0, size - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, size - 1)
    
    c1 = [-1] * size
    c2 = [-1] * size
    
    # Copy slice
    c1[cxpoint1:cxpoint2+1] = p1[cxpoint1:cxpoint2+1]
    c2[cxpoint1:cxpoint2+1] = p2[cxpoint1:cxpoint2+1]
    
    # Map remaining
    for c, parent, other_parent in [(c1, p2, p1), (c2, p1, p2)]:
        for i in range(cxpoint1, cxpoint2 + 1):
            val = parent[i]
            if val not in c:
                pos = i
                while cxpoint1 <= pos <= cxpoint2:
                    val_in_other = other_parent[pos]
                    pos = parent.index(val_in_other)
                c[pos] = val
        for i in range(size):
            if c[i] == -1:
                c[i] = parent[i]
    return c1, c2

def mutate(ind):
    # Swap mutation
    if random.random() < 0.5:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    else:
        # Inversion mutation
        i, j = sorted(random.sample(range(len(ind)), 2))
        ind[i:j+1] = reversed(ind[i:j+1])

if __name__ == "__main__":
    test_ea()
