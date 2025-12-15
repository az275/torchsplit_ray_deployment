import gurobipy as gp
from gurobipy import GRB

# Define models, configs, and nodes
models = ['A', 'B', 'C']
configs = [8, 16, 24, 32, 40]
nodes = ['GPU0', 'GPU1', 'GPU2', 'GPU3']

# Throughput dictionary
throughput = {
    'A': {8: 2627},
    'B': {8: 1728},
    'C': {8: 4326},
}

valid_layouts = [
    [40],
    [8, 8, 8, 8, 8]
]

def solve_leximin(locked_lower_bounds):
    m = gp.Model("Leximin_Level")
    x, y = {}, {}
    T_m = {}
    Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

    for model in models:
        for node in nodes:
            for c in configs:
                if c in throughput[model]:
                    x[model, node, c] = m.addVar(vtype=GRB.INTEGER, name=f"x_{model}_{node}_{c}")

    for node in nodes:
        for lid, layout in enumerate(valid_layouts):
            y[node, lid] = m.addVar(vtype=GRB.BINARY, name=f"y_{node}_{lid}")

    for model in models:
        T_m[model] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"T_{model}")
        m.addConstr(
            T_m[model] == gp.quicksum(
                x[model, node, c] * throughput[model][c]
                for node in nodes for c in configs if (model, node, c) in x
            )
        )

        if model in locked_lower_bounds:
            m.addConstr(T_m[model] >= locked_lower_bounds[model])
        else:
            m.addConstr(Z <= T_m[model])

        m.addConstr(gp.quicksum(
            x[model, node, c] for node in nodes for c in configs if (model, node, c) in x
        ) >= 1)

    for node in nodes:
        m.addConstr(gp.quicksum(y[node, lid] for lid in range(len(valid_layouts))) == 1)
        for c in configs:
            m.addConstr(
                gp.quicksum(
                    x[model, node, c] for model in models if (model, node, c) in x
                ) <= gp.quicksum(
                    y[node, lid] * layout.count(c)
                    for lid, layout in enumerate(valid_layouts)
                )
            )

    m.setObjective(Z, GRB.MAXIMIZE)
    m.setParam("OutputFlag", 0)
    m.optimize()

    throughput_vals = {model: T_m[model].X for model in models}
    assignment = {
        (model, node, c): int(x[model, node, c].X)
        for model in models for node in nodes for c in configs
        if (model, node, c) in x and x[model, node, c].X > 0.5
    }
    return throughput_vals, assignment

# Leximin loop
locked = {}
for _ in range(len(models)):
    T_vals, assignment = solve_leximin(locked)
    unlocked = [m for m in models if m not in locked]
    if not unlocked:
        break
    min_model = min(unlocked, key=lambda m: T_vals[m])
    locked[min_model] = T_vals[min_model]

# Print results
print(f"\nPipeline Throughput: {min(locked.values()):.2f}")
print(" Leximin Model Throughputs:")
for model in models:
    print(f" {model}: {locked[model]:.2f}")


print("\nAssignments:")
for (model, node, c), count in assignment.items():
    print(f" - Model {model} assigned {count}x to {node} with MIG {c}GB")
