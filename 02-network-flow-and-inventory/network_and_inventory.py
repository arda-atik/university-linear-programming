"""
01LIP - Linear Programming (CTU FNSPE)
Module 02: Maximum Network Flow & Multi-Period Production-Inventory Planning
Reference: Lecture Notes Section 1.6.2 (Figure 1.2) & Section 1.6.3 (Figure 1.3)
"""

import pulp


def solve_max_network_flow(tol: float = 1e-7) -> None:
    """
    Finds the maximum steady-state data transmission rate (MB/s) from source 's'
    to destination 't' through intermediate nodes 'a' and 'b'.
    """
    capacities = {
        ("s", "a"): 4.0,
        ("s", "b"): 2.0,
        ("a", "b"): 1.0,
        ("a", "t"): 2.0,
        ("b", "t"): 4.0,
    }
    edges = list(capacities.keys())

    model = pulp.LpProblem("Max_Network_Flow", pulp.LpMaximize)

    # Flow variables x_uv with bounds 0 <= x_uv <= capacity(u, v)
    flow = {
        (u, v): pulp.LpVariable(f"x_{u}{v}", lowBound=0.0, upBound=cap)
        for (u, v), cap in capacities.items()
    }

    # Objective: maximize total flow leaving source 's'
    model += flow[("s", "a")] + flow[("s", "b")], "Total_Throughput_MBps"

    # Flow conservation at intermediate nodes 'a' and 'b'
    model += flow[("s", "a")] == flow[("a", "b")] + flow[("a", "t")], "Flow_Balance_Node_a"
    model += flow[("s", "b")] + flow[("a", "b")] == flow[("b", "t")], "Flow_Balance_Node_b"

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    max_rate = pulp.value(model.objective)

    print("=== 1. Maximum Network Flow Problem ===")
    print(f"Solver status   : {pulp.LpStatus[status]}")
    for (u, v) in edges:
        print(f"  {u} -> {v} : {pulp.value(flow[(u, v)]):.2f} / {capacities[(u, v)]:.2f} MB/s")
    print(f"Max throughput  : {max_rate:.2f} MB/s")

    # Validate node conservation and source-sink flow equality
    inflow_t = pulp.value(flow[("a", "t")]) + pulp.value(flow[("b", "t")])
    res_a = abs(pulp.value(flow[("s", "a")]) - (pulp.value(flow[("a", "b")]) + pulp.value(flow[("a", "t")])))
    res_b = abs((pulp.value(flow[("s", "b")]) + pulp.value(flow[("a", "b")])) - pulp.value(flow[("b", "t")]))

    assert abs(max_rate - inflow_t) <= tol, "Source outflow != sink inflow"
    assert max(res_a, res_b) <= tol, "Node flow conservation residual exceeded tolerance"
    print(f"Validation OK   : sink inflow={inflow_t:.2f} MB/s, max node residual={max(res_a, res_b):.2e}\n")


def solve_production_inventory(tol: float = 1e-7) -> None:
    """
    Multi-period production and inventory planning with linearized absolute
    production-change penalties |x_i - x_{i-1}| <= u_i.
    """
    T = 2
    periods = range(1, T + 1)
    demand = {1: 2.0, 2: 6.0}   # metric tons
    x0 = 2.0                    # initial production rate (tons)
    s0 = 0.0                    # initial inventory (tons)
    sT = 0.0                    # required ending inventory (tons)
    h = 1.0                     # holding cost (thousand CZK per ton)
    k = 2.0                     # production change cost (thousand CZK per ton)

    model = pulp.LpProblem("Production_Inventory_Planning", pulp.LpMinimize)

    x = {i: pulp.LpVariable(f"prod_{i}", lowBound=0.0) for i in periods}
    s = {i: pulp.LpVariable(f"inv_{i}", lowBound=0.0) for i in range(0, T + 1)}
    u = {i: pulp.LpVariable(f"change_{i}", lowBound=0.0) for i in periods}

    # Boundary inventory conditions
    model += s[0] == s0, "Initial_Inventory"
    model += s[T] == sT, "Final_Inventory"

    # Objective: minimize holding cost + production adjustment cost
    model += (
        h * pulp.lpSum(s[i] for i in range(1, T))
        + k * pulp.lpSum(u[i] for i in periods),
        "Total_Cost_Thousand_CZK",
    )

    for i in periods:
        prev_x = x0 if i == 1 else x[i - 1]
        # Inventory balance: s_{i-1} + x_i - s_i == d_i
        model += s[i - 1] + x[i] - s[i] == demand[i], f"Inventory_Balance_t{i}"
        # Linearized absolute value: -u_i <= x_i - x_{i-1} <= u_i
        model += x[i] - prev_x <= u[i], f"Prod_Increase_t{i}"
        model += prev_x - x[i] <= u[i], f"Prod_Decrease_t{i}"

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    opt_cost = pulp.value(model.objective)

    print("=== 2. Multi-Period Production & Inventory Plan ===")
    print(f"Solver status : {pulp.LpStatus[status]}")
    for i in periods:
        print(
            f"  Month {i}: prod x_{i} = {pulp.value(x[i]):.2f} t, "
            f"end-inv s_{i} = {pulp.value(s[i]):.2f} t, "
            f"|dx_{i}| = {pulp.value(u[i]):.2f} t"
        )
    print(f"Minimum cost  : {opt_cost:.2f} thousand CZK")

    # Validate inventory balance and exactness of u_i = |x_i - x_{i-1}|
    for i in periods:
        prev_x = x0 if i == 1 else pulp.value(x[i - 1])
        bal = pulp.value(s[i - 1]) + pulp.value(x[i]) - pulp.value(s[i])
        assert abs(bal - demand[i]) <= tol, f"Balance violated in period {i}"
        assert abs(pulp.value(u[i]) - abs(pulp.value(x[i]) - prev_x)) <= tol
    print("Validation OK : all inventory balances and linearized changes verified.\n")


if __name__ == "__main__":
    solve_max_network_flow()
    solve_production_inventory()
