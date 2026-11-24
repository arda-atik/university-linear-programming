"""
01LIP - Linear Programming (CTU FNSPE)
Module 01: LP Formulation, Blending Problem & Variable Transformations
Reference: Lecture Notes Section 1.2 (Example 1.8) & Section 1.6.1 (Table 1.1)
"""

import pulp


def solve_mixture_problem(tol: float = 1e-7) -> None:
    """
    Least-cost 1 kg mixture problem from three raw materials (A, B, C)
    subject to minimum content requirements for components P and Q.
    """
    materials = ["A", "B", "C"]
    cost_czk = {"A": 8.0, "B": 6.0, "C": 5.0}      # CZK per kg
    comp_p = {"A": 200.0, "B": 100.0, "C": 0.0}    # grams per kg
    comp_q = {"A": 0.0, "B": 100.0, "C": 300.0}    # grams per kg

    model = pulp.LpProblem("Least_Cost_Mixture", pulp.LpMinimize)

    # Decision variables: mass of each raw material in kg (x_i >= 0)
    x = pulp.LpVariable.dicts("mass_kg", materials, lowBound=0.0, cat="Continuous")

    # Objective: minimize total cost in CZK
    model += pulp.lpSum(cost_czk[m] * x[m] for m in materials), "Total_Cost_CZK"

    # Constraints
    model += pulp.lpSum(x[m] for m in materials) == 1.0, "Total_Mass_1kg"
    model += pulp.lpSum(comp_p[m] * x[m] for m in materials) >= 120.0, "Min_Component_P_g"
    model += pulp.lpSum(comp_q[m] * x[m] for m in materials) >= 100.0, "Min_Component_Q_g"

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))

    print("=== 1. Least-Cost Mixture Problem ===")
    print(f"Solver status : {pulp.LpStatus[status]}")
    sol = {m: pulp.value(x[m]) for m in materials}
    for m in materials:
        print(f"  x_{m} = {sol[m]:.4f} kg")
    obj_val = pulp.value(model.objective)
    print(f"Optimal cost  : {obj_val:.2f} CZK")

    # Independent constraint & objective validation
    total_mass = sum(sol.values())
    total_p = sum(comp_p[m] * sol[m] for m in materials)
    total_q = sum(comp_q[m] * sol[m] for m in materials)
    recomputed_cost = sum(cost_czk[m] * sol[m] for m in materials)

    assert abs(total_mass - 1.0) <= tol, "Mass balance violated"
    assert total_p >= 120.0 - tol, "Component P requirement violated"
    assert total_q >= 100.0 - tol, "Component Q requirement violated"
    assert abs(recomputed_cost - obj_val) <= tol, "Objective mismatch"
    print(f"Validation OK : mass={total_mass:.4f} kg, P={total_p:.1f} g, Q={total_q:.1f} g\n")


def solve_free_variable_conversion(tol: float = 1e-7) -> None:
    """
    Demonstrates splitting a free variable x2 = x2_pos - x2_neg (x2_pos, x2_neg >= 0)
    and converting minimization to maximization equality form (Example 1.8).
    """
    # Original form: min 2*x1 - x2 s.t. -x1 + 2*x2 >= 1, x1 + x2 == 3, x1 >= 0, x2 free
    orig = pulp.LpProblem("Original_Free_Var", pulp.LpMinimize)
    x1 = pulp.LpVariable("x1", lowBound=0.0)
    x2 = pulp.LpVariable("x2", lowBound=None)  # free variable
    orig += 2.0 * x1 - x2
    orig += -x1 + 2.0 * x2 >= 1.0
    orig += x1 + x2 == 3.0
    orig.solve(pulp.PULP_CBC_CMD(msg=False))

    # Converted equality maximization form with nonnegative variables + slack s >= 0
    # max (-2*x1 + x2_pos - x2_neg) s.t. x1 - 2*x2_pos + 2*x2_neg + s == -1, x1 + x2_pos - x2_neg == 3
    eq_form = pulp.LpProblem("Equality_Max_Form", pulp.LpMaximize)
    y1 = pulp.LpVariable("x1", lowBound=0.0)
    x2_pos = pulp.LpVariable("x2_pos", lowBound=0.0)
    x2_neg = pulp.LpVariable("x2_neg", lowBound=0.0)
    s = pulp.LpVariable("s", lowBound=0.0)

    eq_form += -2.0 * y1 + x2_pos - x2_neg
    eq_form += y1 - 2.0 * x2_pos + 2.0 * x2_neg + s == -1.0
    eq_form += y1 + x2_pos - x2_neg == 3.0
    eq_form.solve(pulp.PULP_CBC_CMD(msg=False))

    recovered_x2 = pulp.value(x2_pos) - pulp.value(x2_neg)
    recovered_min_obj = -pulp.value(eq_form.objective)

    print("=== 2. Free Variable & Equality Form Conversion ===")
    print(f"Original optimum  : x1 = {pulp.value(x1):.4f}, x2 = {pulp.value(x2):.4f}, min z = {pulp.value(orig.objective):.4f}")
    print(f"Converted optimum : x1 = {pulp.value(y1):.4f}, x2 = {recovered_x2:.4f}, -max(-z) = {recovered_min_obj:.4f}")

    assert abs(pulp.value(x1) - pulp.value(y1)) <= tol
    assert abs(pulp.value(x2) - recovered_x2) <= tol
    assert abs(pulp.value(orig.objective) - recovered_min_obj) <= tol
    print("Equivalence check OK.\n")


if __name__ == "__main__":
    solve_mixture_problem()
    solve_free_variable_conversion()
