import pulp

# --- 1. LP Relaxation vs. Pure Integer Solution (Chapter 11, Example 11.5) ---
# Maximize 3*x1 + 2*x2
# s.t.     4*x1 + 2*x2 <= 9
#          2*x1 + 4*x2 <= 9
#          x1, x2 >= 0, integer

# 1a. LP Relaxation (continuous variables)
lp_relax = pulp.LpProblem("LP_Relaxation", pulp.LpMaximize)
x1_rel = pulp.LpVariable("x1", lowBound=0, cat="Continuous")
x2_rel = pulp.LpVariable("x2", lowBound=0, cat="Continuous")

lp_relax += 3 * x1_rel + 2 * x2_rel
lp_relax += 4 * x1_rel + 2 * x2_rel <= 9
lp_relax += 2 * x1_rel + 4 * x2_rel <= 9
lp_relax.solve(pulp.PULP_CBC_CMD(msg=False))

z_lp = pulp.value(lp_relax.objective)
print("--- 1a. LP Relaxation ---")
print("x1 =", x1_rel.varValue, "x2 =", x2_rel.varValue, "Upper bound z_LP =", z_lp)

# 1b. Pure Integer Program (solved via Branch & Bound in CBC)
ilp_model = pulp.LpProblem("Pure_ILP", pulp.LpMaximize)
x1_int = pulp.LpVariable("x1", lowBound=0, cat="Integer")
x2_int = pulp.LpVariable("x2", lowBound=0, cat="Integer")

ilp_model += 3 * x1_int + 2 * x2_int
ilp_model += 4 * x1_int + 2 * x2_int <= 9
ilp_model += 2 * x1_int + 4 * x2_int <= 9
ilp_model.solve(pulp.PULP_CBC_CMD(msg=False))

z_ilp = pulp.value(ilp_model.objective)
print("\n--- 1b. Integer Optimum (Branch & Bound) ---")
print("x1 =", x1_int.varValue, "x2 =", x2_int.varValue, "Optimal z_ILP =", z_ilp)
print("Integrality gap (z_LP - z_ILP) =", z_lp - z_ilp)


# --- 2. Binary Knapsack Problem (Chapter 11, Example 11.1) ---
# 4 items with profits and weights, knapsack capacity W = 7
knapsack = pulp.LpProblem("Binary_Knapsack", pulp.LpMaximize)

y1 = pulp.LpVariable("item_1", cat="Binary")
y2 = pulp.LpVariable("item_2", cat="Binary")
y3 = pulp.LpVariable("item_3", cat="Binary")
y4 = pulp.LpVariable("item_4", cat="Binary")

# Maximize total profit
knapsack += 10 * y1 + 7 * y2 + 12 * y3 + 8 * y4

# Weight capacity constraint
knapsack += 4 * y1 + 3 * y2 + 5 * y3 + 3 * y4 <= 7

knapsack.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- 2. Binary Knapsack Problem ---")
print("Status:", pulp.LpStatus[knapsack.status])
print("Selected items:", y1.varValue, y2.varValue, y3.varValue, y4.varValue)
print("Total profit =", pulp.value(knapsack.objective))
