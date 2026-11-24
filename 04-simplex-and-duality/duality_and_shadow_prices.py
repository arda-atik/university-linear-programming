import pulp

# --- 1. Primal Problem (Example 7.1) ---
primal = pulp.LpProblem("Primal_LP", pulp.LpMaximize)

x1 = pulp.LpVariable("x1", lowBound=0)
x2 = pulp.LpVariable("x2", lowBound=0)

primal += 2 * x1 + 3 * x2

c1 = 4 * x1 + 8 * x2 <= 12
c2 = 2 * x1 + 1 * x2 <= 3
c3 = 3 * x1 + 2 * x2 <= 4

primal += c1, "Resource_1"
primal += c2, "Resource_2"
primal += c3, "Resource_3"

primal.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- Primal Solution ---")
print("Status:", pulp.LpStatus[primal.status])
print("x1 =", x1.varValue, "x2 =", x2.varValue)
print("Primal optimal value =", pulp.value(primal.objective))

# Shadow prices (dual variables) and slack values from solver
print("\n--- Shadow Prices & Slacks (Complementary Slackness) ---")
for name, constraint in primal.constraints.items():
    print(f"{name}: shadow price (y*) = {constraint.pi:.4f}, slack = {constraint.slack:.4f}")


# --- 2. Explicit Dual Problem (Example 7.1) ---
dual = pulp.LpProblem("Dual_LP", pulp.LpMinimize)

y1 = pulp.LpVariable("y1", lowBound=0)
y2 = pulp.LpVariable("y2", lowBound=0)
y3 = pulp.LpVariable("y3", lowBound=0)

dual += 12 * y1 + 3 * y2 + 4 * y3

dual += 4 * y1 + 2 * y2 + 3 * y3 >= 2
dual += 8 * y1 + 1 * y2 + 2 * y3 >= 3

dual.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- Dual Solution ---")
print("Status:", pulp.LpStatus[dual.status])
print("y1 =", y1.varValue, "y2 =", y2.varValue, "y3 =", y3.varValue)
print("Dual optimal value =", pulp.value(dual.objective))
