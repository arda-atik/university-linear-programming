import pulp

# --- 1. Least-cost mixture problem (Section 1.6.1) ---
model = pulp.LpProblem("Mixture_Problem", pulp.LpMinimize)

xA = pulp.LpVariable("xA", lowBound=0)
xB = pulp.LpVariable("xB", lowBound=0)
xC = pulp.LpVariable("xC", lowBound=0)

# Minimize cost (CZK)
model += 8 * xA + 6 * xB + 5 * xC

# Constraints: total 1 kg, at least 120g P and 100g Q
model += xA + xB + xC == 1
model += 200 * xA + 100 * xB >= 120
model += 100 * xB + 300 * xC >= 100

model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- Mixture Problem ---")
print("Status:", pulp.LpStatus[model.status])
print("xA =", xA.varValue, "kg")
print("xB =", xB.varValue, "kg")
print("xC =", xC.varValue, "kg")
print("Min cost =", pulp.value(model.objective), "CZK")


# --- 2. Free variable conversion (Example 1.8) ---
# Original: min 2*x1 - x2, s.t. -x1 + 2*x2 >= 1, x1 + x2 == 3, x1 >= 0, x2 free
# Convert to max equality form: x2 = x2_pos - x2_neg
model_eq = pulp.LpProblem("Equality_Form", pulp.LpMaximize)

x1 = pulp.LpVariable("x1", lowBound=0)
x2_pos = pulp.LpVariable("x2_pos", lowBound=0)
x2_neg = pulp.LpVariable("x2_neg", lowBound=0)
s = pulp.LpVariable("s", lowBound=0)

model_eq += -2 * x1 + x2_pos - x2_neg
model_eq += x1 - 2 * x2_pos + 2 * x2_neg + s == -1
model_eq += x1 + x2_pos - x2_neg == 3

model_eq.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- Free Variable Conversion ---")
print("Status:", pulp.LpStatus[model_eq.status])
print("x1 =", x1.varValue)
print("x2 =", x2_pos.varValue - x2_neg.varValue)
print("Min objective =", -pulp.value(model_eq.objective))
