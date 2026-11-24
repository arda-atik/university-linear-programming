import pulp

# --- Balanced Transportation Problem (Chapter 9, Example 9.9) ---
# Suppliers S1, S2, S3 -> Supplies: 15, 17, 18 (Total = 50)
# Customers D1, D2, D3 -> Demands : 20, 10, 20 (Total = 50)

model = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)

# Shipment variables x_ij >= 0 (from supplier i to customer j)
x11 = pulp.LpVariable("x11", lowBound=0)
x12 = pulp.LpVariable("x12", lowBound=0)
x13 = pulp.LpVariable("x13", lowBound=0)

x21 = pulp.LpVariable("x21", lowBound=0)
x22 = pulp.LpVariable("x22", lowBound=0)
x23 = pulp.LpVariable("x23", lowBound=0)

x31 = pulp.LpVariable("x31", lowBound=0)
x32 = pulp.LpVariable("x32", lowBound=0)
x33 = pulp.LpVariable("x33", lowBound=0)

# Objective: minimize total shipping cost
model += (
    6 * x11 + 8 * x12 + 10 * x13
    + 7 * x21 + 11 * x22 + 11 * x23
    + 4 * x31 + 5 * x32 + 12 * x33
)

# Supply constraints
model += x11 + x12 + x13 == 15, "Supply_S1"
model += x21 + x22 + x23 == 17, "Supply_S2"
model += x31 + x32 + x33 == 18, "Supply_S3"

# Demand constraints
model += x11 + x21 + x31 == 20, "Demand_D1"
model += x12 + x22 + x32 == 10, "Demand_D2"
model += x13 + x23 + x33 == 20, "Demand_D3"

model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- Transportation Problem Solution ---")
print("Status:", pulp.LpStatus[model.status])
print(f"Row 1 (S1): x11={x11.varValue}, x12={x12.varValue}, x13={x13.varValue}")
print(f"Row 2 (S2): x21={x21.varValue}, x22={x22.varValue}, x23={x23.varValue}")
print(f"Row 3 (S3): x31={x31.varValue}, x32={x32.varValue}, x33={x33.varValue}")
print("Minimum transportation cost =", pulp.value(model.objective))

# MODI potentials (u_i and v_j) correspond to dual variables of supply & demand equations
print("\n--- Dual Potentials (MODI u_i and v_j) ---")
for name, c in model.constraints.items():
    print(f"{name}: potential = {c.pi:.2f}")
