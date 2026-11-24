import pulp

# --- 1. Maximum network flow (Section 1.6.2) ---
flow_model = pulp.LpProblem("Max_Flow", pulp.LpMaximize)

x_sa = pulp.LpVariable("x_sa", lowBound=0, upBound=4)
x_sb = pulp.LpVariable("x_sb", lowBound=0, upBound=2)
x_ab = pulp.LpVariable("x_ab", lowBound=0, upBound=1)
x_at = pulp.LpVariable("x_at", lowBound=0, upBound=2)
x_bt = pulp.LpVariable("x_bt", lowBound=0, upBound=4)

# Maximize total flow from source s
flow_model += x_sa + x_sb

# Flow conservation at intermediate nodes a and b
flow_model += x_sa == x_ab + x_at
flow_model += x_sb + x_ab == x_bt

flow_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- Maximum Network Flow ---")
print("Status:", pulp.LpStatus[flow_model.status])
print("Flows:", x_sa.varValue, x_sb.varValue, x_ab.varValue, x_at.varValue, x_bt.varValue)
print("Max rate =", pulp.value(flow_model.objective), "MB/s")


# --- 2. Two-month production and inventory plan (Section 1.6.3) ---
prod_model = pulp.LpProblem("Production_Plan", pulp.LpMinimize)

# Production x1, x2; end-of-month inventory s1; production change u1, u2
x1 = pulp.LpVariable("x1", lowBound=0)
x2 = pulp.LpVariable("x2", lowBound=0)
s1 = pulp.LpVariable("s1", lowBound=0)
u1 = pulp.LpVariable("u1", lowBound=0)
u2 = pulp.LpVariable("u2", lowBound=0)

# Demand d1=2, d2=6, initial production x0=2, holding cost h=1, change cost k=2
prod_model += 1 * s1 + 2 * (u1 + u2)

# Inventory balance (s0 = 0, s2 = 0)
prod_model += x1 - s1 == 2
prod_model += s1 + x2 == 6

# Linearized |x_i - x_{i-1}| <= u_i
prod_model += -u1 <= x1 - 2
prod_model += x1 - 2 <= u1
prod_model += -u2 <= x2 - x1
prod_model += x2 - x1 <= u2

prod_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- Production & Inventory Plan ---")
print("Status:", pulp.LpStatus[prod_model.status])
print("Month 1 prod =", x1.varValue, "Month 2 prod =", x2.varValue)
print("End of month 1 inventory =", s1.varValue)
print("Min cost =", pulp.value(prod_model.objective), "thousand CZK")
