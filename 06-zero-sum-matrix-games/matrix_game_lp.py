import pulp

# --- Zero-Sum Matrix Game: Smuggler vs. Customs Patrol (Chapter 10, Example 10.1) ---
# Payoff matrix A (column player = smuggler's gain):
#                 C1 (Secondary)   C2 (Highway)
# R1 (Secondary)        0               6
# R2 (Highway)          2               0

# --- 1. Column Player's LP (Maximin strategy x) ---
col_model = pulp.LpProblem("Column_Player_Maximin", pulp.LpMaximize)

x1 = pulp.LpVariable("x1", lowBound=0)  # Prob of C1
x2 = pulp.LpVariable("x2", lowBound=0)  # Prob of C2
v = pulp.LpVariable("v")                # Guaranteed min payoff (free)

col_model += v

# Expected payoff against each row must be at least v
col_model += 0 * x1 + 6 * x2 >= v
col_model += 2 * x1 + 0 * x2 >= v
col_model += x1 + x2 == 1

col_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- Column Player (Smuggler - Maximin) ---")
print("Status:", pulp.LpStatus[col_model.status])
print(f"Mixed strategy: x1 = {x1.varValue:.4f}, x2 = {x2.varValue:.4f}")
print(f"Game value (v*) = {v.varValue:.4f}")


# --- 2. Row Player's LP (Minimax strategy y - Dual LP) ---
row_model = pulp.LpProblem("Row_Player_Minimax", pulp.LpMinimize)

y1 = pulp.LpVariable("y1", lowBound=0)  # Prob of R1
y2 = pulp.LpVariable("y2", lowBound=0)  # Prob of R2
u = pulp.LpVariable("u")                # Guaranteed max loss (free)

row_model += u

# Expected loss against each column must be at most u
row_model += 0 * y1 + 2 * y2 <= u
row_model += 6 * y1 + 0 * y2 <= u
row_model += y1 + y2 == 1

row_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- Row Player (Customs - Minimax) ---")
print("Status:", pulp.LpStatus[row_model.status])
print(f"Mixed strategy: y1 = {y1.varValue:.4f}, y2 = {y2.varValue:.4f}")
print(f"Game value (u*) = {u.varValue:.4f}")
