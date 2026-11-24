import pulp

# --- 1. Linear separation of two point sets (Section 1.6.4) ---
# Class P: (2, 0), (0, 2) | Class Q: (0, 0), (1, 0)
sep_model = pulp.LpProblem("Linear_Separation", pulp.LpMinimize)

w1 = pulp.LpVariable("w1")  # free variable
w2 = pulp.LpVariable("w2")  # free variable
b = pulp.LpVariable("b")    # free variable

sep_model += 0  # feasibility problem

# w^T * p + b >= 1 for class P
sep_model += 2 * w1 + 0 * w2 + b >= 1
sep_model += 0 * w1 + 2 * w2 + b >= 1

# w^T * q + b <= -1 for class Q
sep_model += 0 * w1 + 0 * w2 + b <= -1
sep_model += 1 * w1 + 0 * w2 + b <= -1

sep_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- 1. Linear Separation ---")
print("Status:", pulp.LpStatus[sep_model.status])
print(f"Line: {w1.varValue}*z1 + {w2.varValue}*z2 + ({b.varValue}) = 0")


# --- 2. Line fitting with L1 and L_inf norms (Section 1.6.5) ---
# Points: (0, 0), (1, 1), (2, 0) -> fit y = a*t + b
points = [(0, 0), (1, 1), (2, 0)]

# 2a. Minimize sum of absolute deviations (L1)
l1_model = pulp.LpProblem("L1_Line_Fit", pulp.LpMinimize)
a1 = pulp.LpVariable("a")
b1 = pulp.LpVariable("b")
e = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(len(points))]

l1_model += pulp.lpSum(e)
for i, (t, y) in enumerate(points):
    l1_model += -e[i] <= a1 * t + b1 - y
    l1_model += a1 * t + b1 - y <= e[i]

l1_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- 2a. L1 Line Fit ---")
print(f"Best line: y = {a1.varValue}*t + {b1.varValue}, sum of errors = {pulp.value(l1_model.objective)}")

# 2b. Minimize maximum absolute deviation (L_inf)
linf_model = pulp.LpProblem("Linf_Line_Fit", pulp.LpMinimize)
a2 = pulp.LpVariable("a")
b2 = pulp.LpVariable("b")
e_max = pulp.LpVariable("e_max", lowBound=0)

linf_model += e_max
for t, y in points:
    linf_model += -e_max <= a2 * t + b2 - y
    linf_model += a2 * t + b2 - y <= e_max

linf_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- 2b. L_inf Line Fit ---")
print(f"Best line: y = {a2.varValue}*t + {b2.varValue}, max error = {pulp.value(linf_model.objective)}")


# --- 3. Largest disk inside a triangle (Section 1.6.6) ---
# Triangle: z1 >= 0, z2 >= 0, 3*z1 + 4*z2 <= 12
disk_model = pulp.LpProblem("Largest_Disk", pulp.LpMaximize)

s1 = pulp.LpVariable("s1")             # center x
s2 = pulp.LpVariable("s2")             # center y
r = pulp.LpVariable("r", lowBound=0)   # radius

disk_model += r
disk_model += s1 >= r
disk_model += s2 >= r
disk_model += 3 * s1 + 4 * s2 + 5 * r <= 12  # ||(3, 4)||_2 = 5

disk_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- 3. Largest Disk in Triangle ---")
print(f"Center = ({s1.varValue}, {s2.varValue}), Max radius r = {r.varValue}")
