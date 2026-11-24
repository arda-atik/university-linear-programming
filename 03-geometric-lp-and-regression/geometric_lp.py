import pulp

# --- 1. Separating two sets of points with a line (Section 1.6.4) ---
# Class P: (2, 0), (0, 2)  |  Class Q: (0, 0), (1, 0)
sep_model = pulp.LpProblem("Line_Separation", pulp.LpMinimize)

w1 = pulp.LpVariable("w1")
w2 = pulp.LpVariable("w2")
b = pulp.LpVariable("b")

sep_model += 0  # Only looking for a feasible line

# Class P points: w1*z1 + w2*z2 + b >= 1
sep_model += 2 * w1 + 0 * w2 + b >= 1
sep_model += 0 * w1 + 2 * w2 + b >= 1

# Class Q points: w1*z1 + w2*z2 + b <= -1
sep_model += 0 * w1 + 0 * w2 + b <= -1
sep_model += 1 * w1 + 0 * w2 + b <= -1

sep_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("--- 1. Linear Separation ---")
print("Status:", pulp.LpStatus[sep_model.status])
print("w1 =", w1.varValue, "w2 =", w2.varValue, "b =", b.varValue)


# --- 2a. Line fitting: minimize sum of absolute errors (Section 1.6.5) ---
# Points: (0, 0), (1, 1), (2, 0) -> line y = a*t + b
l1_model = pulp.LpProblem("L1_Fit", pulp.LpMinimize)

a1 = pulp.LpVariable("a1")
b1 = pulp.LpVariable("b1")
e1 = pulp.LpVariable("e1", lowBound=0)
e2 = pulp.LpVariable("e2", lowBound=0)
e3 = pulp.LpVariable("e3", lowBound=0)

l1_model += e1 + e2 + e3

# Point (0, 0): |a1*0 + b1 - 0| <= e1
l1_model += -e1 <= b1
l1_model += b1 <= e1

# Point (1, 1): |a1*1 + b1 - 1| <= e2
l1_model += -e2 <= a1 + b1 - 1
l1_model += a1 + b1 - 1 <= e2

# Point (2, 0): |a1*2 + b1 - 0| <= e3
l1_model += -e3 <= 2 * a1 + b1
l1_model += 2 * a1 + b1 <= e3

l1_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- 2a. L1 Line Fit ---")
print("a =", a1.varValue, "b =", b1.varValue, "Total error =", pulp.value(l1_model.objective))


# --- 2b. Line fitting: minimize maximum absolute error (Section 1.6.5) ---
linf_model = pulp.LpProblem("Linf_Fit", pulp.LpMinimize)

a2 = pulp.LpVariable("a2")
b2 = pulp.LpVariable("b2")
e_max = pulp.LpVariable("e_max", lowBound=0)

linf_model += e_max

linf_model += -e_max <= b2
linf_model += b2 <= e_max

linf_model += -e_max <= a2 + b2 - 1
linf_model += a2 + b2 - 1 <= e_max

linf_model += -e_max <= 2 * a2 + b2
linf_model += 2 * a2 + b2 <= e_max

linf_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- 2b. L_inf Line Fit ---")
print("a =", a2.varValue, "b =", b2.varValue, "Max error =", pulp.value(linf_model.objective))


# --- 3. Largest disk inside a triangle (Section 1.6.6) ---
# Triangle: z1 >= 0, z2 >= 0, 3*z1 + 4*z2 <= 12
disk_model = pulp.LpProblem("Largest_Disk", pulp.LpMaximize)

s1 = pulp.LpVariable("s1")
s2 = pulp.LpVariable("s2")
r = pulp.LpVariable("r", lowBound=0)

disk_model += r
disk_model += s1 >= r
disk_model += s2 >= r
disk_model += 3 * s1 + 4 * s2 + 5 * r <= 12

disk_model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n--- 3. Largest Disk in Triangle ---")
print("Center = (", s1.varValue, ",", s2.varValue, ") Radius =", r.varValue)
