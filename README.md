# University Linear Programming Models (01LIP)

Python implementations of linear programming (LP) and integer linear programming (ILP) models from the **01LIP** (*Linear Programming*) course at **CTU in Prague (FNSPE / FJFI)**[cite: 4, 6].

## Repository Structure

| Module | Directory | Topics Covered |
| :--- | :--- | :--- |
| **01** | `01-blending-and-diet-lp/` | Least-cost mixture formulation, free variable splitting ($x = x^+ - x^-$), and equality-form conversion[cite: 15, 18, 24] |
| **02** | `02-network-flow-and-inventory/` | Maximum steady-state network flow and multi-period production-inventory planning with linearized absolute changes[cite: 25, 26] |
| **03** | `03-geometric-lp-and-regression/` | Linear separation of two point sets, $L_1$ and $L_\infty$ line fitting, and largest inscribed disk in a polygon[cite: 27, 28, 29, 30] |
| **04** | `04-simplex-and-duality/` | Primal-dual formulation, strong duality verification, shadow prices (`.pi`), and complementary slackness (`.slack`)[cite: 78, 79, 82, 91] |
| **05** | `05-transportation-problem/` | Balanced transportation problem and dual potentials corresponding to the MODI method[cite: 96, 102, 104] |
| **06** | `06-zero-sum-matrix-games/` | Two-player zero-sum matrix games, mixed strategies, and the minimax theorem via dual LPs[cite: 112, 113, 118, 120] |
| **07** | `07-integer-programming-milp/` | LP relaxation bounds, integrality gap, branch-and-bound integer solutions, and the binary knapsack problem[cite: 19, 127, 130, 131] |

## Setup & Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
