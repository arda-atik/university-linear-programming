# University Linear Programming Models (01LIP)

Python implementations of linear programming (LP) and integer linear programming (ILP) models from the **01LIP** (*Linear Programming*) coursework.

## Repository Structure

| Module | Directory | Topics Covered |
| :--- | :--- | :--- |
| **01** | `01-blending-and-diet-lp/` | Least-cost mixture formulation, free variable splitting ($x = x^+ - x^-$), and equality-form conversion |
| **02** | `02-network-flow-and-inventory/` | Maximum steady-state network flow and multi-period production-inventory planning with linearized absolute changes |
| **03** | `03-geometric-lp-and-regression/` | Linear separation of two point sets, $L_1$ and $L_\infty$ line fitting, and largest inscribed disk in a polygon |
| **04** | `04-simplex-and-duality/` | Primal-dual formulation, strong duality verification, shadow prices (`.pi`), and complementary slackness (`.slack`) |
| **05** | `05-transportation-problem/` | Balanced transportation problem and dual potentials corresponding to the MODI method |
| **06** | `06-zero-sum-matrix-games/` | Two-player zero-sum matrix games, mixed strategies, and the minimax theorem via dual LPs |
| **07** | `07-integer-programming-milp/` | LP relaxation bounds, integrality gap, branch-and-bound integer solutions, and the binary knapsack problem |

## Setup & Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
