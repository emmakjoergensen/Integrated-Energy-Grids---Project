import pandas as pd

def load_costs(path, year=2030):
    costs = pd.read_csv(path)
    costs = costs[costs["year"] == year].copy()
    costs["value"] = pd.to_numeric(costs["value"], errors="coerce")
    return costs.set_index(["technology", "parameter"])

def get_cost(costs, tech, param):
    return float(costs.loc[(tech, param), "value"])