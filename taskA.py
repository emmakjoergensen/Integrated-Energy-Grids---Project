import os
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

from src.config import YEAR, ZONE, DISCOUNT_RATE, PSR_MAP
from src.data_loader import load_generation, load_load
from src.cost_loader import load_costs, get_cost
from src.profiles import make_profile
from src.network_builder import create_network
from src.plotting import plot_dispatch

pd.options.future.infer_string = False


# ============================================================
# Paths
# ============================================================

DATA_DIR = "Data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
FIG_DIR = "Figures"

os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def annuity(rate, lifetime):
    return rate / (1 - (1 + rate) ** (-lifetime))


def cost_breakdown(costs, tech, discount_rate=DISCOUNT_RATE):
    investment = get_cost(costs, tech, "investment") * 1000
    lifetime = get_cost(costs, tech, "lifetime")
    fom_percent = get_cost(costs, tech, "FOM")

    annualized_capex = investment * annuity(discount_rate, lifetime)
    fixed_opex = fom_percent / 100 * investment
    total_fixed_cost = annualized_capex + fixed_opex

    return {
        "investment": investment,
        "annualized_capex": annualized_capex,
        "fixed_opex": fixed_opex,
        "total_fixed_cost": total_fixed_cost,
    }


def annualized_cost(costs, tech, discount_rate=DISCOUNT_RATE):
    return cost_breakdown(costs, tech, discount_rate)["total_fixed_cost"]


def ocgt_marginal_cost(costs):
    efficiency = get_cost(costs, "OCGT", "efficiency")
    vom = get_cost(costs, "OCGT", "VOM")

    gas_price = 40.0
    co2_price = 80.0
    co2_intensity = 0.202

    return gas_price / efficiency + co2_price * co2_intensity / efficiency + vom


def filter_year(df, time_col, year):
    return df[df[time_col].dt.year == year].copy()


def build_renewable_profile(generation, psr_codes, snapshots, target_cf):
    series = (
        generation[generation["psr_type"].isin(psr_codes)]
        .groupby("time_utc")["generation_MW"]
        .sum()
        .reindex(snapshots)
        .fillna(0)
    )

    return make_profile(series, target_cf)


# ============================================================
# Load data
# ============================================================

snapshots = pd.date_range(
    f"{YEAR}-01-01 00:00",
    f"{YEAR}-12-31 23:00",
    freq="h"
)

generation_path = os.path.join(
    RAW_DIR,
    "generation",
    f"gen_{ZONE}_2020_2025.csv"
)

solar_generation_path = os.path.join(
    RAW_DIR,
    "generation",
    f"solar_{ZONE}_2020_2025.csv"
)

load_path = os.path.join(
    RAW_DIR,
    "load",
    f"load_{ZONE}_2020_2025.csv"
)

cost_path = os.path.join(DATA_DIR, "costs_PyPSA.csv")

generation_all = load_generation(generation_path)
solar_generation_all = load_generation(solar_generation_path)
load_all = load_load(load_path)
costs = load_costs(cost_path, year=2030)

generation = filter_year(generation_all, "time_utc", YEAR)
solar_generation = filter_year(solar_generation_all, "time_utc", YEAR)
load_data = filter_year(load_all, "time_utc", YEAR)

load = (
    load_data
    .set_index("time_utc")["load_MW"]
    .reindex(snapshots)
    .ffill()
    .fillna(0)
)


# ============================================================
# Renewable availability profiles
# ============================================================

solar_cf = build_renewable_profile(
    solar_generation,
    PSR_MAP["solar"],
    snapshots,
    target_cf=0.11
)

onshore_cf = build_renewable_profile(
    generation,
    PSR_MAP["onshore_wind"],
    snapshots,
    target_cf=0.25
)

offshore_cf = build_renewable_profile(
    generation,
    PSR_MAP["offshore_wind"],
    snapshots,
    target_cf=0.52
)

print("\nRenewable profile means:")
print("Solar CF mean:", solar_cf.mean())
print("Onshore CF mean:", onshore_cf.mean())
print("Offshore CF mean:", offshore_cf.mean())


# ============================================================
# Technology costs
# ============================================================

solar_capital_cost = annualized_cost(costs, "solar")
onshore_capital_cost = annualized_cost(costs, "onwind")
offshore_capital_cost = annualized_cost(costs, "offwind")
ocgt_capital_cost = annualized_cost(costs, "OCGT")

ocgt_efficiency = get_cost(costs, "OCGT", "efficiency")
ocgt_mc = ocgt_marginal_cost(costs)

solar_costs = cost_breakdown(costs, "solar")
onshore_costs = cost_breakdown(costs, "onwind")
offshore_costs = cost_breakdown(costs, "offwind")
ocgt_costs = cost_breakdown(costs, "OCGT")

cost_table = pd.DataFrame({
    "Technology": ["Solar PV", "Onshore wind", "Offshore wind", "OCGT"],
    "Investment CAPEX [€/MW]": [
        solar_costs["investment"],
        onshore_costs["investment"],
        offshore_costs["investment"],
        ocgt_costs["investment"],
    ],
    "Annualized CAPEX [€/MW/year]": [
        solar_costs["annualized_capex"],
        onshore_costs["annualized_capex"],
        offshore_costs["annualized_capex"],
        ocgt_costs["annualized_capex"],
    ],
    "Fixed OPEX [€/MW/year]": [
        solar_costs["fixed_opex"],
        onshore_costs["fixed_opex"],
        offshore_costs["fixed_opex"],
        ocgt_costs["fixed_opex"],
    ],
    "Total fixed cost [€/MW/year]": [
        solar_costs["total_fixed_cost"],
        onshore_costs["total_fixed_cost"],
        offshore_costs["total_fixed_cost"],
        ocgt_costs["total_fixed_cost"],
    ],
    "Marginal cost [€/MWh]": [0, 0, 0, ocgt_mc],
}).round(2)

print("\nCost assumptions:")
print(cost_table.to_string(index=False))


# ============================================================
# Build PyPSA network
# ============================================================

network = create_network(snapshots)

network.add("Carrier", "AC")
network.add("Bus", ZONE, carrier="AC")

network.add(
    "Load",
    f"{ZONE}_load",
    bus=ZONE,
    p_set=load,
)

network.add(
    "Generator",
    "solar",
    bus=ZONE,
    p_nom_extendable=True,
    p_max_pu=solar_cf,
    capital_cost=solar_capital_cost,
    marginal_cost=0,
)

network.add(
    "Generator",
    "onshore_wind",
    bus=ZONE,
    p_nom_extendable=True,
    p_max_pu=onshore_cf,
    capital_cost=onshore_capital_cost,
    marginal_cost=0,
)

network.add(
    "Generator",
    "offshore_wind",
    bus=ZONE,
    p_nom_extendable=True,
    p_max_pu=offshore_cf,
    capital_cost=offshore_capital_cost,
    marginal_cost=0,
)

network.add(
    "Generator",
    "ocgt",
    bus=ZONE,
    p_nom_extendable=True,
    capital_cost=ocgt_capital_cost,
    marginal_cost=ocgt_mc,
    efficiency=ocgt_efficiency,
)


# ============================================================
# Optimise
# ============================================================

network.optimize()


# ============================================================
# Results
# ============================================================

capacities = network.generators[["p_nom_opt"]].rename(
    columns={"p_nom_opt": "Optimal capacity [MW]"}
)

print("\nOptimal capacities:")
print(capacities.round(2))

dispatch = network.generators_t.p[
    ["solar", "onshore_wind", "offshore_wind", "ocgt"]
]

annual_generation = dispatch.sum().rename("Annual generation [MWh]")

print("\nAnnual generation:")
print(annual_generation.round(2))

capacity_factors = (
    dispatch.sum()
    / (network.generators["p_nom_opt"] * len(network.snapshots))
)

print("\nCapacity factors:")
print(capacity_factors.round(3))


# ============================================================
# Plots
# ============================================================

summer_week = slice(f"{YEAR}-07-07", f"{YEAR}-07-13 23:00")
winter_week = slice(f"{YEAR}-01-13", f"{YEAR}-01-19 23:00")

plot_dispatch(
    dispatch,
    load,
    summer_week,
    f"Dispatch in {ZONE} - Summer week",
    os.path.join(FIG_DIR, "taskA_dispatch_summer.png"),
)

plot_dispatch(
    dispatch,
    load,
    winter_week,
    f"Dispatch in {ZONE} - Winter week",
    os.path.join(FIG_DIR, "taskA_dispatch_winter.png"),
)

annual_generation.plot(kind="bar", figsize=(8, 4))
plt.title(f"Annual electricity mix in {ZONE}")
plt.ylabel("MWh")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "taskA_annual_mix.png"), dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
for tech in dispatch.columns:
    dispatch[tech].sort_values(ascending=False).reset_index(drop=True).plot(label=tech)

plt.title(f"Generation duration curves - {ZONE}")
plt.xlabel("Hour rank")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "taskA_duration_curves.png"), dpi=300)
plt.show()