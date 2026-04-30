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


def annualized_cost_no_fom(costs, tech, discount_rate=DISCOUNT_RATE):
    investment = get_cost(costs, tech, "investment") * 1000
    lifetime = get_cost(costs, tech, "lifetime")
    return investment * annuity(discount_rate, lifetime)


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


def plot_dispatch_with_battery(network, period, title, savepath):
    dispatch = network.generators_t.p[
        ["solar", "onshore_wind", "offshore_wind", "ocgt"]
    ].copy()

    load_ts = network.loads_t.p[f"{ZONE}_load"].copy()
    battery_p = network.storage_units_t.p["battery"].copy()

    battery_discharge = battery_p.clip(lower=0)
    battery_charge = -battery_p.clip(upper=0)

    dispatch["battery_discharge"] = battery_discharge
    effective_load = load_ts + battery_charge

    ax = dispatch.loc[period].plot.area(figsize=(12, 5))

    effective_load.loc[period].plot(
        ax=ax,
        color="black",
        linewidth=2,
        label="load + battery charging",
    )

    load_ts.loc[period].plot(
        ax=ax,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label="load",
    )

    plt.title(title)
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_battery_behavior(network, period, title, savepath):
    battery_p = network.storage_units_t.p["battery"].copy()
    battery_soc = network.storage_units_t.state_of_charge["battery"].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))

    battery_p.loc[period].plot(ax=ax1, label="battery dispatch [MW]")
    ax1.axhline(0, linewidth=1)
    ax1.set_ylabel("MW")

    ax2 = ax1.twinx()
    battery_soc.loc[period].plot(
        ax=ax2,
        linestyle="--",
        label="state of charge [MWh]",
    )
    ax2.set_ylabel("MWh")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


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
    .astype("float64")
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

BATTERY_DURATION = 4
BATTERY_EFF_STORE = 0.95
BATTERY_EFF_DISPATCH = 0.95
BATTERY_MARGINAL_COST = 0

battery_inverter_investment = get_cost(costs, "battery inverter", "investment") * 1000  # €/MW
battery_inverter_lifetime = get_cost(costs, "battery inverter", "lifetime")
battery_inverter_fom = get_cost(costs, "battery inverter", "FOM") / 100 * battery_inverter_investment

battery_storage_investment = get_cost(costs, "battery storage", "investment") * 1000  # €/MWh
battery_storage_lifetime = get_cost(costs, "battery storage", "lifetime")

battery_inverter_cost = (
    battery_inverter_investment * annuity(DISCOUNT_RATE, battery_inverter_lifetime)
    + battery_inverter_fom
)

battery_storage_cost = (
    battery_storage_investment * annuity(DISCOUNT_RATE, battery_storage_lifetime)
)

battery_capital_cost = (
    battery_inverter_cost
    + battery_storage_cost
)

print("\nBattery cost components:")
print("Battery inverter investment [€/MW]:", round(battery_inverter_investment, 2))
print("Battery storage investment [€/MWh]:", round(battery_storage_investment, 2))
print("Battery inverter annualized [€/MW/year]:", round(battery_inverter_cost, 2))
print("Battery storage annualized [€/MWh/year]:", round(battery_storage_cost, 2))
print("Battery total capital cost [€/MW/year]:", round(battery_capital_cost, 2))

print("\nCost check:")
print("Solar capital cost [€/MW/year]:", round(solar_capital_cost, 2))
print("Onshore capital cost [€/MW/year]:", round(onshore_capital_cost, 2))
print("Offshore capital cost [€/MW/year]:", round(offshore_capital_cost, 2))
print("OCGT capital cost [€/MW/year]:", round(ocgt_capital_cost, 2))
print("OCGT marginal cost [€/MWh]:", round(ocgt_mc, 2))
print("Battery capital cost [€/MW/year]:", round(battery_capital_cost, 2))


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

network.add(
    "StorageUnit",
    "battery",
    bus=ZONE,
    p_nom_extendable=True,
    max_hours=BATTERY_DURATION,
    efficiency_store=BATTERY_EFF_STORE,
    efficiency_dispatch=BATTERY_EFF_DISPATCH,
    capital_cost=battery_capital_cost,
    marginal_cost=BATTERY_MARGINAL_COST,
    cyclic_state_of_charge=True,
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

battery_power = network.storage_units.at["battery", "p_nom_opt"]
battery_energy = battery_power * network.storage_units.at["battery", "max_hours"]

battery_dispatch = network.storage_units_t.p["battery"]
battery_soc = network.storage_units_t.state_of_charge["battery"]

annual_charge = -battery_dispatch.clip(upper=0).sum()
annual_discharge = battery_dispatch.clip(lower=0).sum()
equivalent_cycles = annual_discharge / battery_energy if battery_energy > 0 else 0

print("\nOptimal generator capacities:")
print(capacities.round(2))

print("\nBattery:")
print(f"Power capacity [MW]: {battery_power:.2f}")
print(f"Energy capacity [MWh]: {battery_energy:.2f}")
print(f"Annual charge [MWh]: {annual_charge:.2f}")
print(f"Annual discharge [MWh]: {annual_discharge:.2f}")
print(f"Equivalent full cycles: {equivalent_cycles:.2f}")

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
# Curtailment diagnostic
# ============================================================

renewable_available = (
    network.generators_t.p_max_pu["solar"] * network.generators.at["solar", "p_nom_opt"]
    + network.generators_t.p_max_pu["onshore_wind"] * network.generators.at["onshore_wind", "p_nom_opt"]
    + network.generators_t.p_max_pu["offshore_wind"] * network.generators.at["offshore_wind", "p_nom_opt"]
)

renewable_dispatch = (
    network.generators_t.p["solar"]
    + network.generators_t.p["onshore_wind"]
    + network.generators_t.p["offshore_wind"]
)

curtailment = renewable_available - renewable_dispatch

print("\nCurtailment:")
print("Annual curtailment [MWh]:", round(curtailment.sum(), 2))
print("Max curtailment [MW]:", round(curtailment.max(), 2))


# ============================================================
# Save results
# ============================================================

taskC_results = pd.concat([
    network.generators["p_nom_opt"],
    pd.Series({
        "battery_power": battery_power,
        "battery_energy": battery_energy,
        "annual_battery_charge": annual_charge,
        "annual_battery_discharge": annual_discharge,
        "equivalent_full_cycles": equivalent_cycles,
        "annual_curtailment": curtailment.sum(),
        "max_curtailment": curtailment.max(),
    })
])

taskC_results.to_csv(os.path.join(DATA_DIR, "taskC_results.csv"))


# ============================================================
# Plots
# ============================================================

summer_week = slice(f"{YEAR}-07-07", f"{YEAR}-07-13 23:00")
winter_week = slice(f"{YEAR}-01-13", f"{YEAR}-01-19 23:00")
winter = slice(f"{YEAR}-01-01", f"{YEAR}-03-31 23:00")
spring = slice(f"{YEAR}-04-01", f"{YEAR}-06-30 23:00")
summer = slice(f"{YEAR}-07-01", f"{YEAR}-09-30 23:00")
fall = slice(f"{YEAR}-10-01", f"{YEAR}-12-31 23:00")
full_year = slice(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00")

plot_dispatch_with_battery(
    network,
    summer_week,
    f"Dispatch in {ZONE} with battery - Summer week",
    os.path.join(FIG_DIR, "taskC_dispatch_summer_battery.png"),
)

plot_dispatch_with_battery(
    network,
    winter_week,
    f"Dispatch in {ZONE} with battery - Winter week",
    os.path.join(FIG_DIR, "taskC_dispatch_winter_battery.png"),
)

plot_battery_behavior(
    network,
    summer_week,
    "Battery behavior - Summer week",
    os.path.join(FIG_DIR, "taskC_battery_behavior_summer_week.png"),
)

plot_battery_behavior(
    network,
    winter_week,
    "Battery behavior - Winter week",
    os.path.join(FIG_DIR, "taskC_battery_behavior_winter_week.png"),
)

plot_battery_behavior(
    network,
    winter,
    "Battery behavior - Winter",
    os.path.join(FIG_DIR, "taskC_battery_behavior_winter.png"),
)

plot_battery_behavior(
    network,
    spring,
    "Battery behavior - Spring",
    os.path.join(FIG_DIR, "taskC_battery_behavior_spring.png"),
)

plot_battery_behavior(
    network,
    summer,
    "Battery behavior - Summer",
    os.path.join(FIG_DIR, "taskC_battery_behavior_summer.png"),
)

plot_battery_behavior(
    network,
    fall,
    "Battery behavior - Fall",
    os.path.join(FIG_DIR, "taskC_battery_behavior_fall.png"),
)

plt.figure(figsize=(12, 4))
battery_soc.plot()
plt.title("Battery state of charge over full year")
plt.ylabel("MWh")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskC_battery_soc_full_year.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

plt.figure(figsize=(12, 4))
battery_soc.rolling(24 * 7).mean().plot()
plt.title("Smoothed battery state of charge - weekly average")
plt.ylabel("MWh")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskC_battery_soc_weekly_average.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()