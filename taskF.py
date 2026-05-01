import os
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

from src.config import YEAR, ZONE, DISCOUNT_RATE, PSR_MAP
from src.data_loader import load_generation, load_load
from src.cost_loader import load_costs, get_cost
from src.profiles import make_profile
from src.network_builder import create_network


# ============================================================
# Plot style (global)
# ============================================================

PLOT_FONT_SIZE = 24  # <-- change ONLY this number

plt.rcParams.update({
    "font.size": PLOT_FONT_SIZE,           # base font size
    "axes.titlesize": PLOT_FONT_SIZE + 2,  # title
    "axes.labelsize": PLOT_FONT_SIZE,      # x/y labels
    "xtick.labelsize": PLOT_FONT_SIZE - 1,
    "ytick.labelsize": PLOT_FONT_SIZE - 1,
    "legend.fontsize": PLOT_FONT_SIZE - 1,
    "figure.titlesize": PLOT_FONT_SIZE + 2,
})

pd.options.future.infer_string = False

# ============================================================
# Paths
# ============================================================

DATA_DIR = "Data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
FIG_DIR = "Figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Constants
# ============================================================

GAS_PRICE = 40.0
CO2_INTENSITY = 0.202  # tCO2 / MWh_th

BATTERY_DURATION = 4
BATTERY_EFF_STORE = 0.95
BATTERY_EFF_DISPATCH = 0.95
BATTERY_MARGINAL_COST = 0.0

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


def ocgt_marginal_cost_without_co2(costs):
    efficiency = get_cost(costs, "OCGT", "efficiency")
    vom = get_cost(costs, "OCGT", "VOM")
    return GAS_PRICE / efficiency + vom


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
ocgt_mc_taskf = ocgt_marginal_cost_without_co2(costs)

battery_inverter_investment = get_cost(costs, "battery inverter", "investment") * 1000
battery_inverter_lifetime = get_cost(costs, "battery inverter", "lifetime")
battery_inverter_fom = get_cost(costs, "battery inverter", "FOM") / 100 * battery_inverter_investment

battery_storage_investment = get_cost(costs, "battery storage", "investment") * 1000
battery_storage_lifetime = get_cost(costs, "battery storage", "lifetime")

battery_inverter_cost = (
    battery_inverter_investment * annuity(DISCOUNT_RATE, battery_inverter_lifetime)
    + battery_inverter_fom
)

battery_storage_cost = (
    battery_storage_investment * annuity(DISCOUNT_RATE, battery_storage_lifetime)
)

battery_capital_cost = battery_inverter_cost + battery_storage_cost

print("\nTask F cost check:")
print("Solar capital cost [€/MW/year]:", round(solar_capital_cost, 2))
print("Onshore capital cost [€/MW/year]:", round(onshore_capital_cost, 2))
print("Offshore capital cost [€/MW/year]:", round(offshore_capital_cost, 2))
print("OCGT capital cost [€/MW/year]:", round(ocgt_capital_cost, 2))
print("OCGT marginal cost without CO2 price [€/MWh]:", round(ocgt_mc_taskf, 2))
print("Battery capital cost [€/MW/year]:", round(battery_capital_cost, 2))

# ============================================================
# Build Task F network
# ============================================================

def build_task_f_network(co2_cap=None):
    network = create_network(snapshots)

    if "AC" not in network.carriers.index:
        network.add("Carrier", "AC")

    if "gas_fuel" not in network.carriers.index:
        network.add("Carrier", "gas_fuel", co2_emissions=CO2_INTENSITY)

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
        marginal_cost=0.0,
    )

    network.add(
        "Generator",
        "onshore_wind",
        bus=ZONE,
        p_nom_extendable=True,
        p_max_pu=onshore_cf,
        capital_cost=onshore_capital_cost,
        marginal_cost=0.0,
    )

    network.add(
        "Generator",
        "offshore_wind",
        bus=ZONE,
        p_nom_extendable=True,
        p_max_pu=offshore_cf,
        capital_cost=offshore_capital_cost,
        marginal_cost=0.0,
    )

    network.add(
        "Generator",
        "ocgt",
        bus=ZONE,
        carrier="gas_fuel",
        p_nom_extendable=True,
        capital_cost=ocgt_capital_cost,
        marginal_cost=ocgt_mc_taskf,
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

    if co2_cap is not None:
        network.add(
            "GlobalConstraint",
            "CO2Limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=float(co2_cap),
        )

    return network


# ============================================================
# Baseline run
# ============================================================

network_base = build_task_f_network(co2_cap=None)
network_base.optimize()

baseline_ocgt_generation = network_base.generators_t.p["ocgt"].sum()
baseline_emissions = baseline_ocgt_generation / ocgt_efficiency * CO2_INTENSITY

print("\nBaseline results:")
print("Baseline OCGT generation [MWh_el]:", round(baseline_ocgt_generation, 2))
print("Baseline emissions [tCO2]:", round(baseline_emissions, 2))

print("\nBaseline capacities [MW]:")
print(network_base.generators["p_nom_opt"].round(2))
print("\nBaseline battery power [MW]:")
print(network_base.storage_units["p_nom_opt"].round(2))

# ============================================================
# Scenario analysis
# ============================================================

cap_fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]
results_f = []

for frac in cap_fractions:
    print(f"\nRunning Task F for cap = {frac:.0%} of baseline emissions")

    co2_cap = frac * baseline_emissions
    network = build_task_f_network(co2_cap=co2_cap)
    network.optimize()

    ocgt_gen = network.generators_t.p["ocgt"].sum()
    emissions = ocgt_gen / ocgt_efficiency * CO2_INTENSITY

    results_f.append({
        "cap_fraction": frac,
        "cap_percent": frac * 100,
        "co2_cap_tCO2": co2_cap,
        "system_cost": network.objective,
        "emissions_tCO2": emissions,
        "solar_cap_MW": network.generators.at["solar", "p_nom_opt"],
        "onshore_cap_MW": network.generators.at["onshore_wind", "p_nom_opt"],
        "offshore_cap_MW": network.generators.at["offshore_wind", "p_nom_opt"],
        "ocgt_cap_MW": network.generators.at["ocgt", "p_nom_opt"],
        "battery_cap_MW": network.storage_units.at["battery", "p_nom_opt"],
        "solar_gen_MWh": network.generators_t.p["solar"].sum(),
        "onshore_gen_MWh": network.generators_t.p["onshore_wind"].sum(),
        "offshore_gen_MWh": network.generators_t.p["offshore_wind"].sum(),
        "ocgt_gen_MWh": ocgt_gen,
    })

results_f_df = pd.DataFrame(results_f)
results_f_df = results_f_df.sort_values("cap_percent", ascending=False).reset_index(drop=True)

print("\nTask F results:")
print(results_f_df.round(2))

# ============================================================
# Save results
# ============================================================

results_f_df.to_csv(os.path.join(DATA_DIR, "task_f_results.csv"), index=False)

x_labels = results_f_df["cap_percent"].astype(int).astype(str)

# ============================================================
# Plot 1: capacities vs CO2 cap
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(x_labels, results_f_df["solar_cap_MW"], marker="o", label="solar_cap_MW")
plt.plot(x_labels, results_f_df["onshore_cap_MW"], marker="o", label="onshore_cap_MW")
plt.plot(x_labels, results_f_df["offshore_cap_MW"], marker="o", label="offshore_cap_MW")
plt.plot(x_labels, results_f_df["ocgt_cap_MW"], marker="o", label="ocgt_cap_MW")
plt.plot(x_labels, results_f_df["battery_cap_MW"], marker="o", label="battery_cap_MW")

plt.xlabel("CO2 cap (% of baseline emissions)")
plt.ylabel("Installed capacity (MW)")
plt.title("Task F: Optimal capacities vs CO2 cap")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "task_f_capacities_vs_co2cap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# Plot 2: generation mix vs CO2 cap
# ============================================================

plot_df_gen = results_f_df.set_index(x_labels)[
    ["solar_gen_MWh", "onshore_gen_MWh", "offshore_gen_MWh", "ocgt_gen_MWh"]
]

ax = plot_df_gen.plot(kind="bar", stacked=True, figsize=(10, 5))
ax.set_xlabel("CO2 cap (% of baseline emissions)")
ax.set_ylabel("Annual generation (MWh)")
ax.set_title("Task F: Annual generation mix vs CO2 cap")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "task_f_generation_mix_vs_co2cap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# Plot 3: system cost vs CO2 cap
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(x_labels, results_f_df["system_cost"], marker="o")

plt.xlabel("CO2 cap (% of baseline emissions)")
plt.ylabel("Objective value (€)")
plt.title("Task F: System cost vs CO2 cap")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "task_f_system_cost_vs_co2cap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()