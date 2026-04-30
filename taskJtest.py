import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

pd.options.future.infer_string = False
pd.options.mode.string_storage = "python"

from src.cost_loader import load_costs

from taskD import (
    YEAR,
    COST_YEAR,
    DATA_DIR,
    FIG_DIR,
    ZONES,
    build_network,
    hydro_energy_constraint_no2,
    pivot_load,
    pivot_generation,
    compute_p_max_pu,
    align_to_snapshots,
    annualized_fixed_cost,
)


# ============================================================
# Settings
# ============================================================

GRIB_FILE = os.path.join(DATA_DIR, "weather_data_2025")

T_BASE = 15.0

HEAT_DEMAND_SHARE = {
    "DK1": 0.25,
    "DK2": 0.25,
    "NO2": 0.20,
    "DE": 0.60,
}

HEAT_PUMP_COP = 3.0
HEAT_PUMP_CAPITAL_COST = 80_000

BORNHOLM_BUS = "bus_Bornholm"
BORNHOLM_OFFSHORE_CAPACITY = 3800.0

BORNHOLM_LINES = {
    "Bornholm_DK2": {
        "bus0": BORNHOLM_BUS,
        "bus1": "bus_DK2",
        "s_nom": 1200.0,
    },
    "Bornholm_DE": {
        "bus0": BORNHOLM_BUS,
        "bus1": "bus_DE",
        "s_nom": 2000.0,
    },
}


# ============================================================
# Heat demand profile
# ============================================================

def load_temperature_profile(path):
    ds = xr.open_dataset(path, engine="cfgrib")

    temperature_c = ds["t2m"] - 273.15
    temperature_country = temperature_c.mean(dim=["latitude", "longitude"]).to_pandas()

    if temperature_country.index.tz is not None:
        temperature_country.index = (
            temperature_country.index
            .tz_convert("UTC")
            .tz_localize(None)
        )

    return temperature_country


def build_heat_profile(temperature_c, snapshots):
    heating_degree_hours = (T_BASE - temperature_c).clip(lower=0.0)

    heating_degree_hours = (
        heating_degree_hours
        .groupby(heating_degree_hours.index)
        .first()
        .reindex(snapshots)
        .interpolate()
        .ffill()
        .bfill()
    )

    if heating_degree_hours.sum() <= 0:
        raise ValueError("Heating degree hours are zero. Check temperature data.")

    heat_profile = heating_degree_hours / heating_degree_hours.sum()

    return heat_profile.astype("float64")


def build_zone_heat_demand(zone, heat_profile, snapshots):
    electricity_load = pivot_load(zone)

    electricity_load = (
        electricity_load
        .groupby(electricity_load.index)
        .first()
        .reindex(snapshots)
        .ffill()
        .fillna(0.0)
    )

    annual_electricity_demand = electricity_load.sum()
    annual_heat_demand = HEAT_DEMAND_SHARE[zone] * annual_electricity_demand

    heat_demand = heat_profile * annual_heat_demand
    heat_demand = heat_demand.reindex(snapshots).fillna(0.0)

    return heat_demand.astype("float64")


# ============================================================
# Add heat sector
# ============================================================

def add_heat_sector(network, heat_profile):
    for carrier in ["heat", "heat_pump"]:
        if carrier not in network.carriers.index:
            network.add("Carrier", carrier)

    for zone in ZONES:
        heat_bus = f"heat_{zone}"

        heat_demand = build_zone_heat_demand(
            zone=zone,
            heat_profile=heat_profile,
            snapshots=network.snapshots,
        )

        network.add(
            "Bus",
            heat_bus,
            carrier="heat",
        )

        network.add(
            "Load",
            f"heat_load_{zone}",
            bus=heat_bus,
            p_set=heat_demand,
        )

        network.add(
            "Link",
            f"heat_pump_{zone}",
            bus0=f"bus_{zone}",
            bus1=heat_bus,
            carrier="heat_pump",
            efficiency=HEAT_PUMP_COP,
            p_nom_extendable=True,
            capital_cost=HEAT_PUMP_CAPITAL_COST,
            marginal_cost=0.0,
        )

    return network


# ============================================================
# Add Bornholm energy island
# ============================================================

def build_bornholm_offshore_profile(network):
    gen_dk2 = pivot_generation("DK2")

    offshore_profile = compute_p_max_pu(gen_dk2, "wind_off")
    offshore_profile = align_to_snapshots(offshore_profile, network)

    return offshore_profile


def add_bornholm_energy_island(network, costs):
    for carrier in ["AC", "offwind"]:
        if carrier not in network.carriers.index:
            network.add("Carrier", carrier)

    network.add(
        "Bus",
        BORNHOLM_BUS,
        carrier="AC",
    )

    offshore_profile = build_bornholm_offshore_profile(network)

    network.add(
        "Generator",
        "offshore_wind_Bornholm",
        bus=BORNHOLM_BUS,
        carrier="offwind",
        p_nom=BORNHOLM_OFFSHORE_CAPACITY,
        p_nom_extendable=False,
        p_max_pu=offshore_profile,
        capital_cost=annualized_fixed_cost(costs, "offwind"),
        marginal_cost=0.0,
    )

    for line_name, line in BORNHOLM_LINES.items():
        network.add(
            "Line",
            line_name,
            bus0=line["bus0"],
            bus1=line["bus1"],
            carrier="AC",
            x=0.1,
            r=0.0001,
            s_nom=line["s_nom"],
        )

    return network


# ============================================================
# Build model
# ============================================================

def build_task_j_model(costs, heat_profile, include_bornholm=False):
    network = build_network(costs)
    network = add_heat_sector(network, heat_profile)

    if include_bornholm:
        network = add_bornholm_energy_island(network, costs)

    return network


def optimise(network):
    network.optimize(extra_functionality=hydro_energy_constraint_no2)
    return network


# ============================================================
# Results
# ============================================================

def generation_by_carrier(network):
    return (
        network.generators_t.p
        .sum()
        .groupby(network.generators.carrier)
        .sum()
        .sort_values(ascending=False)
    )


def capacity_by_carrier(network):
    return (
        network.generators
        .assign(capacity_MW=network.generators["p_nom_opt"].clip(lower=0.0))
        .groupby("carrier")["capacity_MW"]
        .sum()
        .sort_values(ascending=False)
    )


def compare_results(base_network, bornholm_network):
    capacity_comparison = pd.DataFrame({
        "Base system [MW]": capacity_by_carrier(base_network),
        "With Bornholm [MW]": capacity_by_carrier(bornholm_network),
    }).fillna(0.0)

    capacity_comparison["Difference [MW]"] = (
        capacity_comparison["With Bornholm [MW]"]
        - capacity_comparison["Base system [MW]"]
    )

    generation_comparison = pd.DataFrame({
        "Base system [MWh]": generation_by_carrier(base_network),
        "With Bornholm [MWh]": generation_by_carrier(bornholm_network),
    }).fillna(0.0)

    generation_comparison["Difference [MWh]"] = (
        generation_comparison["With Bornholm [MWh]"]
        - generation_comparison["Base system [MWh]"]
    )

    objective_comparison = pd.DataFrame({
        "Objective [EUR/year]": {
            "Base system": base_network.objective,
            "With Bornholm": bornholm_network.objective,
            "Difference": bornholm_network.objective - base_network.objective,
        }
    })

    print("\n--- Capacity comparison ---")
    print(capacity_comparison.round(2))

    print("\n--- Generation comparison ---")
    print(generation_comparison.round(2))

    print("\n--- Objective comparison ---")
    print(objective_comparison.round(2))

    capacity_comparison.to_csv(os.path.join(DATA_DIR, "taskJ_capacity_comparison.csv"))
    generation_comparison.to_csv(os.path.join(DATA_DIR, "taskJ_generation_comparison.csv"))
    objective_comparison.to_csv(os.path.join(DATA_DIR, "taskJ_objective_comparison.csv"))


def print_bornholm_results(network):
    print("\n--- Bornholm energy island ---")
    print(
        network.generators.loc[
            ["offshore_wind_Bornholm"],
            ["bus", "carrier", "p_nom", "p_nom_opt", "capital_cost"],
        ].round(2)
    )

    bornholm_generation = network.generators_t.p["offshore_wind_Bornholm"].sum()
    bornholm_cf = bornholm_generation / (
        BORNHOLM_OFFSHORE_CAPACITY * len(network.snapshots)
    )

    print("\n--- Bornholm annual generation ---")
    print(f"Bornholm offshore generation: {bornholm_generation:,.0f} MWh")
    print(f"Bornholm offshore capacity factor: {bornholm_cf:.3f}")

    bornholm_flows = network.lines_t.p0[["Bornholm_DK2", "Bornholm_DE"]]

    annual_exports = bornholm_flows.clip(lower=0).sum()
    annual_imports = -bornholm_flows.clip(upper=0).sum()
    max_loading = (
        bornholm_flows.abs().max()
        / network.lines.loc[["Bornholm_DK2", "Bornholm_DE"], "s_nom"]
        * 100
    )

    line_summary = pd.DataFrame({
        "Annual export from Bornholm [MWh]": annual_exports,
        "Annual import to Bornholm [MWh]": annual_imports,
        "Max loading [%]": max_loading,
    })

    print("\n--- Bornholm line use ---")
    print(line_summary.round(2))

    line_summary.to_csv(os.path.join(DATA_DIR, "taskJ_bornholm_line_summary.csv"))


# ============================================================
# Plots
# ============================================================

def plot_generation_comparison(base_network, bornholm_network):
    comparison = pd.DataFrame({
        "Base system": generation_by_carrier(base_network),
        "With Bornholm": generation_by_carrier(bornholm_network),
    }).fillna(0.0)

    comparison.plot(kind="bar", figsize=(9, 5))

    plt.ylabel("Annual generation [MWh]")
    plt.title("Annual generation with and without Bornholm energy island")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_generation_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_capacity_comparison(base_network, bornholm_network):
    comparison = pd.DataFrame({
        "Base system": capacity_by_carrier(base_network),
        "With Bornholm": capacity_by_carrier(bornholm_network),
    }).fillna(0.0)

    comparison.plot(kind="bar", figsize=(9, 5))

    plt.ylabel("Installed capacity [MW]")
    plt.title("Installed capacity with and without Bornholm energy island")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_capacity_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_bornholm_export_duration(network):
    flows = network.lines_t.p0[["Bornholm_DK2", "Bornholm_DE"]].copy()
    total_export = flows.sum(axis=1)

    total_export.sort_values(ascending=False).reset_index(drop=True).plot(figsize=(8, 5))

    plt.ylabel("Bornholm net export [MW]")
    plt.xlabel("Hour rank")
    plt.title("Bornholm export duration curve")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_bornholm_export_duration_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_bornholm_line_flows(network):
    network.lines_t.p0[["Bornholm_DK2", "Bornholm_DE"]].plot(figsize=(12, 5))

    plt.axhline(0, linewidth=1)
    plt.ylabel("Flow [MW]")
    plt.title("Bornholm transmission line flows")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_bornholm_line_flows.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_prices(network):
    price_buses = ["bus_DK1", "bus_DK2", "bus_DE", "bus_NO2", "bus_Bornholm"]
    price_buses = [bus for bus in price_buses if bus in network.buses_t.marginal_price.columns]

    network.buses_t.marginal_price[price_buses].plot(figsize=(12, 5))

    plt.ylabel("Price [EUR/MWh]")
    plt.title("Electricity prices with Bornholm energy island")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_prices.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_heat_pump_capacities(network):
    heat_pumps = network.links[network.links.carrier == "heat_pump"]

    capacities = heat_pumps["p_nom_opt"].copy()
    capacities.index = capacities.index.str.replace("heat_pump_", "", regex=False)

    capacities.plot(kind="bar", figsize=(7, 5))

    plt.ylabel("Heat pump capacity [MW_el]")
    plt.title("Optimised heat pump capacities")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_heat_pump_capacities.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_electricity_load_with_heating(network, zone, period):
    base_load = network.loads_t.p_set[f"load_{zone}"].loc[period]
    hp_load = network.links_t.p0[f"heat_pump_{zone}"].loc[period]

    fig, ax = plt.subplots(figsize=(12, 4))

    base_load.plot(ax=ax, color="black", linewidth=2, label="Electricity load")
    (base_load + hp_load).plot(
        ax=ax,
        linestyle="--",
        linewidth=2,
        label="Electricity load + heat pump load",
    )

    ax.set_ylabel("Electricity [MW]")
    ax.set_title(f"Electricity demand with heat integration - {zone}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"taskJ_electricity_load_with_heating_{zone}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# ============================================================
# Run Task J
# ============================================================

if __name__ == "__main__":

    costs = load_costs(os.path.join(DATA_DIR, "costs_PyPSA.csv"), year=COST_YEAR)

    temperature = load_temperature_profile(GRIB_FILE)

    temp_network = build_network(costs)
    heat_profile = build_heat_profile(temperature, temp_network.snapshots)

    print("\n--- Optimising base system without Bornholm ---")
    base_network = build_task_j_model(
        costs=costs,
        heat_profile=heat_profile,
        include_bornholm=False,
    )

    base_network = optimise(base_network)

    print("\n--- Optimising system with Bornholm energy island ---")
    bornholm_network = build_task_j_model(
        costs=costs,
        heat_profile=heat_profile,
        include_bornholm=True,
    )

    print("\n--- Task J network overview with Bornholm ---")
    print("Buses:", len(bornholm_network.buses))
    print("Loads:", len(bornholm_network.loads))
    print("Generators:", len(bornholm_network.generators))
    print("Links:", len(bornholm_network.links))
    print("Lines:", len(bornholm_network.lines))
    print("Storage units:", len(bornholm_network.storage_units))

    bornholm_network = optimise(bornholm_network)

    compare_results(base_network, bornholm_network)
    print_bornholm_results(bornholm_network)

    winter_week = slice(f"{YEAR}-01-13", f"{YEAR}-01-19 23:00")

    plot_generation_comparison(base_network, bornholm_network)
    plot_capacity_comparison(base_network, bornholm_network)
    plot_bornholm_export_duration(bornholm_network)
    plot_bornholm_line_flows(bornholm_network)
    plot_prices(bornholm_network)
    plot_heat_pump_capacities(bornholm_network)

    for zone in ZONES:
        plot_electricity_load_with_heating(bornholm_network, zone, winter_week)