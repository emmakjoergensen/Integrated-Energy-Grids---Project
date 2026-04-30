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
    pivot_load,
)

from taskG import hydro_energy_constraint_no2

from taskH import (
    build_task_g_model,
    calculate_emissions,
    CO2_LIMIT_SHARE,
    co2_limit_constraint,
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

    # Fix duplicate timestamps before reindexing
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
    heat_demand = (
        heat_demand
        .groupby(heat_demand.index)
        .first()
        .reindex(snapshots)
        .interpolate()
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    return heat_demand.astype("float64")


# ============================================================
# Add heat sector
# ============================================================

def add_heat_sector(network, heat_profile):
    if "heat" not in network.carriers.index:
        network.add("Carrier", "heat")

    if "heat_pump" not in network.carriers.index:
        network.add("Carrier", "heat_pump")

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
# Optimisation with both hydro and CO2 constraint
# ============================================================

def combined_extra_functionality(network, snapshots):
    hydro_energy_constraint_no2(network, snapshots)
    co2_limit_constraint(network, snapshots)


def optimise_task_i_network(network, co2_limit):
    network.co2_limit = co2_limit

    network.optimize(
        extra_functionality=combined_extra_functionality,
    )

    return network


# ============================================================
# Results
# ============================================================

def print_heat_results(network):
    heat_pumps = network.links[network.links.carrier == "heat_pump"].copy()

    print("\n--- Heat pump capacities ---")
    print(heat_pumps[["bus0", "bus1", "p_nom_opt", "efficiency"]].round(2))

    annual_heat_demand = {}
    annual_hp_electricity = {}

    for zone in ZONES:
        annual_heat_demand[zone] = network.loads_t.p_set[f"heat_load_{zone}"].sum()
        annual_hp_electricity[zone] = network.links_t.p0[f"heat_pump_{zone}"].sum()

    heat_summary = pd.DataFrame({
        "Annual heat demand [MWh_th]": annual_heat_demand,
        "Heat pump electricity use [MWh_el]": annual_hp_electricity,
    })

    print("\n--- Annual heat sector summary ---")
    print(heat_summary.round(2))

    heat_summary.to_csv(os.path.join(DATA_DIR, "taskI_heat_summary.csv"))


def print_capacity_results(network):
    print("\n--- Optimised generator capacities [MW] ---")
    print(network.generators[["bus", "carrier", "p_nom_opt"]].round(2))

    print("\n--- Optimised link capacities [MW] ---")
    print(network.links[["bus0", "bus1", "carrier", "p_nom_opt"]].round(2))


# ============================================================
# Plots
# ============================================================

def plot_heat_dispatch(network, zone, period):
    heat_load = network.loads_t.p_set[f"heat_load_{zone}"].loc[period]
    heat_output = -network.links_t.p1[f"heat_pump_{zone}"].loc[period]

    fig, ax = plt.subplots(figsize=(12, 4))

    heat_output.plot.area(ax=ax, alpha=0.7, label="Heat pump output")
    heat_load.plot(ax=ax, color="black", linewidth=2, label="Heat demand")

    ax.set_ylabel("Heat [MW_th]")
    ax.set_title(f"Heat sector dispatch - {zone}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"taskI_heat_dispatch_{zone}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_heat_pump_electricity_use(network, zone, period):
    hp_electricity = network.links_t.p0[f"heat_pump_{zone}"].loc[period]

    fig, ax = plt.subplots(figsize=(12, 4))

    hp_electricity.plot(ax=ax)

    ax.set_ylabel("Electricity [MW_el]")
    ax.set_title(f"Electricity consumption of heat pumps - {zone}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"taskI_heat_pump_electricity_{zone}.png"),
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
        os.path.join(FIG_DIR, f"taskI_electricity_load_with_heating_{zone}.png"),
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
        os.path.join(FIG_DIR, "taskI_heat_pump_capacities.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# ============================================================
# Run Task I
# ============================================================

if __name__ == "__main__":

    costs = load_costs(os.path.join(DATA_DIR, "costs_PyPSA.csv"), year=COST_YEAR)

    # --------------------------------------------------------
    # Build baseline Task G model to calculate Task H CO2 limit
    # --------------------------------------------------------

    baseline = build_task_g_model(costs)
    baseline.optimize(extra_functionality=hydro_energy_constraint_no2)

    baseline_emissions = calculate_emissions(baseline)
    co2_limit = CO2_LIMIT_SHARE * baseline_emissions["total"]

    print("\n--- Task I CO2 limit from Task H ---")
    print(f"Baseline emissions: {baseline_emissions['total']:,.0f} tCO2")
    print(f"CO2 limit: {co2_limit:,.0f} tCO2")
    print(f"CO2 limit share: {CO2_LIMIT_SHARE:.0%}")

    # --------------------------------------------------------
    # Build Task I model: Task G/H system + heat sector
    # --------------------------------------------------------

    network = build_task_g_model(costs)

    temperature = load_temperature_profile(GRIB_FILE)
    heat_profile = build_heat_profile(temperature, network.snapshots)

    network = add_heat_sector(network, heat_profile)

    print("\n--- Task I network overview before optimisation ---")
    print("Buses:", len(network.buses))
    print("Loads:", len(network.loads))
    print("Generators:", len(network.generators))
    print("Links:", len(network.links))
    print("Lines:", len(network.lines))
    print("Storage units:", len(network.storage_units))

    network = optimise_task_i_network(network, co2_limit)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    task_i_emissions = calculate_emissions(network)

    print("\n--- Task I emissions ---")
    print(f"Gas emissions:   {task_i_emissions['gas']:,.0f} tCO2")
    print(f"Coal emissions:  {task_i_emissions['coal']:,.0f} tCO2")
    print(f"Total emissions: {task_i_emissions['total']:,.0f} tCO2")

    print_heat_results(network)
    print_capacity_results(network)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    winter_week = slice(f"{YEAR}-01-13", f"{YEAR}-01-19 23:00")

    plot_heat_pump_capacities(network)

    for zone in ZONES:
        plot_heat_dispatch(network, zone, winter_week)
        plot_heat_pump_electricity_use(network, zone, winter_week)
        plot_electricity_load_with_heating(network, zone, winter_week)