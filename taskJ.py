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
BORNHOLM_OFFSHORE_CAPACITY = 3800.0  # MW

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


def build_heat_profile(temperature_c):
    heating_degree_hours = (T_BASE - temperature_c).clip(lower=0.0)

    if heating_degree_hours.sum() <= 0:
        raise ValueError("Heating degree hours are zero. Check temperature data.")

    heat_profile = heating_degree_hours / heating_degree_hours.sum()

    if not np.isclose(heat_profile.sum(), 1.0):
        raise ValueError("Heat profile does not sum to 1.")

    return heat_profile


def build_zone_heat_demand(zone, heat_profile, snapshots):
    electricity_load = pivot_load(zone)
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
    """
    Uses the DK2 offshore wind profile as proxy for Bornholm.
    This is reasonable because Bornholm is electrically connected to DK2
    and located in the Baltic Sea close to the DK2 region.
    """
    gen_dk2 = pivot_generation("DK2")

    offshore_profile = compute_p_max_pu(gen_dk2, "wind_off")
    offshore_profile = align_to_snapshots(offshore_profile, network)

    return offshore_profile


def add_bornholm_energy_island(network, costs):
    if "offwind" not in network.carriers.index:
        network.add("Carrier", "offwind")

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
# Results
# ============================================================
def print_task_j_results(network):
    print("\n--- Objective value ---")
    print(f"Total system cost: {network.objective:,.0f} EUR/year")

    print("\n--- Bornholm energy island ---")
    print(
        network.generators.loc[
            ["offshore_wind_Bornholm"],
            ["bus", "carrier", "p_nom", "p_nom_opt", "capital_cost"],
        ].round(2)
    )

    bornholm_generation = network.generators_t.p["offshore_wind_Bornholm"].sum()
    bornholm_cf = bornholm_generation / (BORNHOLM_OFFSHORE_CAPACITY * len(network.snapshots))

    print("\n--- Bornholm annual generation ---")
    print(f"Bornholm offshore generation: {bornholm_generation:,.0f} MWh")
    print(f"Bornholm offshore capacity factor: {bornholm_cf:.3f}")

    print("\n--- Bornholm transmission lines ---")
    print(
        network.lines.loc[
            ["Bornholm_DK2", "Bornholm_DE"],
            ["bus0", "bus1", "s_nom", "x", "r"],
        ]
    )

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

    print("\n--- Installed generator capacities by carrier [MW] ---")
    gen_capacity = (
        network.generators
        .groupby("carrier")["p_nom_opt"]
        .sum()
        .sort_values(ascending=False)
    )
    print(gen_capacity.round(2))

    print("\n--- Annual generation by carrier [MWh] ---")
    gen_mix = (
        network.generators_t.p
        .sum()
        .groupby(network.generators.carrier)
        .sum()
        .sort_values(ascending=False)
    )
    print(gen_mix.round(2))

    print("\n--- Storage capacities [MW] ---")
    print(network.storage_units[["bus", "p_nom_opt", "max_hours"]].round(2))

    print("\n--- Heat pump capacities [MW_el] ---")
    heat_pumps = network.links[network.links.carrier == "heat_pump"]
    print(heat_pumps[["bus0", "bus1", "p_nom_opt", "efficiency"]].round(2))

    print("\n--- Average electricity prices [EUR/MWh] ---")
    print(network.buses_t.marginal_price.mean().round(2))

    line_summary.to_csv(os.path.join(DATA_DIR, "taskJ_bornholm_line_summary.csv"))
    gen_capacity.to_csv(os.path.join(DATA_DIR, "taskJ_capacity_by_carrier.csv"))
    gen_mix.to_csv(os.path.join(DATA_DIR, "taskJ_generation_by_carrier.csv"))

# ============================================================
# Plots
# ============================================================
def plot_generation_mix(network):
    gen_mix = (
        network.generators_t.p
        .sum()
        .groupby(network.generators.carrier)
        .sum()
        .sort_values(ascending=False)
    )

    gen_mix.plot(kind="bar", figsize=(8, 5))

    plt.ylabel("Annual generation [MWh]")
    plt.title("Annual generation mix with Bornholm energy island")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_generation_mix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_installed_capacity_by_carrier(network):
    capacities = (
        network.generators
        .groupby("carrier")["p_nom_opt"]
        .sum()
        .sort_values(ascending=False)
    )

    capacities.plot(kind="bar", figsize=(8, 5))

    plt.ylabel("Installed capacity [MW]")
    plt.title("Installed generation capacity with Bornholm energy island")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_capacity_by_carrier.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_installed_capacity_by_country(network):
    gen_capacity = network.generators.copy()
    gen_capacity["capacity_MW"] = gen_capacity["p_nom_opt"].clip(lower=0)
    gen_capacity["country"] = gen_capacity["bus"].str.replace("bus_", "", regex=False)

    capacity_table = (
        gen_capacity
        .groupby(["country", "carrier"])["capacity_MW"]
        .sum()
        .unstack(fill_value=0)
    )

    capacity_table.plot(kind="bar", stacked=True, figsize=(10, 6))

    plt.ylabel("Installed capacity [MW]")
    plt.title("Installed generation capacity by country with Bornholm")
    plt.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_capacity_by_country.png"),
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


def plot_prices(network):
    price_buses = ["bus_DK1", "bus_DK2", "bus_DE", "bus_NO2", "bus_Bornholm"]
    prices = network.buses_t.marginal_price[price_buses]

    prices.plot(figsize=(12, 5))

    plt.ylabel("Price [EUR/MWh]")
    plt.title("Electricity prices with Bornholm energy island")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskJ_prices.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# ============================================================
# Run Task J
# ============================================================

costs = load_costs(os.path.join(DATA_DIR, "costs_PyPSA.csv"), year=COST_YEAR)

temperature = load_temperature_profile(GRIB_FILE)
heat_profile = build_heat_profile(temperature)

network = build_network(costs)
network = add_heat_sector(network, heat_profile)
network = add_bornholm_energy_island(network, costs)

print("\n--- Task J network overview ---")
print("Buses:", len(network.buses))
print("Loads:", len(network.loads))
print("Generators:", len(network.generators))
print("Links:", len(network.links))
print("Lines:", len(network.lines))
print("Storage units:", len(network.storage_units))

network.optimize(
    extra_functionality=hydro_energy_constraint_no2,
)

print_task_j_results(network)

winter_week = slice(f"{YEAR}-01-13", f"{YEAR}-01-19 23:00")
summer_week = slice(f"{YEAR}-07-07", f"{YEAR}-07-13 23:00")

'''
plot_bornholm_generation(network)
plot_bornholm_line_flows(network)
plot_bornholm_week(network, winter_week)
plot_bornholm_week(network, summer_week)
'''

plot_generation_mix(network)
plot_installed_capacity_by_carrier(network)
plot_installed_capacity_by_country(network)
plot_bornholm_export_duration(network)
plot_prices(network)

plot_heat_pump_capacities(network)

for zone in ZONES:
    plot_electricity_load_with_heating(network, zone, winter_week)