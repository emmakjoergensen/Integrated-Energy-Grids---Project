import os
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

pd.options.future.infer_string = False
pd.options.mode.string_storage = "python"

from src.config import DISCOUNT_RATE
from src.cost_loader import load_costs, get_cost
from src.network_builder import create_network


# ============================================================
# Settings
# ============================================================

YEAR = 2025
COST_YEAR = 2030

DATA_DIR = "Data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
FIG_DIR = "Figures"

os.makedirs(FIG_DIR, exist_ok=True)

ZONES = ["DK1", "DK2", "NO2", "DE"]

TECH_ORDER = [
    "onwind",
    "offwind",
    "solar",
    "hydro",
    "gas",
    "coal",
    "battery",
]

TECH_COLORS = {
    "onwind":  "#1f77b4",  # blue
    "offwind": "#ff7f0e",  # orange
    "solar":   "#2ca02c",  # green
    "hydro":   "#9467bd",  # purple
    "gas":     "#d62728",  # red
    "coal":    "#8c564b",  # brown
    "battery": "#17becf",  # teal
}

PSR_MAP = {
    "wind_on": ["B19"],
    "wind_off": ["B18"],
    "solar": ["B16"],
    "hydro_ror": ["B11"],
    "hydro_res": ["B12"],
    "coal": ["B02", "B03", "B05"],
    "gas": ["B04"],
}

CF_TARGET = {
    "solar": 0.11,
    "wind_on": 0.25,
    "wind_off": 0.52,
    "hydro_ror": 0.45,
}

BATTERY = {
    "duration": 4.0,
    "marginal_cost": 0.0,
}

LINES = {
    "DK1_NO2": {"bus0": "bus_DK1", "bus1": "bus_NO2", "s_nom": 1700.0},
    "DK1_DE": {"bus0": "bus_DK1", "bus1": "bus_DE", "s_nom": 2500.0},
    "DK1_DK2": {"bus0": "bus_DK1", "bus1": "bus_DK2", "s_nom": 600.0},
    "DK2_DE": {"bus0": "bus_DK2", "bus1": "bus_DE", "s_nom": 600.0},
}


# ============================================================
# Cost functions
# ============================================================

def annuity(rate, lifetime):
    return rate / (1 - (1 + rate) ** (-lifetime))


def annualized_fixed_cost(costs, tech, discount_rate=DISCOUNT_RATE):
    investment = get_cost(costs, tech, "investment") * 1000
    lifetime = get_cost(costs, tech, "lifetime")
    fom = get_cost(costs, tech, "FOM") / 100 * investment
    return investment * annuity(discount_rate, lifetime) + fom


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


def coal_costs():
    coal_capex = 1500 * 1000
    coal_lifetime = 40
    coal_fom = 0.03 * coal_capex
    coal_efficiency = 0.38
    coal_vom = 3

    coal_price = 15.0
    co2_price = 80.0
    coal_co2_intensity = 0.34

    capital_cost = coal_capex * annuity(DISCOUNT_RATE, coal_lifetime) + coal_fom

    marginal_cost = (
        coal_price / coal_efficiency
        + co2_price * coal_co2_intensity / coal_efficiency
        + coal_vom
    )

    return capital_cost, marginal_cost


def battery_capital_cost(costs):
    inverter_cost = annualized_fixed_cost(costs, "battery inverter")
    storage_cost = annualized_cost_no_fom(costs, "battery storage")

    # Kept consistent with your Task D version.
    return inverter_cost + storage_cost


# ============================================================
# Data functions
# ============================================================

def pivot_generation(zone):
    path = os.path.join(RAW_DIR, "generation", f"gen_{zone}_{YEAR}.csv")

    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert(None)

    tech_series = []

    for tech, psr_codes in PSR_MAP.items():
        series = (
            df[df["psr_type"].isin(psr_codes)]
            .groupby("time_utc")["generation_MW"]
            .sum()
            .rename(tech)
        )
        tech_series.append(series)

    return pd.concat(tech_series, axis=1).fillna(0.0)


def pivot_load(zone):
    path = os.path.join(RAW_DIR, "load", f"load_{zone}_{YEAR}.csv")

    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert(None)

    return df.set_index("time_utc")["load_MW"].rename("load")


def align_to_snapshots(series, network):
    series = series.copy()

    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)

    if not series.index.is_unique:
        series = series.groupby(series.index).sum()

    return series.reindex(network.snapshots).fillna(0.0).astype("float64")


def compute_p_max_pu(gen, tech):
    generation = gen[tech].astype(float)

    if generation.max() <= 0:
        return pd.Series(0.0, index=gen.index, name=tech)

    profile = generation / generation.max()
    profile = profile / profile.mean() * CF_TARGET[tech]

    return profile.clip(lower=0.0, upper=1.0).astype("float64")


def mean_capacity_factors():
    rows = {}

    for zone in ZONES:
        gen = pivot_generation(zone)
        zone_result = {}

        for tech in ["solar", "wind_on", "wind_off", "hydro_ror"]:
            if tech in gen.columns and gen[tech].sum() > 0:
                zone_result[tech] = compute_p_max_pu(gen, tech).mean()

        rows[zone] = zone_result

    return pd.DataFrame(rows).T


# ============================================================
# Hydro energy constraint for NO2
# ============================================================

def get_hydro_energy_no2():
    gen_no2 = pivot_generation("NO2")
    return gen_no2["hydro_res"].sum()


def hydro_energy_constraint_no2(network, snapshots):
    if "hydro_res_NO2" not in network.generators.index:
        return

    model = network.model
    hydro_energy_no2 = get_hydro_energy_no2()

    p = model.variables["Generator-p"].loc[
        dict(name="hydro_res_NO2")
    ]

    weights = network.snapshot_weightings.generators

    model.add_constraints(
        (p * weights).sum(dim="snapshot") <= hydro_energy_no2,
        name="hydro_energy_limit_NO2",
    )


# ============================================================
# Network construction
# ============================================================

def add_zone(network, zone, costs):
    bus = f"bus_{zone}"

    network.add("Bus", bus, carrier="AC")

    load = align_to_snapshots(pivot_load(zone), network)

    network.add(
        "Load",
        f"load_{zone}",
        bus=bus,
        p_set=load,
    )

    gen = pivot_generation(zone)

    if "wind_on" in gen.columns and gen["wind_on"].sum() > 0:
        existing_capacity = gen["wind_on"].max()

        network.add(
            "Generator",
            f"wind_on_{zone}",
            bus=bus,
            carrier="onwind",
            p_nom_extendable=True,
            p_nom=existing_capacity,
            p_nom_min=existing_capacity,
            p_max_pu=align_to_snapshots(compute_p_max_pu(gen, "wind_on"), network),
            capital_cost=annualized_fixed_cost(costs, "onwind") * 1.5,
            marginal_cost=0.0,
        )

    if "wind_off" in gen.columns and gen["wind_off"].sum() > 0:
        existing_capacity = gen["wind_off"].max()

        network.add(
            "Generator",
            f"wind_off_{zone}",
            bus=bus,
            carrier="offwind",
            p_nom_extendable=True,
            p_nom=existing_capacity,
            p_nom_min=existing_capacity,
            p_max_pu=align_to_snapshots(compute_p_max_pu(gen, "wind_off"), network),
            capital_cost=annualized_fixed_cost(costs, "offwind"),
            marginal_cost=0.0,
        )

    if "solar" in gen.columns and gen["solar"].sum() > 0:
        existing_capacity = gen["solar"].max()

        network.add(
            "Generator",
            f"solar_{zone}",
            bus=bus,
            carrier="solar",
            p_nom_extendable=True,
            p_nom=existing_capacity,
            p_nom_min=existing_capacity,
            p_max_pu=align_to_snapshots(compute_p_max_pu(gen, "solar"), network),
            capital_cost=annualized_fixed_cost(costs, "solar"),
            marginal_cost=0.0,
        )

    existing_gas = gen["gas"].max() if "gas" in gen.columns else 0.0

    network.add(
        "Generator",
        f"gas_{zone}",
        bus=bus,
        carrier="gas",
        p_nom_extendable=True,
        p_nom=existing_gas,
        p_nom_min=existing_gas,
        capital_cost=annualized_fixed_cost(costs, "OCGT"),
        marginal_cost=ocgt_marginal_cost(costs),
        efficiency=get_cost(costs, "OCGT", "efficiency"),
    )

    network.add(
        "StorageUnit",
        f"battery_{zone}",
        bus=bus,
        carrier="battery",
        p_nom_extendable=True,
        max_hours=BATTERY["duration"],
        efficiency_store=get_cost(costs, "battery inverter", "efficiency"),
        efficiency_dispatch=get_cost(costs, "battery inverter", "efficiency"),
        capital_cost=battery_capital_cost(costs),
        marginal_cost=BATTERY["marginal_cost"],
        cyclic_state_of_charge=True,
    )

    if zone == "DE" and "coal" in gen.columns and gen["coal"].sum() > 0:
        coal_capital_cost, coal_marginal_cost = coal_costs()
        existing_coal = gen["coal"].max()

        network.add(
            "Generator",
            "coal_DE",
            bus=bus,
            carrier="coal",
            p_nom_extendable=True,
            p_nom=existing_coal,
            p_nom_min=existing_coal,
            capital_cost=coal_capital_cost,
            marginal_cost=coal_marginal_cost,
        )

    if zone == "NO2" and "hydro_res" in gen.columns and gen["hydro_res"].sum() > 0:
        existing_hydro = gen["hydro_res"].max()

        network.add(
            "Generator",
            "hydro_res_NO2",
            bus=bus,
            carrier="hydro",
            p_nom_extendable=False,
            p_nom=existing_hydro,
            capital_cost=annualized_fixed_cost(costs, "hydro"),
            marginal_cost=0.0,
        )


def add_lines(network):
    for name, line in LINES.items():
        network.add(
            "Line",
            name,
            bus0=line["bus0"],
            bus1=line["bus1"],
            carrier="AC",
            x=0.1,
            r=0.0001,
            s_nom=line["s_nom"],
        )


def build_network(costs):
    snapshots = pd.date_range(
        f"{YEAR}-01-01 00:00",
        f"{YEAR}-12-31 23:00",
        freq="h",
    )

    network = create_network(snapshots)

    for carrier in ["AC", "onwind", "offwind", "solar", "gas", "coal", "hydro", "battery"]:
        network.add("Carrier", carrier)

    for zone in ZONES:
        add_zone(network, zone, costs)

    network.buses.loc["bus_DK1", "slack"] = True

    add_lines(network)

    return network


# ============================================================
# Results and plotting
# ============================================================

def plot_installed_capacity_percent(network):
    # --- Installed generator capacities ---
    gen = network.generators.copy()
    gen["capacity_GW"] = gen["p_nom_opt"].clip(lower=0) / 1e3
    gen["zone"] = gen["bus"].str.replace("bus_", "", regex=False)

    gen_cap = (
        gen
        .groupby(["zone", "carrier"])["capacity_GW"]
        .sum()
        .unstack(fill_value=0.0)
    )

    # --- Battery power capacity ---
    storage = network.storage_units.copy()
    storage["capacity_GW"] = storage["p_nom_opt"].clip(lower=0) / 1e3
    storage["zone"] = storage["bus"].str.replace("bus_", "", regex=False)
    storage["carrier"] = "battery"

    storage_cap = (
        storage
        .groupby(["zone", "carrier"])["capacity_GW"]
        .sum()
        .unstack(fill_value=0.0)
    )

    # --- Combine ---
    cap = gen_cap.add(storage_cap, fill_value=0.0)

    # --- Convert to percentages ---
    cap_pct = cap.div(cap.sum(axis=1), axis=0) * 100

    # --- Enforce common order and colours ---
    cap_pct = cap_pct.reindex(columns=TECH_ORDER, fill_value=0.0)
    colors = [TECH_COLORS[c] for c in cap_pct.columns]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cap_pct.plot(kind="bar", stacked=True, ax=ax, color=colors)

    # --- Absolute capacity labels ---
    for i, zone in enumerate(cap.index):
        ax.text(
            i,
            102,
            f"{cap.loc[zone].sum():.1f} GW",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Share of installed capacity [%]")
    ax.set_xlabel("Zone")
    ax.set_ylim(0, 110)
    ax.set_title("Installed generation capacity by zone")
    ax.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "taskD_installed_capacity_percent.png"),
        dpi=300
    )
    plt.close(fig)


def plot_battery_behavior(network, zone):
    name = f"battery_{zone}"

    if name not in network.storage_units.index:
        return

    battery_dispatch = network.storage_units_t.p[name]
    battery_soc = network.storage_units_t.state_of_charge[name]

    fig, ax1 = plt.subplots(figsize=(12, 4))

    battery_dispatch.plot(ax=ax1, label="Battery dispatch [MW]")
    ax1.axhline(0, linewidth=1)
    ax1.set_ylabel("MW")

    ax2 = ax1.twinx()
    battery_soc.plot(ax=ax2, linestyle="--", label="State of charge [MWh]")
    ax2.set_ylabel("MWh")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.title(f"Battery behavior in {zone} in {YEAR}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"taskD_battery_behavior_{zone}.png"), dpi=300)
    plt.close(fig)


def plot_line_flows(network):
    network.lines_t.p0.plot(figsize=(12, 5))

    plt.axhline(0, linewidth=1)
    plt.ylabel("MW")
    plt.title("HVAC line flows")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, "taskD_line_flows.png"), dpi=300)
    plt.close(fig)


def print_results(network):
    print("\n--- Generators ---")
    print(network.generators[[
        "bus", "carrier", "p_nom", "p_nom_min",
        "p_nom_extendable", "p_nom_opt",
        "capital_cost", "marginal_cost"
    ]].round(2))

    print("\n--- Storage units ---")
    print(network.storage_units[[
        "bus", "carrier", "p_nom", "p_nom_opt", "max_hours", "capital_cost"
    ]].round(2))

    battery_power = network.storage_units["p_nom_opt"]
    battery_energy = battery_power * network.storage_units["max_hours"]

    battery_summary = pd.DataFrame({
        "Power capacity [MW]": battery_power,
        "Energy capacity [MWh]": battery_energy,
    })

    print("\n--- Battery power and energy capacity ---")
    print(battery_summary.round(2))

    print("\n--- Lines ---")
    print(network.lines[["bus0", "bus1", "x", "r", "s_nom"]])

    print("\n--- Annual generation by technology [MWh] ---")
    gen_by_carrier = (
        network.generators_t.p
        .sum()
        .groupby(network.generators.carrier)
        .sum()
    )
    print(gen_by_carrier.round(2))

    print("\n--- Annual generation by country [MWh] ---")
    gen_by_country = (
        network.generators_t.p
        .sum()
        .groupby(network.generators.bus)
        .sum()
    )
    print(gen_by_country.round(2))

    print("\n--- Max line loading [%] ---")
    line_loading = network.lines_t.p0.abs().max() / network.lines.s_nom * 100
    print(line_loading.round(2))

    print("\n--- Average annual electricity price by bus [€/MWh] ---")
    print(network.buses_t.marginal_price.mean().round(2))

def plot_annual_electricity_mix(network):
    gen_carrier_bus = {}

    for carrier in network.generators.carrier.unique():
        gens = network.generators.index[network.generators.carrier == carrier]
        gen_carrier_bus[carrier] = (
            network.generators_t.p[gens]
            .sum()
            .groupby(network.generators.loc[gens, "bus"])
            .sum()
        )

    gen_df = pd.DataFrame(gen_carrier_bus).fillna(0.0) / 1e6  # TWh

    # --- Convert to percentages ---
    gen_pct = gen_df.div(gen_df.sum(axis=1), axis=0) * 100

    # --- Enforce common order and colours ---
    gen_pct = gen_pct.reindex(columns=TECH_ORDER, fill_value=0.0)
    colors = [TECH_COLORS[c] for c in gen_pct.columns]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    gen_pct.plot(kind="bar", stacked=True, ax=ax, color=colors)

    # --- Absolute energy labels ---
    for i, bus in enumerate(gen_df.index):
        ax.text(
            i,
            102,
            f"{gen_df.loc[bus].sum():.1f} TWh",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Share of annual electricity production [%]")
    ax.set_xlabel("Zone")
    ax.set_ylim(0, 110)
    ax.set_title("Annual electricity mix by zone")
    ax.legend(title="Technology", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "taskD_annual_electricity_mix.png"), dpi=300)
    plt.close(fig)


def plot_nodal_price_duration_curves(network):
    prices = network.buses_t.marginal_price

    fig, ax = plt.subplots(figsize=(10, 6))

    for bus in prices.columns:
        ax.plot(
            prices[bus].sort_values(ascending=False).values,
            label=bus.replace("bus_", "")
        )

    ax.set_xlabel("Hours (sorted)")
    ax.set_ylabel("Electricity price [€/MWh]")
    ax.set_title("Nodal electricity price duration curves")
    ax.legend(title="Zone")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "taskD_nodal_price_duration_curves.png"), dpi=300)
    plt.close(fig)


def plot_line_loading_duration_curves(network):
    loadings = network.lines_t.p0.abs().divide(network.lines.s_nom, axis=1) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    for line in loadings.columns:
        ax.plot(
            loadings[line].sort_values(ascending=False).values,
            label=line
        )

    ax.set_xlabel("Hours (sorted)")
    ax.set_ylabel("Line loading [% of capacity]")
    ax.set_title("Line loading duration curves")
    ax.legend(title="Interconnector")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "taskD_line_loading_duration_curves.png"), dpi=300)
    plt.close(fig)


def plot_net_imports_exports(network):
    # --- Generation by bus (annual) ---
    generation = (
        network.generators_t.p
        .T
        .groupby(network.generators.bus)
        .sum()
        .sum(axis=1)
    )

    # --- Load by bus (annual) ---
    load = network.loads_t.p.sum()

    # Net imports: positive = importer, negative = exporter
    net_imports = (load - generation) / 1e6  # TWh

    fig, ax = plt.subplots(figsize=(8, 5))
    net_imports.plot(kind="bar", ax=ax)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Net imports [TWh]")
    ax.set_xlabel("Zone")
    ax.set_title("Net annual electricity imports (+) / exports (−)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "taskD_net_imports_exports.png"), dpi=300)
    plt.close(fig)

# ============================================================
# Run Task D
# ============================================================

if __name__ == "__main__":
    costs = load_costs(os.path.join(DATA_DIR, "costs_PyPSA.csv"), year=COST_YEAR)

    print("\n--- Mean renewable capacity factors from profiles ---")
    print(mean_capacity_factors().round(3))

    multi_n = build_network(costs)

    print("\n--- Initial lines ---")
    print(multi_n.lines[["bus0", "bus1", "x", "r", "s_nom"]])

    multi_n.optimize(
        extra_functionality=hydro_energy_constraint_no2,
    )

    print_results(multi_n)

    plot_installed_capacity_percent(multi_n)

    for zone in ZONES:
        plot_battery_behavior(multi_n, zone)

    plot_line_flows(multi_n)

    plot_annual_electricity_mix(multi_n)
    plot_nodal_price_duration_curves(multi_n)
    plot_line_loading_duration_curves(multi_n)
    plot_net_imports_exports(multi_n)