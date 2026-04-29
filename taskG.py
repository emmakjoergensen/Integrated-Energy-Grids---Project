import os
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import networkx as nx

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

ELECTRICITY_LINES = {
    "DK1_NO2": {"bus0": "bus_DK1", "bus1": "bus_NO2", "s_nom": 1700.0},
    "DK1_DE": {"bus0": "bus_DK1", "bus1": "bus_DE", "s_nom": 2500.0},
    "DK1_DK2": {"bus0": "bus_DK1", "bus1": "bus_DK2", "s_nom": 600.0},
    "DK2_DE": {"bus0": "bus_DK2", "bus1": "bus_DE", "s_nom": 600.0},
}

CH4_PIPELINES = {
    "DK1_NO2": 3000.0,
    "DK1_DE": 5000.0,
    "DK1_DK2": 1500.0,
    "DK2_DE": 3000.0,
}

PIPELINE_EFFICIENCY = 0.99


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

    return inverter_cost + storage_cost * BATTERY["duration"]


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


# ============================================================
# Hydro constraint
# ============================================================

def get_hydro_energy_no2():
    gen_no2 = pivot_generation("NO2")
    return gen_no2["hydro_res"].sum()


def hydro_energy_constraint_no2(network, snapshots):
    if "hydro_res_NO2" not in network.generators.index:
        return

    model = network.model
    hydro_energy_no2 = get_hydro_energy_no2()

    p = model.variables["Generator-p"].loc[dict(name="hydro_res_NO2")]
    weights = network.snapshot_weightings.generators

    model.add_constraints(
        (p * weights).sum(dim="snapshot") <= hydro_energy_no2,
        name="hydro_energy_limit_NO2",
    )


# ============================================================
# Build Task D electricity network
# ============================================================

def add_electricity_zone(network, zone, costs):
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
        existing_capacity = gen["coal"].max()

        network.add(
            "Generator",
            "coal_DE",
            bus=bus,
            carrier="coal",
            p_nom_extendable=True,
            p_nom=existing_capacity,
            p_nom_min=existing_capacity,
            capital_cost=coal_capital_cost,
            marginal_cost=coal_marginal_cost,
        )

    if zone == "NO2" and "hydro_res" in gen.columns and gen["hydro_res"].sum() > 0:
        existing_capacity = gen["hydro_res"].max()

        network.add(
            "Generator",
            "hydro_res_NO2",
            bus=bus,
            carrier="hydro",
            p_nom_extendable=False,
            p_nom=existing_capacity,
            capital_cost=annualized_fixed_cost(costs, "hydro"),
            marginal_cost=0.0,
        )


def add_electricity_lines(network):
    for name, line in ELECTRICITY_LINES.items():
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


def build_base_electricity_network(costs):
    snapshots = pd.date_range(
        f"{YEAR}-01-01 00:00",
        f"{YEAR}-12-31 23:00",
        freq="h",
    )

    network = create_network(snapshots)

    for carrier in [
        "AC", "onwind", "offwind", "solar",
        "coal", "hydro", "battery",
        "CH4", "gas_to_power", "CH4_pipeline",
    ]:
        network.add("Carrier", carrier)

    for zone in ZONES:
        add_electricity_zone(network, zone, costs)

    network.buses.loc["bus_DK1", "slack"] = True

    add_electricity_lines(network)

    return network


# ============================================================
# Add CH4 network for Task G
# ============================================================

def add_ch4_network(network, costs):
    gas_price = 40.0
    co2_price = 80.0
    co2_intensity = 0.202

    ocgt_efficiency = get_cost(costs, "OCGT", "efficiency")
    ocgt_vom = get_cost(costs, "OCGT", "VOM")
    ocgt_capital_cost = annualized_fixed_cost(costs, "OCGT")

    ch4_supply_cost = gas_price + co2_price * co2_intensity

    for zone in ZONES:
        network.add(
            "Bus",
            f"CH4_bus_{zone}",
            carrier="CH4",
        )

        network.add(
            "Generator",
            f"CH4_supply_{zone}",
            bus=f"CH4_bus_{zone}",
            carrier="CH4",
            p_nom_extendable=True,
            marginal_cost=ch4_supply_cost,
            capital_cost=0.0,
        )

        network.add(
            "Link",
            f"CH4_to_power_{zone}",
            bus0=f"CH4_bus_{zone}",
            bus1=f"bus_{zone}",
            carrier="gas_to_power",
            p_nom_extendable=True,
            efficiency=ocgt_efficiency,
            capital_cost=ocgt_capital_cost,
            marginal_cost=ocgt_vom,
        )

    for pipe_name, capacity in CH4_PIPELINES.items():
        zone0, zone1 = pipe_name.split("_")

        network.add(
            "Link",
            f"CH4_pipeline_{pipe_name}",
            bus0=f"CH4_bus_{zone0}",
            bus1=f"CH4_bus_{zone1}",
            carrier="CH4_pipeline",
            p_nom_extendable=False,
            p_nom=capacity,
            p_min_pu=-1.0,
            efficiency=PIPELINE_EFFICIENCY,
            marginal_cost=0.0,
        )

    return network


# ============================================================
# Results and plots
# ============================================================

def get_pipeline_names(network):
    return network.links[network.links.carrier == "CH4_pipeline"].index.tolist()


def print_transport_comparison(network):
    pipeline_names = get_pipeline_names(network)

    electricity_transport = network.lines_t.p0.abs().sum().sum()
    gas_transport = network.links_t.p0[pipeline_names].abs().sum().sum()

    print("\n--- Transported energy ---")
    print(f"Electricity transported by HVAC lines: {electricity_transport:,.0f} MWh")
    print(f"CH4 transported by gas pipelines: {gas_transport:,.0f} MWh")

    if gas_transport > electricity_transport:
        print("The CH4 pipeline network transports more energy.")
    else:
        print("The electricity transmission network transports more energy.")

    comparison = pd.Series({
        "Electricity lines": electricity_transport,
        "CH4 pipelines": gas_transport,
    })

    ax = comparison.plot(kind="bar", figsize=(7, 5))
    plt.ylabel("Annual transported energy [MWh]")
    plt.title("Energy transported by electricity and CH4 networks")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "taskG_transport_comparison.png"), dpi=300)
    plt.show()


def print_gas_results(network):
    pipeline_names = get_pipeline_names(network)

    print("\n--- Gas-to-power capacities ---")
    print(
        network.links[network.links.carrier == "gas_to_power"][
            ["bus0", "bus1", "p_nom_opt", "efficiency"]
        ].round(2)
    )

    print("\n--- Pipeline capacities ---")
    print(
        network.links.loc[pipeline_names, ["bus0", "bus1", "p_nom", "efficiency"]]
    )

    pipeline_loading = (
        network.links_t.p0[pipeline_names].abs().max()
        / network.links.loc[pipeline_names, "p_nom"]
        * 100
    )

    print("\n--- Max pipeline loading [%] ---")
    print(pipeline_loading.round(2))


def plot_ch4_pipeline_flows(network):
    pipeline_names = get_pipeline_names(network)

    ax = network.links_t.p0[pipeline_names].plot(figsize=(12, 5))

    plt.axhline(0, linewidth=1)
    plt.ylabel("MW_th")
    plt.title("CH4 pipeline flows")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "taskG_ch4_pipeline_flows.png"), dpi=300)
    plt.show()


def plot_network_diagram():
    graph = nx.Graph()

    electricity_nodes = ["NO2", "DK1", "DK2", "DE"]
    gas_nodes = [f"CH4_{zone}" for zone in electricity_nodes]

    electricity_edges = [
        ("DK1", "NO2"),
        ("DK1", "DE"),
        ("DK1", "DK2"),
        ("DK2", "DE"),
    ]

    gas_edges = [
        ("CH4_DK1", "CH4_NO2"),
        ("CH4_DK1", "CH4_DE"),
        ("CH4_DK1", "CH4_DK2"),
        ("CH4_DK2", "CH4_DE"),
    ]

    conversion_edges = [
        ("CH4_NO2", "NO2"),
        ("CH4_DK1", "DK1"),
        ("CH4_DK2", "DK2"),
        ("CH4_DE", "DE"),
    ]

    graph.add_nodes_from(electricity_nodes + gas_nodes)
    graph.add_edges_from(electricity_edges + gas_edges + conversion_edges)

    positions = {
        "NO2": (0, 2),
        "DK1": (0, 0),
        "DK2": (3, 0),
        "DE": (3, -2),
        "CH4_NO2": (0.35, 2.35),
        "CH4_DK1": (0.35, 0.35),
        "CH4_DK2": (3.35, 0.35),
        "CH4_DE": (3.35, -1.65),
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=electricity_nodes,
        node_color="#bcdffb",
        edgecolors="black",
        node_size=1700,
        linewidths=1.3,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=gas_nodes,
        node_color="#c8efc1",
        edgecolors="black",
        node_shape="s",
        node_size=1500,
        linewidths=1.3,
        ax=ax,
    )

    labels = {
        "NO2": "NO2",
        "DK1": "DK1",
        "DK2": "DK2",
        "DE": "DE",
        "CH4_NO2": "CH4\nNO2",
        "CH4_DK1": "CH4\nDK1",
        "CH4_DK2": "CH4\nDK2",
        "CH4_DE": "CH4\nDE",
    }

    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=10, ax=ax)

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[("DK1", "NO2"), ("DK1", "DK2"), ("DK2", "DE")],
        width=2.4,
        edge_color="black",
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[("DK1", "DE")],
        width=2.4,
        edge_color="black",
        connectionstyle="arc3,rad=-0.25",
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[("CH4_DK1", "CH4_NO2"), ("CH4_DK1", "CH4_DK2"), ("CH4_DK2", "CH4_DE")],
        width=2.4,
        edge_color="green",
        style="dashed",
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[("CH4_DK1", "CH4_DE")],
        width=2.4,
        edge_color="green",
        style="dashed",
        connectionstyle="arc3,rad=-0.25",
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=conversion_edges,
        width=1.8,
        edge_color="red",
        style="dotted",
        ax=ax,
    )

    ax.plot([], [], color="black", linewidth=2.4, label="HVAC line")
    ax.plot([], [], color="green", linestyle="--", linewidth=2.4, label="CH4 pipeline")
    ax.plot([], [], color="red", linestyle=":", linewidth=2.4, label="Gas-to-power")

    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.set_title("Integrated electricity and CH4 network")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "taskG_network_diagram.png"), dpi=300)
    plt.show()


# ============================================================
# Run Task G
# ============================================================

costs = load_costs(os.path.join(DATA_DIR, "costs_PyPSA.csv"), year=COST_YEAR)

n_g = build_base_electricity_network(costs)
n_g = add_ch4_network(n_g, costs)

print("\n--- Task G network overview ---")
print("Buses:", len(n_g.buses))
print("Generators:", len(n_g.generators))
print("Links:", len(n_g.links))
print("Lines:", len(n_g.lines))
print("Storage units:", len(n_g.storage_units))

n_g.optimize(
    extra_functionality=hydro_energy_constraint_no2,
)

print_transport_comparison(n_g)
print_gas_results(n_g)
plot_ch4_pipeline_flows(n_g)
plot_network_diagram()