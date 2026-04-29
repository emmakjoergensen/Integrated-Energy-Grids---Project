import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

pd.options.future.infer_string = False
pd.options.mode.string_storage = "python"

from src.cost_loader import load_costs, get_cost

from taskD import (
    YEAR,
    COST_YEAR,
    DATA_DIR,
    FIG_DIR,
    ZONES,
    build_network,
    hydro_energy_constraint_no2,
    annualized_fixed_cost,
)


# ============================================================
# Settings
# ============================================================

CH4_PIPELINES = {
    "DK1_NO2": 3000.0,
    "DK1_DE": 5000.0,
    "DK1_DK2": 1500.0,
    "DK2_DE": 3000.0,
}

PIPELINE_EFFICIENCY = 0.99

GAS_PRICE = 40.0
CO2_PRICE = 80.0
GAS_CO2_INTENSITY = 0.202


# ============================================================
# CH4 network functions
# ============================================================

def add_ch4_carriers(network):
    for carrier in ["CH4", "gas_to_power", "CH4_pipeline"]:
        if carrier not in network.carriers.index:
            network.add("Carrier", carrier)


def remove_electric_gas_generators(network):
    gas_generators = network.generators[
        network.generators.carrier == "gas"
    ].index

    for generator in gas_generators:
        network.remove("Generator", generator)


def add_ch4_buses_and_supply(network):
    ch4_supply_cost = GAS_PRICE + CO2_PRICE * GAS_CO2_INTENSITY

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


def add_gas_to_power_links(network, costs):
    ocgt_efficiency = get_cost(costs, "OCGT", "efficiency")
    ocgt_vom = get_cost(costs, "OCGT", "VOM")
    ocgt_capital_cost = annualized_fixed_cost(costs, "OCGT")

    for zone in ZONES:
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


def add_ch4_pipelines(network):
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


def add_ch4_network(network, costs):
    add_ch4_carriers(network)
    remove_electric_gas_generators(network)
    add_ch4_buses_and_supply(network)
    add_gas_to_power_links(network, costs)
    add_ch4_pipelines(network)

    return network


# ============================================================
# Results
# ============================================================

def get_pipeline_names(network):
    return network.links[network.links.carrier == "CH4_pipeline"].index.tolist()


def transport_comparison(network):
    pipeline_names = get_pipeline_names(network)

    electricity_transport = network.lines_t.p0.abs().sum().sum()
    gas_transport = network.links_t.p0[pipeline_names].abs().sum().sum()

    return electricity_transport, gas_transport


def print_transport_comparison(network):
    electricity_transport, gas_transport = transport_comparison(network)

    print("\n--- Transported energy ---")
    print(f"Electricity transported by HVAC lines: {electricity_transport:,.0f} MWh")
    print(f"CH4 transported by gas pipelines: {gas_transport:,.0f} MWh")

    if gas_transport > electricity_transport:
        print("The CH4 pipeline network transports more energy.")
    else:
        print("The electricity transmission network transports more energy.")


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


# ============================================================
# Plots
# ============================================================

def plot_transport_comparison(network):
    electricity_transport, gas_transport = transport_comparison(network)

    comparison = pd.Series({
        "Electricity lines": electricity_transport,
        "CH4 pipelines": gas_transport,
    })

    comparison.plot(kind="bar", figsize=(7, 5))

    plt.ylabel("Annual transported energy [MWh]")
    plt.title("Energy transported by electricity and CH4 networks")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "taskG_transport_comparison.png"), dpi=300)
    plt.show()


def plot_ch4_pipeline_flows(network):
    pipeline_names = get_pipeline_names(network)

    network.links_t.p0[pipeline_names].plot(figsize=(12, 5))

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
        edgelist=[
            ("CH4_DK1", "CH4_NO2"),
            ("CH4_DK1", "CH4_DK2"),
            ("CH4_DK2", "CH4_DE"),
        ],
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

if __name__ == "__main__":
    costs = load_costs(os.path.join(DATA_DIR, "costs_PyPSA.csv"), year=COST_YEAR)

    n_g = build_network(costs)
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

    plot_transport_comparison(n_g)
    plot_ch4_pipeline_flows(n_g)
    plot_network_diagram()