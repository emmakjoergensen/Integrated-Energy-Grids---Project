import os
import pandas as pd
import matplotlib.pyplot as plt

pd.options.future.infer_string = False

from src.cost_loader import load_costs, get_cost

from taskG import (
    DATA_DIR,
    FIG_DIR,
    COST_YEAR,
    ZONES,
    build_base_electricity_network,
    add_ch4_network,
    hydro_energy_constraint_no2,
    coal_costs,
)


# ============================================================
# Settings
# ============================================================

CO2_LIMIT_SHARE = 0.80   # 80% of baseline emissions

GAS_CO2_INTENSITY = 0.202   # tCO2/MWh_th
COAL_CO2_INTENSITY = 0.34   # tCO2/MWh_th
COAL_EFFICIENCY = 0.38

EXISTING_CO2_PRICE = 80.0   # €/tCO2 already included in fuel assumptions


# ============================================================
# Helper functions
# ============================================================

def build_task_g_model(costs):
    network = build_base_electricity_network(costs)
    network = add_ch4_network(network, costs)
    return network


def optimise_network(network):
    network.optimize(extra_functionality=hydro_energy_constraint_no2)
    return network


def calculate_emissions(network):
    emissions_gas = 0.0
    emissions_coal = 0.0

    gas_links = network.links[network.links.carrier == "gas_to_power"].index

    if len(gas_links) > 0:
        ch4_input = network.links_t.p0[gas_links].sum().sum()
        emissions_gas = ch4_input * GAS_CO2_INTENSITY

    coal_generators = network.generators[network.generators.carrier == "coal"].index

    if len(coal_generators) > 0:
        coal_generation = network.generators_t.p[coal_generators].sum().sum()
        emissions_coal = (
            coal_generation / COAL_EFFICIENCY * COAL_CO2_INTENSITY
        )

    return {
        "gas": emissions_gas,
        "coal": emissions_coal,
        "total": emissions_gas + emissions_coal,
    }


def add_co2_limit(network, co2_limit):
    network.carriers.loc["CH4", "co2_emissions"] = GAS_CO2_INTENSITY
    network.carriers.loc["coal", "co2_emissions"] = COAL_CO2_INTENSITY / COAL_EFFICIENCY

    for carrier in [
        "AC",
        "onwind",
        "offwind",
        "solar",
        "hydro",
        "battery",
        "gas_to_power",
        "CH4_pipeline",
    ]:
        if carrier in network.carriers.index:
            network.carriers.loc[carrier, "co2_emissions"] = 0.0

    network.add(
        "GlobalConstraint",
        "CO2Limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limit,
    )

    return network


def get_required_co2_price(network):
    mu = network.global_constraints.loc["CO2Limit", "mu"]

    additional_price = -mu
    total_price = EXISTING_CO2_PRICE + additional_price

    return mu, additional_price, total_price


def compare_capacities(baseline, constrained):
    generator_comparison = pd.DataFrame({
        "Baseline [MW]": baseline.generators["p_nom_opt"],
        "CO2 constrained [MW]": constrained.generators["p_nom_opt"],
    }).round(2)

    link_comparison = pd.DataFrame({
        "Baseline [MW]": baseline.links["p_nom_opt"],
        "CO2 constrained [MW]": constrained.links["p_nom_opt"],
    }).round(2)

    return generator_comparison, link_comparison


def plot_emissions(baseline_emissions, constrained_emissions, co2_limit):
    emissions_plot = pd.Series({
        "Baseline": baseline_emissions["total"],
        "CO2 constrained": constrained_emissions["total"],
        "CO2 limit": co2_limit,
    })

    emissions_plot.plot(kind="bar", figsize=(7, 5))

    plt.ylabel("Annual CO2 emissions [tCO2]")
    plt.title("CO2 emissions before and after CO2 constraint")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskH_emissions_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# ============================================================
# Run Task H
# ============================================================

costs = load_costs(
    os.path.join(DATA_DIR, "costs_PyPSA.csv"),
    year=COST_YEAR,
)

# ----------------------------
# Baseline Task G model
# ----------------------------

baseline = build_task_g_model(costs)
baseline = optimise_network(baseline)

baseline_emissions = calculate_emissions(baseline)
co2_limit = CO2_LIMIT_SHARE * baseline_emissions["total"]

print("\n--- Baseline CO2 emissions ---")
print(f"Gas emissions:   {baseline_emissions['gas']:,.0f} tCO2")
print(f"Coal emissions:  {baseline_emissions['coal']:,.0f} tCO2")
print(f"Total emissions: {baseline_emissions['total']:,.0f} tCO2")

print("\n--- Selected CO2 allowance ---")
print(f"CO2 limit: {co2_limit:,.0f} tCO2")
print(f"CO2 limit share of baseline: {CO2_LIMIT_SHARE:.0%}")


# ----------------------------
# CO2 constrained model
# ----------------------------

constrained = build_task_g_model(costs)
constrained = add_co2_limit(constrained, co2_limit)
constrained = optimise_network(constrained)

constrained_emissions = calculate_emissions(constrained)

print("\n--- CO2 constrained emissions ---")
print(f"Gas emissions:   {constrained_emissions['gas']:,.0f} tCO2")
print(f"Coal emissions:  {constrained_emissions['coal']:,.0f} tCO2")
print(f"Total emissions: {constrained_emissions['total']:,.0f} tCO2")


# ----------------------------
# CO2 shadow price
# ----------------------------

mu, additional_co2_price, total_co2_price = get_required_co2_price(constrained)

print("\n--- CO2 price result ---")
print(f"Raw shadow price: {mu:.2f} €/tCO2")
print(f"Additional CO2 price required by cap: {additional_co2_price:.2f} €/tCO2")
print(f"Existing assumed CO2 price: {EXISTING_CO2_PRICE:.2f} €/tCO2")
print(f"Total effective CO2 price: {total_co2_price:.2f} €/tCO2")


# ----------------------------
# Capacity comparison
# ----------------------------

generator_comparison, link_comparison = compare_capacities(
    baseline,
    constrained,
)

print("\n--- Generator capacity comparison ---")
print(generator_comparison)

print("\n--- Link capacity comparison ---")
print(link_comparison)


# ----------------------------
# Save outputs
# ----------------------------

summary = pd.DataFrame({
    "baseline_emissions_tCO2": [baseline_emissions["total"]],
    "co2_limit_tCO2": [co2_limit],
    "constrained_emissions_tCO2": [constrained_emissions["total"]],
    "raw_shadow_price_EUR_per_tCO2": [mu],
    "additional_CO2_price_EUR_per_tCO2": [additional_co2_price],
    "existing_assumed_CO2_price_EUR_per_tCO2": [EXISTING_CO2_PRICE],
    "total_effective_CO2_price_EUR_per_tCO2": [total_co2_price],
})

summary.to_csv(os.path.join(DATA_DIR, "taskH_co2_price_summary.csv"), index=False)
generator_comparison.to_csv(os.path.join(DATA_DIR, "taskH_generator_capacity_comparison.csv"))
link_comparison.to_csv(os.path.join(DATA_DIR, "taskH_link_capacity_comparison.csv"))


# ----------------------------
# Plot
# ----------------------------

plot_emissions(
    baseline_emissions,
    constrained_emissions,
    co2_limit,
)