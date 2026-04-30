import os
import pandas as pd
import matplotlib.pyplot as plt

pd.options.future.infer_string = False

from src.cost_loader import load_costs, get_cost

from taskG import (
    DATA_DIR,
    FIG_DIR,
    COST_YEAR,
    build_network,
    add_ch4_network,
    hydro_energy_constraint_no2,
)


# ============================================================
# Settings
# ============================================================

CO2_LIMIT_SHARE = 0.80

GAS_CO2_INTENSITY = 0.202
COAL_CO2_INTENSITY = 0.34
COAL_EFFICIENCY = 0.38

EXISTING_CO2_PRICE = 80.0


# ============================================================
# Build and optimize model
# ============================================================

def build_task_g_model(costs):
    n = build_network(costs)
    n = add_ch4_network(n, costs)
    return n


def optimise_network(n):
    n.optimize(extra_functionality=hydro_energy_constraint_no2)
    return n


# ============================================================
# Emissions
# ============================================================

def calculate_emissions(n):
    emissions_gas = 0.0
    emissions_coal = 0.0

    gas_links = n.links[n.links.carrier == "gas_to_power"].index

    if len(gas_links) > 0:
        gas_input_mwh_th = n.links_t.p0[gas_links].clip(lower=0).sum().sum()
        emissions_gas = gas_input_mwh_th * GAS_CO2_INTENSITY

    coal_gens = n.generators[n.generators.carrier == "coal"].index

    if len(coal_gens) > 0:
        coal_el_mwh = n.generators_t.p[coal_gens].clip(lower=0).sum().sum()
        coal_input_mwh_th = coal_el_mwh / COAL_EFFICIENCY
        emissions_coal = coal_input_mwh_th * COAL_CO2_INTENSITY

    return {
        "gas": emissions_gas,
        "coal": emissions_coal,
        "total": emissions_gas + emissions_coal,
    }


# ============================================================
# CO2 constraint
# ============================================================

def co2_limit_constraint(n, snapshots):
    model = n.model

    gas_links = n.links[n.links.carrier == "gas_to_power"].index
    coal_gens = n.generators[n.generators.carrier == "coal"].index

    emissions = 0

    if len(gas_links) > 0:
        gas_input = model.variables["Link-p"].loc[
            dict(name=gas_links)
        ]
        emissions += (
            gas_input
            * n.snapshot_weightings.generators
            * GAS_CO2_INTENSITY
        ).sum()

    if len(coal_gens) > 0:
        coal_output = model.variables["Generator-p"].loc[
            dict(name=coal_gens)
        ]
        emissions += (
            coal_output
            * n.snapshot_weightings.generators
            / COAL_EFFICIENCY
            * COAL_CO2_INTENSITY
        ).sum()

    model.add_constraints(
        emissions <= n.co2_limit,
        name="CO2Limit",
    )


def optimise_network_with_co2_limit(n, co2_limit):
    n.co2_limit = co2_limit
    n.optimize(extra_functionality=co2_limit_constraint)
    return n


def get_required_co2_price(n):
    mu = n.model.constraints["CO2Limit"].dual.item()

    additional_price = -mu
    total_price = EXISTING_CO2_PRICE + additional_price

    return mu, additional_price, total_price


# ============================================================
# Comparisons
# ============================================================

def compare_capacities(baseline, constrained):
    gen_comparison = pd.DataFrame({
        "Baseline [MW]": baseline.generators["p_nom_opt"],
        "CO2 constrained [MW]": constrained.generators["p_nom_opt"],
    }).round(2)

    link_comparison = pd.DataFrame({
        "Baseline [MW]": baseline.links["p_nom_opt"],
        "CO2 constrained [MW]": constrained.links["p_nom_opt"],
    }).round(2)

    return gen_comparison, link_comparison


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

if __name__ == "__main__":

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
    constrained = optimise_network_with_co2_limit(constrained, co2_limit)

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

    summary.to_csv(
        os.path.join(DATA_DIR, "taskH_co2_price_summary.csv"),
        index=False,
    )

    generator_comparison.to_csv(
        os.path.join(DATA_DIR, "taskH_generator_capacity_comparison.csv"),
    )

    link_comparison.to_csv(
        os.path.join(DATA_DIR, "taskH_link_capacity_comparison.csv"),
    )

    # ----------------------------
    # Plot
    # ----------------------------

    plot_emissions(
        baseline_emissions,
        constrained_emissions,
        co2_limit,
    )