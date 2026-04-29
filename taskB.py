import os
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

pd.options.future.infer_string = False

from src.config import YEAR, ZONE, DISCOUNT_RATE
from src.data_loader import load_load
from src.cost_loader import load_costs, get_cost
from src.profiles import make_profile
from src.network_builder import create_network


# ============================================================
# Paths and settings
# ============================================================

DATA_DIR = "Data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
FIG_DIR = "Figures"

os.makedirs(FIG_DIR, exist_ok=True)

WEATHER_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

TARGET_CF = {
    "solar": 0.11,
    "onshore_wind": 0.25,
    "offshore_wind": 0.52,
}


# ============================================================
# Helper functions
# ============================================================

def annuity(rate, lifetime):
    return rate / (1 - (1 + rate) ** (-lifetime))


def annualized_fixed_cost(costs, tech, discount_rate=DISCOUNT_RATE):
    investment = get_cost(costs, tech, "investment") * 1000
    lifetime = get_cost(costs, tech, "lifetime")
    fom = get_cost(costs, tech, "FOM") / 100 * investment

    return investment * annuity(discount_rate, lifetime) + fom


def ocgt_marginal_cost(costs):
    efficiency = get_cost(costs, "OCGT", "efficiency")
    vom = get_cost(costs, "OCGT", "VOM")

    gas_price = 40.0
    co2_price = 80.0
    co2_intensity = 0.202

    return (
        gas_price / efficiency
        + co2_price * co2_intensity / efficiency
        + vom
    )


def read_weather_data(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    df["HourUTC"] = pd.to_datetime(df["HourUTC"], utc=True, errors="coerce")
    df["HourUTC"] = df["HourUTC"].dt.tz_convert(None)

    return (
        df.dropna(subset=["HourUTC"])
        .set_index("HourUTC")
        .sort_index()
    )


def get_generation_series(df_year):
    solar = (
        pd.to_numeric(df_year["SolarPowerLt10kW_MWh"], errors="coerce").fillna(0)
        + pd.to_numeric(df_year["SolarPowerGe10Lt40kW_MWh"], errors="coerce").fillna(0)
        + pd.to_numeric(df_year["SolarPowerGe40kW_MWh"], errors="coerce").fillna(0)
    )

    onshore = (
        pd.to_numeric(df_year["OnshoreWindLt50kW_MWh"], errors="coerce").fillna(0)
        + pd.to_numeric(df_year["OnshoreWindGe50kW_MWh"], errors="coerce").fillna(0)
    )

    offshore = (
        pd.to_numeric(df_year["OffshoreWindLt100MW_MWh"], errors="coerce").fillna(0)
        + pd.to_numeric(df_year["OffshoreWindGe100MW_MWh"], errors="coerce").fillna(0)
    )

    return {
        "solar": solar,
        "onshore_wind": onshore,
        "offshore_wind": offshore,
    }


def prepare_weather_profiles(df_weather, weather_year, snapshots):
    df_year = df_weather[df_weather.index.year == weather_year].copy()

    if df_year.empty:
        raise ValueError(f"No data found for {weather_year}")

    full_index = pd.date_range(
        f"{weather_year}-01-01 00:00",
        f"{weather_year}-12-31 23:00",
        freq="h",
    )

    df_year = df_year.groupby(df_year.index).first()
    df_year = df_year.reindex(full_index)

    generation = get_generation_series(df_year)

    profiles = {}

    for tech, series in generation.items():
        profile = make_profile(series, TARGET_CF[tech])

        # Remove leap day by trimming to 8760 hours
        if len(profile) > len(snapshots):
            profile = profile.iloc[:len(snapshots)]

        profile.index = snapshots
        profiles[tech] = profile.astype("float64")

    return profiles


def build_network(
    snapshots,
    load,
    profiles,
    costs,
):
    n = create_network(snapshots)

    n.add("Carrier", "AC")
    n.add("Bus", ZONE, carrier="AC")

    n.add(
        "Load",
        f"{ZONE}_load",
        bus=ZONE,
        p_set=load,
    )

    solar_capital_cost = annualized_fixed_cost(costs, "solar")
    onshore_capital_cost = annualized_fixed_cost(costs, "onwind")
    offshore_capital_cost = annualized_fixed_cost(costs, "offwind")
    ocgt_capital_cost = annualized_fixed_cost(costs, "OCGT")

    ocgt_efficiency = get_cost(costs, "OCGT", "efficiency")
    ocgt_mc = ocgt_marginal_cost(costs)

    n.add(
        "Generator",
        "solar",
        bus=ZONE,
        carrier="solar",
        p_nom_extendable=True,
        p_max_pu=profiles["solar"],
        capital_cost=solar_capital_cost,
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "onshore_wind",
        bus=ZONE,
        carrier="onwind",
        p_nom_extendable=True,
        p_max_pu=profiles["onshore_wind"],
        capital_cost=onshore_capital_cost,
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "offshore_wind",
        bus=ZONE,
        carrier="offwind",
        p_nom_extendable=True,
        p_max_pu=profiles["offshore_wind"],
        capital_cost=offshore_capital_cost,
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "ocgt",
        bus=ZONE,
        carrier="gas",
        p_nom_extendable=True,
        capital_cost=ocgt_capital_cost,
        marginal_cost=ocgt_mc,
        efficiency=ocgt_efficiency,
    )

    return n


# ============================================================
# Load fixed demand and costs
# ============================================================

snapshots = pd.date_range(
    f"{YEAR}-01-01 00:00",
    f"{YEAR}-12-31 23:00",
    freq="h",
)

load_path = os.path.join(RAW_DIR, "load", f"load_{ZONE}_{YEAR}.csv")
cost_path = os.path.join(DATA_DIR, "costs_PyPSA.csv")
weather_path = os.path.join(DATA_DIR, "ProductionConsumptionSettlement_taskB.csv")

load_data = load_load(load_path)

demand_fixed = (
    load_data
    .set_index("time_utc")["load_MW"]
    .reindex(snapshots)
    .ffill()
    .fillna(0)
    .astype("float64")
)

costs = load_costs(cost_path, year=2030)
df_weather = read_weather_data(weather_path)

print("Weather data range:")
print("Min timestamp:", df_weather.index.min())
print("Max timestamp:", df_weather.index.max())

print("\nRows per year:")
print(df_weather.index.year.value_counts().sort_index())


# ============================================================
# Run sensitivity analysis
# ============================================================

results = []

for weather_year in WEATHER_YEARS:
    print(f"\nRunning optimisation for weather year {weather_year}...")

    try:
        profiles = prepare_weather_profiles(
            df_weather=df_weather,
            weather_year=weather_year,
            snapshots=snapshots,
        )

        n = build_network(
            snapshots=snapshots,
            load=demand_fixed,
            profiles=profiles,
            costs=costs,
        )

        n.optimize()

        capacities = n.generators["p_nom_opt"].copy()
        capacities.name = weather_year
        results.append(capacities)

    except Exception as e:
        print(f"Skipping {weather_year}: {e}")


# ============================================================
# Collect results
# ============================================================

results_df = pd.DataFrame(results)
results_df.index.name = "weather_year"

summary_df = pd.DataFrame({
    "mean_capacity_MW": results_df.mean(),
    "std_capacity_MW": results_df.std(),
    "min_capacity_MW": results_df.min(),
    "max_capacity_MW": results_df.max(),
}).round(2)

print("\nOptimal capacities by weather year [MW]:")
print(results_df.round(2))

print("\nAverage capacity and variability [MW]:")
print(summary_df)


# ============================================================
# Save results
# ============================================================

results_df.to_csv(os.path.join(DATA_DIR, "taskB_capacities_by_weather_year.csv"))
summary_df.to_csv(os.path.join(DATA_DIR, "taskB_capacity_summary.csv"))


# ============================================================
# Plots
# ============================================================

mean_caps = results_df.mean()
std_caps = results_df.std()

plt.figure(figsize=(8, 5))

mean_caps.plot(
    kind="bar",
    yerr=std_caps,
    capsize=4,
)

plt.ylabel("MW")
plt.title("Average optimal capacity and variability across weather years")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_capacity_mean_variability.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


results_df.plot(marker="o", figsize=(10, 5))

plt.ylabel("MW")
plt.title("Optimal capacities by weather year")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_capacities_by_weather_year.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()