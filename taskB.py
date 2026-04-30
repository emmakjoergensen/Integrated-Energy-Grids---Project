import os
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

pd.options.future.infer_string = False

from src.config import YEAR, ZONE, DISCOUNT_RATE, PSR_MAP
from src.data_loader import load_generation, load_load
from src.cost_loader import load_costs, get_cost
from src.profiles import make_profile
from src.network_builder import create_network


# ============================================================
# Settings
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

    return gas_price / efficiency + co2_price * co2_intensity / efficiency + vom


def build_generation_series(generation, psr_codes, year, full_index):
    df_year = generation[generation["time_utc"].dt.year == year].copy()

    if df_year.empty:
        raise ValueError(f"No generation data found for {year}")

    series = (
        df_year[df_year["psr_type"].isin(psr_codes)]
        .groupby("time_utc")["generation_MW"]
        .sum()
        .reindex(full_index)
        .fillna(0)
    )

    if series.max() <= 0:
        raise ValueError(f"No positive generation values found for {year}")

    return series


def make_weather_profiles(year, snapshots, wind_generation, solar_generation):
    full_index = pd.date_range(
        f"{year}-01-01 00:00",
        f"{year}-12-31 23:00",
        freq="h",
    )

    solar_gen = build_generation_series(
        solar_generation,
        PSR_MAP["solar"],
        year,
        full_index,
    )

    onshore_gen = build_generation_series(
        wind_generation,
        PSR_MAP["onshore_wind"],
        year,
        full_index,
    )

    offshore_gen = build_generation_series(
        wind_generation,
        PSR_MAP["offshore_wind"],
        year,
        full_index,
    )

    solar_cf = make_profile(solar_gen, TARGET_CF["solar"])
    onshore_cf = make_profile(onshore_gen, TARGET_CF["onshore_wind"])
    offshore_cf = make_profile(offshore_gen, TARGET_CF["offshore_wind"])

    if len(full_index) == 8784:
        solar_cf = solar_cf.iloc[:8760]
        onshore_cf = onshore_cf.iloc[:8760]
        offshore_cf = offshore_cf.iloc[:8760]

    solar_cf.index = snapshots
    onshore_cf.index = snapshots
    offshore_cf.index = snapshots

    return {
        "solar": solar_cf.astype("float64"),
        "onshore_wind": onshore_cf.astype("float64"),
        "offshore_wind": offshore_cf.astype("float64"),
    }


def build_network(snapshots, load, profiles, costs):
    n = create_network(snapshots)

    n.add("Carrier", "AC")
    n.add("Bus", ZONE, carrier="AC")

    n.add("Load", f"{ZONE}_load", bus=ZONE, p_set=load)

    n.add(
        "Generator",
        "solar",
        bus=ZONE,
        p_nom_extendable=True,
        p_max_pu=profiles["solar"],
        capital_cost=annualized_fixed_cost(costs, "solar"),
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "onshore_wind",
        bus=ZONE,
        p_nom_extendable=True,
        p_max_pu=profiles["onshore_wind"],
        capital_cost=annualized_fixed_cost(costs, "onwind"),
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "offshore_wind",
        bus=ZONE,
        p_nom_extendable=True,
        p_max_pu=profiles["offshore_wind"],
        capital_cost=annualized_fixed_cost(costs, "offwind"),
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "ocgt",
        bus=ZONE,
        p_nom_extendable=True,
        capital_cost=annualized_fixed_cost(costs, "OCGT"),
        marginal_cost=ocgt_marginal_cost(costs),
        efficiency=get_cost(costs, "OCGT", "efficiency"),
    )

    return n


# ============================================================
# Load data
# ============================================================

snapshots = pd.date_range(
    f"{YEAR}-01-01 00:00",
    f"{YEAR}-12-31 23:00",
    freq="h",
)

load_path = os.path.join(RAW_DIR, "load", f"load_{ZONE}_2020_2025.csv")
wind_path = os.path.join(RAW_DIR, "generation", f"gen_{ZONE}_2020_2025.csv")
solar_path = os.path.join(RAW_DIR, "generation", f"solar_{ZONE}_2020_2025.csv")
cost_path = os.path.join(DATA_DIR, "costs_PyPSA.csv")

load_data = load_load(load_path)
wind_generation = load_generation(wind_path)
solar_generation = load_generation(solar_path)
costs = load_costs(cost_path, year=2030)

load_2025 = load_data[load_data["time_utc"].dt.year == YEAR].copy()

demand_fixed = (
    load_2025
    .set_index("time_utc")["load_MW"]
    .groupby(level=0)
    .first()
    .reindex(snapshots)
    .ffill()
    .fillna(0)
    .astype("float64")
)


# ============================================================
# Run optimisation and store profiles
# ============================================================

results = []
profiles_by_year = {}

for year in WEATHER_YEARS:
    print(f"\nRunning optimisation for weather year {year}...")

    try:
        profiles = make_weather_profiles(
            year=year,
            snapshots=snapshots,
            wind_generation=wind_generation,
            solar_generation=solar_generation,
        )

        profiles_by_year[year] = profiles

        print("Solar CF:", profiles["solar"].mean())
        print("Onshore CF:", profiles["onshore_wind"].mean())
        print("Offshore CF:", profiles["offshore_wind"].mean())

        n = build_network(
            snapshots=snapshots,
            load=demand_fixed,
            profiles=profiles,
            costs=costs,
        )

        n.loads_t.p_set = n.loads_t.p_set.astype("float64")
        n.generators_t.p_max_pu = n.generators_t.p_max_pu.astype("float64")

        for col in ["capital_cost", "marginal_cost", "efficiency"]:
            if col in n.generators.columns:
                n.generators[col] = pd.to_numeric(
                    n.generators[col],
                    errors="coerce",
                ).astype("float64")

        n.optimize()

        capacities = n.generators["p_nom_opt"].copy()
        capacities.name = year
        results.append(capacities)

    except Exception as e:
        print(f"Skipping {year}: {e}")


# ============================================================
# Results
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

results_df.to_csv(os.path.join(DATA_DIR, "taskB_capacities_by_weather_year.csv"))
summary_df.to_csv(os.path.join(DATA_DIR, "taskB_capacity_summary.csv"))


# ============================================================
# Diagnostic plots: why is 2025 onshore attractive?
# ============================================================

# 1) Weekly average onshore vs offshore for each weather year
for year, profiles in profiles_by_year.items():
    plt.figure(figsize=(12, 4))

    profiles["onshore_wind"].rolling(24 * 7).mean().plot(label="onshore wind")
    profiles["offshore_wind"].rolling(24 * 7).mean().plot(label="offshore wind")

    plt.title(f"Weekly average wind availability - weather year {year}")
    plt.ylabel("p_max_pu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"taskB_weekly_wind_availability_{year}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# 2) Difference between onshore and offshore availability
plt.figure(figsize=(12, 5))

for year, profiles in profiles_by_year.items():
    diff = (
        profiles["onshore_wind"].rolling(24 * 7).mean()
        - profiles["offshore_wind"].rolling(24 * 7).mean()
    )
    diff.plot(label=str(year))

plt.axhline(0, linewidth=1)
plt.title("Weekly average difference: onshore wind minus offshore wind")
plt.ylabel("p_max_pu difference")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_onshore_minus_offshore_weekly.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# 3) Wind availability during the highest-load hours
top_load_hours = demand_fixed.sort_values(ascending=False).head(500).index

critical_hour_table = []

for year, profiles in profiles_by_year.items():
    critical_hour_table.append({
        "weather_year": year,
        "onshore_mean_top_500_load_hours": profiles["onshore_wind"].loc[top_load_hours].mean(),
        "offshore_mean_top_500_load_hours": profiles["offshore_wind"].loc[top_load_hours].mean(),
        "solar_mean_top_500_load_hours": profiles["solar"].loc[top_load_hours].mean(),
    })

critical_hour_df = pd.DataFrame(critical_hour_table).set_index("weather_year")

print("\nAverage availability in top 500 demand hours:")
print(critical_hour_df.round(3))

critical_hour_df.plot(kind="bar", figsize=(10, 5))
plt.ylabel("Mean p_max_pu")
plt.title("Average availability during top 500 demand hours")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_availability_top_500_load_hours.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# 4) Correlation between fixed load and wind availability
correlation_table = []

for year, profiles in profiles_by_year.items():
    correlation_table.append({
        "weather_year": year,
        "corr_load_onshore": demand_fixed.corr(profiles["onshore_wind"]),
        "corr_load_offshore": demand_fixed.corr(profiles["offshore_wind"]),
        "corr_load_solar": demand_fixed.corr(profiles["solar"]),
    })

correlation_df = pd.DataFrame(correlation_table).set_index("weather_year")

print("\nCorrelation with fixed 2025 load:")
print(correlation_df.round(3))

correlation_df.plot(kind="bar", figsize=(10, 5))
plt.ylabel("Correlation")
plt.title("Correlation between availability profiles and fixed 2025 load")
plt.axhline(0, linewidth=1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_profile_load_correlation.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# 5) Duration curves: onshore vs offshore, especially 2025
plt.figure(figsize=(10, 5))

for year, profiles in profiles_by_year.items():
    profiles["onshore_wind"].sort_values(ascending=False).reset_index(drop=True).plot(
        label=f"onshore {year}"
    )

plt.title("Onshore wind duration curves by weather year")
plt.xlabel("Hour rank")
plt.ylabel("p_max_pu")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_onshore_duration_curves.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


plt.figure(figsize=(10, 5))

for year, profiles in profiles_by_year.items():
    profiles["offshore_wind"].sort_values(ascending=False).reset_index(drop=True).plot(
        label=f"offshore {year}"
    )

plt.title("Offshore wind duration curves by weather year")
plt.xlabel("Hour rank")
plt.ylabel("p_max_pu")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskB_offshore_duration_curves.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# 6) Final Task B capacity plots
mean_caps = results_df.mean()
std_caps = results_df.std()

plt.figure(figsize=(8, 5))
mean_caps.plot(
    kind="bar",
    yerr=std_caps,
    capsize=4,
)
plt.ylabel("MW")
plt.title("Average capacity and variability across weather years (2020-2025)")
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