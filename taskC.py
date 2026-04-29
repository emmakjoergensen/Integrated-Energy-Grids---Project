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
PLOT_YEAR = 2025

BATTERY_NAME = "battery storage"

TARGET_CF = {
    "solar": 0.11,
    "onshore_wind": 0.25,
    "offshore_wind": 0.52,
}

BATTERY = {
    "duration": 4,
    "efficiency_store": 0.95,
    "efficiency_dispatch": 0.95,
    "marginal_cost": 0.0,
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

        if len(profile) > len(snapshots):
            profile = profile.iloc[:len(snapshots)]

        profile.index = snapshots
        profiles[tech] = profile.astype("float64")

    return profiles


def annualized_cost_no_fom(costs, tech, discount_rate=DISCOUNT_RATE):
    investment = get_cost(costs, tech, "investment") * 1000
    lifetime = get_cost(costs, tech, "lifetime")

    return investment * annuity(discount_rate, lifetime)


def get_battery_cost(costs):
    inverter_cost = annualized_fixed_cost(costs, "battery inverter")

    storage_energy_cost = annualized_cost_no_fom(
        costs,
        "battery storage"
    )

    total_battery_cost = (
        inverter_cost
        + storage_energy_cost * BATTERY["duration"]
    )

    return total_battery_cost


def build_network_with_battery(snapshots, load, profiles, costs):
    n = create_network(snapshots)

    for carrier in ["AC", "solar", "onwind", "offwind", "gas", BATTERY_NAME]:
        n.add("Carrier", carrier)

    n.add("Bus", ZONE, carrier="AC")

    n.add(
        "Load",
        f"{ZONE}_load",
        bus=ZONE,
        p_set=load,
    )

    n.add(
        "Generator",
        "solar",
        bus=ZONE,
        carrier="solar",
        p_nom_extendable=True,
        p_max_pu=profiles["solar"],
        capital_cost=annualized_fixed_cost(costs, "solar"),
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "onshore_wind",
        bus=ZONE,
        carrier="onwind",
        p_nom_extendable=True,
        p_max_pu=profiles["onshore_wind"],
        capital_cost=annualized_fixed_cost(costs, "onwind"),
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "offshore_wind",
        bus=ZONE,
        carrier="offwind",
        p_nom_extendable=True,
        p_max_pu=profiles["offshore_wind"],
        capital_cost=annualized_fixed_cost(costs, "offwind"),
        marginal_cost=0,
    )

    n.add(
        "Generator",
        "ocgt",
        bus=ZONE,
        carrier="gas",
        p_nom_extendable=True,
        capital_cost=annualized_fixed_cost(costs, "OCGT"),
        marginal_cost=ocgt_marginal_cost(costs),
        efficiency=get_cost(costs, "OCGT", "efficiency"),
    )

    n.add(
        "StorageUnit",
        BATTERY_NAME,
        bus=ZONE,
        carrier=BATTERY_NAME,
        p_nom_extendable=True,
        max_hours=BATTERY["duration"],
        efficiency_store=BATTERY["efficiency_store"],
        efficiency_dispatch=BATTERY["efficiency_dispatch"],
        capital_cost=get_battery_cost(costs),
        marginal_cost=BATTERY["marginal_cost"],
        cyclic_state_of_charge=True,
    )

    return n


def plot_dispatch_with_battery(n, period, title, filename):
    dispatch = n.generators_t.p[["solar", "onshore_wind", "offshore_wind", "ocgt"]].copy()
    load = n.loads_t.p[f"{ZONE}_load"].copy()

    battery_p = n.storage_units_t.p[BATTERY_NAME].copy()
    battery_discharge = battery_p.clip(lower=0)
    battery_charge = -battery_p.clip(upper=0)

    dispatch["battery_discharge"] = battery_discharge
    effective_load = load + battery_charge

    ax = dispatch.loc[period].plot.area(figsize=(12, 5))

    effective_load.loc[period].plot(
        ax=ax,
        color="black",
        linewidth=2,
        label="load + battery charging",
    )

    load.loc[period].plot(
        ax=ax,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label="load",
    )

    plt.title(title)
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()


def plot_battery_behavior(n, period, title, filename):
    battery_p = n.storage_units_t.p[BATTERY_NAME].copy()
    battery_soc = n.storage_units_t.state_of_charge[BATTERY_NAME].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))

    battery_p.loc[period].plot(ax=ax1, label="battery dispatch [MW]")
    ax1.axhline(0, linewidth=1)
    ax1.set_ylabel("MW")

    ax2 = ax1.twinx()
    battery_soc.loc[period].plot(
        ax=ax2,
        linestyle="--",
        label="state of charge [MWh]",
    )
    ax2.set_ylabel("MWh")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Load fixed demand, weather data and costs
# ============================================================

snapshots = pd.date_range(
    f"{YEAR}-01-01 00:00",
    f"{YEAR}-12-31 23:00",
    freq="h",
)

load_path = os.path.join(RAW_DIR, "load", f"load_{ZONE}_{YEAR}.csv")
cost_path = os.path.join(DATA_DIR, "costs_PyPSA.csv")
weather_path = os.path.join(DATA_DIR, "ProductionConsumptionSettlement_taskC.csv")

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
# Run Task C sensitivity with battery
# ============================================================

results = []
plot_network = None

for weather_year in WEATHER_YEARS:
    print(f"\nRunning optimisation for weather year {weather_year}...")

    try:
        profiles = prepare_weather_profiles(
            df_weather=df_weather,
            weather_year=weather_year,
            snapshots=snapshots,
        )

        n = build_network_with_battery(
            snapshots=snapshots,
            load=demand_fixed,
            profiles=profiles,
            costs=costs,
        )

        n.optimize()

        generator_capacities = n.generators["p_nom_opt"].copy()
        storage_capacities = n.storage_units["p_nom_opt"].copy()

        capacities = pd.concat([generator_capacities, storage_capacities])
        capacities.name = weather_year

        results.append(capacities)

        if weather_year == PLOT_YEAR:
            plot_network = n

    except Exception as e:
        print(f"Skipping {weather_year}: {e}")


# ============================================================
# Results tables
# ============================================================

results_df = pd.DataFrame(results)
results_df.index.name = "weather_year"

if results_df.empty:
    raise RuntimeError("No optimisation results were created. Check input data and cost assumptions.")

summary_df = pd.DataFrame({
    "mean_capacity_MW": results_df.mean(),
    "std_capacity_MW": results_df.std(),
    "min_capacity_MW": results_df.min(),
    "max_capacity_MW": results_df.max(),
}).round(2)

print("\nOptimal capacities by weather year [MW]:")
print(results_df.round(2))

print("\nAverage capacity and variability with battery [MW]:")
print(summary_df)

results_df.to_csv(os.path.join(DATA_DIR, "taskC_capacities_by_weather_year.csv"))
summary_df.to_csv(os.path.join(DATA_DIR, "taskC_capacity_summary.csv"))


# ============================================================
# Battery-specific results for plot year
# ============================================================

if plot_network is not None:
    battery_power = plot_network.storage_units.at[BATTERY_NAME, "p_nom_opt"]
    battery_energy = battery_power * plot_network.storage_units.at[BATTERY_NAME, "max_hours"]

    battery_p = plot_network.storage_units_t.p[BATTERY_NAME]
    battery_soc = plot_network.storage_units_t.state_of_charge[BATTERY_NAME]

    annual_charge = -battery_p.clip(upper=0).sum()
    annual_discharge = battery_p.clip(lower=0).sum()
    equivalent_cycles = annual_discharge / battery_energy if battery_energy > 0 else 0

    print(f"\nBattery results for {PLOT_YEAR}:")
    print(f"Power capacity: {battery_power:.2f} MW")
    print(f"Energy capacity: {battery_energy:.2f} MWh")
    print(f"Annual charge: {annual_charge:.2f} MWh")
    print(f"Annual discharge: {annual_discharge:.2f} MWh")
    print(f"Equivalent full cycles: {equivalent_cycles:.2f}")

    battery_summary = pd.DataFrame({
        "battery_power_MW": [battery_power],
        "battery_energy_MWh": [battery_energy],
        "annual_charge_MWh": [annual_charge],
        "annual_discharge_MWh": [annual_discharge],
        "equivalent_full_cycles": [equivalent_cycles],
    })

    battery_summary.to_csv(
        os.path.join(DATA_DIR, "taskC_battery_summary.csv"),
        index=False,
    )


# ============================================================
# Plots
# ============================================================

mean_caps = results_df.mean()
std_caps = results_df.std()

plt.figure(figsize=(8, 5))
mean_caps.plot(kind="bar", yerr=std_caps, capsize=4)
plt.ylabel("MW")
plt.title("Average optimal capacity and variability with battery")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskC_capacity_mean_variability.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


results_df.plot(marker="o", figsize=(10, 5))
plt.ylabel("MW")
plt.title("Optimal capacities by weather year with battery")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "taskC_capacities_by_weather_year.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


if plot_network is not None:
    summer_week = slice(f"{YEAR}-07-07", f"{YEAR}-07-13 23:00")
    winter_week = slice(f"{YEAR}-01-13", f"{YEAR}-01-19 23:00")
    full_year = slice(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00")

    plot_dispatch_with_battery(
        plot_network,
        summer_week,
        f"Dispatch in {ZONE} with battery - Summer week ({PLOT_YEAR})",
        "taskC_dispatch_summer_battery.png",
    )

    plot_dispatch_with_battery(
        plot_network,
        winter_week,
        f"Dispatch in {ZONE} with battery - Winter week ({PLOT_YEAR})",
        "taskC_dispatch_winter_battery.png",
    )

    plot_battery_behavior(
        plot_network,
        summer_week,
        f"Battery behavior - Summer week ({PLOT_YEAR})",
        "taskC_battery_behavior_summer.png",
    )

    plot_battery_behavior(
        plot_network,
        winter_week,
        f"Battery behavior - Winter week ({PLOT_YEAR})",
        "taskC_battery_behavior_winter.png",
    )

    plot_battery_behavior(
        plot_network,
        full_year,
        f"Battery behavior - Full year ({PLOT_YEAR})",
        "taskC_battery_behavior_full_year.png",
    )

    battery_soc = plot_network.storage_units_t.state_of_charge[BATTERY_NAME]
    soc_smooth = battery_soc.rolling(24 * 7).mean()

    plt.figure(figsize=(12, 4))
    soc_smooth.plot()
    plt.title(f"Smoothed battery state of charge - weekly average ({PLOT_YEAR})")
    plt.ylabel("MWh")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "taskC_battery_soc_weekly_average.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()