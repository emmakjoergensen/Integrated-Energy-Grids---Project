# Assumptions and References

This document summarizes the main assumptions applied in Tasks A-H of the Integrated Energy Grids project. It only includes assumptions that are still reflected in the current modelling approach.

---

## General data sources

### Electricity demand and generation data

Hourly electricity demand and generation data are based on Energinet / Energi Data Service and ENTSO-E style bidding-zone data. For DK1, DK2, NO2 and DE, the model uses hourly load files and generation files stored in the project folder under:

```text
Data/raw/load/
Data/raw/generation/
```

The generation data use ENTSO-E PSR production type codes. The following mapping is applied:

| Technology | PSR codes |
|---|---|
| Onshore wind | B19 |
| Offshore wind | B18 |
| Solar PV | B16 |
| Run-of-river hydro | B11 |
| Reservoir hydro | B12 |
| Coal | B02, B03, B05 |
| Gas | B04 |

Relevant source links:

- Energi Data Service: https://www.energidataservice.dk/
- Production and consumption settlement: https://www.energidataservice.dk/tso-electricity/productionconsumptionsettlement
- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/

---

## Task A: Single-node DK1 capacity expansion

### Model scope

Task A models DK1 as a single-node electricity system. The model includes:

- Solar PV
- Onshore wind
- Offshore wind
- OCGT gas-fired generation

All generation capacities are optimised endogenously using `p_nom_extendable=True`. The model assumes perfect foresight over the full year and requires demand to be met in every hour.

### Demand

Hourly electricity demand for DK1 is based on the 2025 load data. Demand is assumed to be inelastic and must be met in every modelled hour.

### Renewable generation profiles

Hourly generation time series are used to construct renewable availability profiles. Because the model optimises installed capacity, historical generation is not used directly as fixed generation. Instead, it is converted into an hourly availability profile, `p_max_pu`.

The raw profile is calculated as:

$$
p^{\text{raw}}_t = \frac{\text{generation}_t}{\max(\text{generation})}
$$

The profile is then scaled to match an assumed target capacity factor:

$$
p_{\text{max\_pu},t} = \frac{p^{\text{raw}}_t}{\overline{p^{\text{raw}}}} \cdot CF_{\text{target}}
$$

Finally, the profile is clipped to ensure values remain between 0 and 1.

The capacity factor assumptions are:

| Technology | Target capacity factor |
|---|---:|
| Solar PV | 0.11 |
| Onshore wind | 0.25 |
| Offshore wind | 0.52 |

The offshore wind value of 52% is chosen because offshore wind turbines can reach higher capacity factors than onshore turbines due to stronger and more consistent offshore wind conditions. Opoura reports that offshore wind turbine capacity factors can range up to 52%, and notes that offshore turbines generally perform better than onshore turbines. The 52% assumption is therefore used as a high but defensible value for offshore wind in a future-oriented 2030 cost scenario.

Source:

- Opoura, *What is the capacity factor of a wind turbine?*: https://opoura.com/energy-explained-learning-hub/wind-turbine-capacity-factor/

### Cost assumptions

Technology costs are taken from the project file `costs_PyPSA.csv`, based on PyPSA Technology Data for 2030. Investment costs are converted from EUR/kW to EUR/MW.

Relevant source:

- PyPSA Technology Data: https://github.com/PyPSA/technology-data

The annualized fixed cost is calculated as annualized CAPEX plus fixed OPEX:

$$
\text{Total fixed cost} = \text{CAPEX} \cdot \frac{r}{1 - (1+r)^{-n}} + \text{FOM}
$$

where:

- $r$ is the discount rate
- $n$ is the technology lifetime
- FOM is fixed operation and maintenance cost

The discount rate is assumed to be 7%.

### Marginal costs

Renewable generators are assumed to have zero marginal cost:

$$
MC_{\text{RES}} = 0
$$

OCGT marginal cost is calculated as:

$$
MC_{\text{OCGT}} = \frac{P_{\text{gas}}}{\eta} + \frac{P_{\text{CO2}} \cdot I_{\text{CO2}}}{\eta} + VOM
$$

The assumptions are:

| Parameter | Value |
|---|---:|
| Gas price | 40 EUR/MWh_th |
| CO2 price | 80 EUR/tCO2 |
| Gas CO2 intensity | 0.202 tCO2/MWh_th |
| OCGT efficiency | From `costs_PyPSA.csv` |
| OCGT VOM | From `costs_PyPSA.csv` |

---

## Task B: Sensitivity to interannual renewable variability

Task B investigates how the optimal capacity mix changes when renewable weather years are varied.

### Main assumptions

- Weather years 2020-2025 are tested.
- Electricity demand is kept fixed to the 2025 DK1 demand profile in all model runs.
- Only solar, onshore wind and offshore wind availability profiles vary between weather years.
- Technology costs, fuel prices, CO2 price and model structure are held constant across all years.
- The same target capacity factors are used as in Task A:
  - Solar PV: 0.11
  - Onshore wind: 0.25
  - Offshore wind: 0.52

Keeping demand fixed isolates the effect of interannual renewable variability on the optimal capacity mix.

---

## Task C: Adding battery storage

Task C extends the Task B model by adding battery storage.

### Battery representation

Battery storage is modelled using a PyPSA `StorageUnit`. The battery is allowed to expand endogenously using `p_nom_extendable=True`.

The following assumptions are applied:

| Parameter | Assumption |
|---|---:|
| Battery duration | 4 hours |
| Charging efficiency | From `battery inverter` in `costs_PyPSA.csv` or assumed 0.95 in the single-node version |
| Discharging efficiency | From `battery inverter` in `costs_PyPSA.csv` or assumed 0.95 in the single-node version |
| Marginal cost | 0 EUR/MWh |
| Cyclic state of charge | True |

### Battery cost treatment

Battery costs are split into:

- `battery inverter`, representing power capacity cost in EUR/kW
- `battery storage`, representing energy capacity cost in EUR/kWh

For a 4-hour battery, the annualized battery cost per MW is calculated as:

$$
C_{\text{battery}} = C_{\text{inverter}} + 4 \cdot C_{\text{storage}}
$$

This ensures that the cost of energy capacity is scaled consistently with the assumed storage duration.

---

## Task D: Interconnected electricity system

Task D extends the model to include four bidding zones:

- DK1
- DK2
- NO2
- DE

The network includes a closed cycle:

$$
DK1 \rightarrow DK2 \rightarrow DE \rightarrow DK1
$$

### Electricity network

All interconnectors are represented as HVAC lines using linearised power flow. The assignment specifies a voltage level of 400 kV and unitary reactance $x = 0.1$. The model uses:

| Line | Capacity |
|---|---:|
| DK1 - NO2 | 1700 MW |
| DK1 - DE | 2500 MW |
| DK1 - DK2 | 600 MW |
| DK2 - DE | 600 MW |

Line resistance is set to a small value, $r = 0.0001$, for numerical stability.

### Regional generation

For each bidding zone, existing renewable and fossil capacities are approximated from the maximum observed historical hourly generation in the data. These existing capacities are used as lower bounds in the optimisation by setting:

```python
p_nom_min = existing_capacity
```

This means the model can expand capacity but cannot remove existing capacity.

### Hydro reservoir in NO2

Hydro reservoir generation in NO2 is represented as a dispatchable generator with fixed existing capacity. Explicit reservoir inflow and storage are not modelled. Instead, total annual hydro dispatch is constrained by historical annual hydro generation:

$$
\sum_{t \in T} p_{g,t} \cdot \Delta t \le E^{\text{hist}}_g
$$

where:

- $p_{g,t}$ is hydro dispatch in MW
- $\Delta t$ is the snapshot duration, equal to 1 hour
- $E^{\text{hist}}_g$ is historical annual hydro reservoir generation in MWh

This approximates seasonal hydro flexibility without explicitly modelling reservoir storage dynamics.

### Coal in Germany

Coal generation is included only in Germany. The assumptions are:

| Parameter | Value |
|---|---:|
| Coal CAPEX | 1500 EUR/kW |
| Lifetime | 40 years |
| FOM | 3%/year |
| Efficiency | 0.38 |
| VOM | 3 EUR/MWh |
| Fuel price | 15 EUR/MWh_th |
| CO2 intensity | 0.34 tCO2/MWh_th |

Coal marginal cost is calculated as:

$$
MC_{\text{coal}} = \frac{P_{\text{coal}}}{\eta} + \frac{P_{\text{CO2}} \cdot I_{\text{CO2}}}{\eta} + VOM
$$

References used for broad coal assumptions:

- Brown, T., Hörsch, J., & Schlachtberger, D. (2018). *PyPSA: Python for Power System Analysis*. Journal of Open Research Software, 6(1).
- International Energy Agency, *Projected Costs of Generating Electricity*.

---

## Task F: CO2 sensitivity in the single-country model

Task F uses the single-country model from Task C and investigates sensitivity to a global CO2 constraint.

The applied assumptions are:

- The system is based on the DK1 model with battery storage.
- A system-wide CO2 limit is imposed using a PyPSA `GlobalConstraint`.
- The CO2 constraint applies to fossil gas generation.
- Renewable technologies and battery storage are treated as zero direct emissions.
- The generation mix and capacity mix are evaluated as a function of the imposed CO2 allowance.

The emission factor for gas is:

$$
I_{\text{gas}} = 0.202\ \text{tCO2/MWh}_{th}
$$

---

## Task G: Integrated electricity and CH4 network

Task G extends the interconnected electricity system from Task D by adding a methane gas network.

### Gas network structure

For each electricity zone, a corresponding CH4 bus is added:

- CH4_bus_DK1
- CH4_bus_DK2
- CH4_bus_NO2
- CH4_bus_DE

The gas system includes:

- CH4 supply at each gas bus
- CH4 pipelines between regions
- Gas-to-power links converting CH4 to electricity

The gas-to-power links represent OCGT units consuming methane and producing electricity.

### CH4 supply

CH4 supply is modelled as an extendable generator at each CH4 bus. The marginal cost includes fuel and CO2 cost:

$$
MC_{\text{CH4 supply}} = P_{\text{gas}} + P_{\text{CO2}} \cdot I_{\text{CO2}}
$$

with:

| Parameter | Value |
|---|---:|
| Gas price | 40 EUR/MWh_th |
| CO2 price | 80 EUR/tCO2 |
| Gas CO2 intensity | 0.202 tCO2/MWh_th |

### Gas-to-power conversion

Gas-to-power conversion is represented using PyPSA `Link` components:

```text
CH4 bus -> electricity bus
```

The link efficiency is equal to the OCGT efficiency from `costs_PyPSA.csv`.

### CH4 pipelines

CH4 pipelines are represented as bidirectional PyPSA `Link` components using:

```python
p_min_pu = -1
```

This allows gas transport in both directions. Pipeline efficiency is set to 0.99 to represent simplified compressor losses.

The assumed pipeline capacities are:

| Pipeline | Capacity |
|---|---:|
| DK1 - NO2 | 3000 MW_th |
| DK1 - DE | 5000 MW_th |
| DK1 - DK2 | 1500 MW_th |
| DK2 - DE | 3000 MW_th |

The model compares annual transported energy in:

- HVAC electricity lines
- CH4 pipelines

Annual transported energy is calculated as the sum of absolute hourly flows.

---

## Task H: CO2 price required for a selected decarbonisation target

Task H uses the integrated electricity and CH4 system from Task G.

### CO2 allowance

A CO2 allowance is selected as a fraction of baseline emissions. In the current implementation, the cap is set to:

$$
CO2_{\text{limit}} = 0.8 \cdot CO2_{\text{baseline}}
$$

This corresponds to an emissions allowance equal to 80% of baseline model emissions.

### Baseline emissions

Baseline emissions are calculated from:

1. CH4 input to gas-to-power links
2. Coal generation in Germany

Gas emissions:

$$
CO2_{\text{gas}} = E_{\text{CH4 input}} \cdot I_{\text{gas}}
$$

Coal emissions:

$$
CO2_{\text{coal}} = \frac{E_{\text{coal,electricity}}}{\eta_{\text{coal}}} \cdot I_{\text{coal}}
$$

Total baseline emissions:

$$
CO2_{\text{baseline}} = CO2_{\text{gas}} + CO2_{\text{coal}}
$$

### CO2 constraint

The CO2 cap is implemented using a PyPSA `GlobalConstraint` of type `primary_energy`.

Emission factors are assigned to carriers:

| Carrier | Emission factor |
|---|---:|
| CH4 | 0.202 tCO2/MWh_th |
| Coal | 0.34 / 0.38 tCO2/MWh_el equivalent |
| Wind, solar, hydro, battery, electricity lines, CH4 pipelines | 0 |

### CO2 shadow price

The CO2 price required to achieve the selected emissions cap is obtained from the shadow price of the CO2 constraint.

Because of PyPSA's sign convention for `<=` constraints, the additional CO2 price is calculated as:

$$
P_{\text{CO2,additional}} = -\mu
$$

The total effective CO2 price is then:

$$
P_{\text{CO2,total}} = P_{\text{CO2,existing}} + P_{\text{CO2,additional}}
$$

where the existing assumed CO2 price is 80 EUR/tCO2.

Task H compares:

- Baseline emissions
- CO2-constrained emissions
- The selected CO2 limit
- Capacity changes before and after the CO2 constraint
- The required additional and total effective CO2 price

---

## General modelling limitations

The following simplifications apply across the model:

- Perfect foresight over the full model year is assumed.
- Demand is price-inelastic.
- Unit commitment constraints are not included.
- Ramping constraints, minimum stable generation and start-up costs are not modelled.
- Network losses are simplified or ignored except for stylised pipeline efficiency.
- Existing capacities are approximated from maximum observed hourly generation, not from official installed capacity statistics.
- Renewable generation profiles are derived from historical generation rather than meteorological reanalysis data.
- Fuel and CO2 prices are assumed constant over the year.
