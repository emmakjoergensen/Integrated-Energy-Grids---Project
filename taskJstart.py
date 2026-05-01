# ================================
# Bornholm Energy Island parameters
# ================================

island_name = "Bornholm"

# HVAC interconnector capacities [MW]
cap_Bornholm_DK2 = 2000.0   # MW
cap_Bornholm_DE  = 3000.0   # MW

# Offshore wind on the island
offshore_capacity = 4000.0  # MW
offshore_capital_cost = offshore_capital_cost  # €/MW

multi_n.add(
    "Bus",
    f"bus_{island_name}",
    carrier="AC"
)

multi_n.add(
    "Line",
    f"{island_name}_DK2",
    bus0=f"bus_{island_name}",
    bus1="bus_DK2",
    x=0.1,
    s_nom=cap_Bornholm_DK2
)
multi_n.add(
    "Line",
    f"{island_name}_DE",
    bus0=f"bus_{island_name}",
    bus1="bus_DE",
    x=0.1,
    s_nom=cap_Bornholm_DE
)

multi_n.add(
    "Generator",
    f"offwind_{island_name}",
    bus=f"bus_{island_name}",
    p_nom=offshore_capacity,        # fixed capacity
    p_max_pu=offshore_cf,           # your offshore wind profile
    capital_cost=offshore_capital_cost,
    marginal_cost=0.0,
    carrier="offwind"
)

print("Bornholm bus:")
print(multi_n.buses.loc[f"bus_{island_name}"])

print("\nBornholm connections:")
print(multi_n.lines.loc[
    multi_n.lines.index.str.contains(island_name),
    ["bus0", "bus1", "s_nom"]
])

print("\nBornholm offshore wind:")
print(multi_n.generators.loc[f"offwind_{island_name}"])

multi_n.optimize()