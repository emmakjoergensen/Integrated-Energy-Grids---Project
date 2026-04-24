import requests
import pandas as pd
from lxml import etree
from datetime import datetime

# API Token for Transparancy ENSO E
# 39890c47-a242-475b-9730-cd70dfadf457

API_TOKEN = "39890c47-a242-475b-9730-cd70dfadf457"

BASE_URL = "https://web-api.tp.entsoe.eu/api"

ZONES = {
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "NO2": "10YNO-2--------T",
    "DE":  "10Y1001A1001A83F",
}

YEAR = 2025


def entsoe_request(params):
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return etree.fromstring(response.content)


def monthly_periods(year):
    periods = []
    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        periods.append((
            start.strftime("%Y%m%d%H%M"),
            end.strftime("%Y%m%d%H%M")
        ))
    return periods


def download_load(zone_name, zone_code):
    all_data = []

    for start, end in monthly_periods(YEAR):
        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": zone_code,
            "periodStart": start,
            "periodEnd": end,
            "securityToken": API_TOKEN,
        }

        xml = entsoe_request(params)

        for period in xml.findall(".//{*}Period"):
            start_time = pd.to_datetime(
                period.find(".//{*}timeInterval/{*}start").text
            )

            for point in period.findall(".//{*}Point"):
                hour = int(point.find("{*}position").text) - 1
                value = float(point.find("{*}quantity").text)

                all_data.append({
                    "zone": zone_name,
                    "time_utc": start_time + pd.Timedelta(hours=hour),
                    "load_MW": value,
                })

    df = pd.DataFrame(all_data)
    print(df)
    df.to_csv(f"Data/raw/load/load_{zone_name}_2025.csv", index=False)


def download_generation(zone_name, zone_code):
    all_data = []

    for start, end in monthly_periods(YEAR):
        params = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": zone_code,
            "periodStart": start,
            "periodEnd": end,
            "securityToken": API_TOKEN,
        }

        xml = entsoe_request(params)

        for ts in xml.findall(".//{*}TimeSeries"):
            psr_type = ts.find(".//{*}psrType").text

            for period in ts.findall(".//{*}Period"):
                start_time = pd.to_datetime(
                    period.find(".//{*}timeInterval/{*}start").text
                )

                for point in period.findall(".//{*}Point"):
                    hour = int(point.find("{*}position").text) - 1
                    value = float(point.find("{*}quantity").text)

                    all_data.append({
                        "zone": zone_name,
                        "time_utc": start_time + pd.Timedelta(hours=hour),
                        "psr_type": psr_type,
                        "generation_MW": value,
                    })

    df = pd.DataFrame(all_data)
    df.to_csv(f"Data/raw/generation/gen_{zone_name}_2025.csv", index=False)


if __name__ == "__main__":
    for zone_name, zone_code in ZONES.items():
        print(f"Downloading LOAD for {zone_name}")
        download_load(zone_name, zone_code)

        print(f"Downloading GENERATION for {zone_name}")
        download_generation(zone_name, zone_code)
