import requests
import pandas as pd
from lxml import etree
from datetime import datetime

# =========================
# ENTSO-E configuration
# =========================

API_TOKEN = "39890c47-a242-475b-9730-cd70dfadf457"
BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Only DK1
ZONE_NAME = "DK1"
ZONE_CODE = "10YDK-1--------W"

YEARS = range(2020, 2026)  # 2020–2025 inclusive


# =========================
# Helper functions
# =========================

def entsoe_request(params):
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return etree.fromstring(response.content)


def monthly_periods(year):
    """Return monthly (start, end) timestamps for one year."""
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


# =========================
# Load (demand)
# =========================

def download_load():
    all_data = []

    for year in YEARS:
        for start, end in monthly_periods(year):
            params = {
                "documentType": "A65",        # Load data
                "processType": "A16",         # Realised
                "outBiddingZone_Domain": ZONE_CODE,
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
                        "zone": ZONE_NAME,
                        "time_utc": start_time + pd.Timedelta(hours=hour),
                        "load_MW": value,
                    })

    df = pd.DataFrame(all_data)
    df.to_csv(
        "Data/raw/load/load_DK1_2020_2025.csv",
        index=False
    )


# =========================
# Generation
# =========================

def download_generation():
    all_data = []

    for year in YEARS:
        for start, end in monthly_periods(year):
            params = {
                "documentType": "A75",        # Generation per type
                "processType": "A16",         # Realised
                "in_Domain": ZONE_CODE,
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
                            "zone": ZONE_NAME,
                            "time_utc": start_time + pd.Timedelta(hours=hour),
                            "psr_type": psr_type,
                            "generation_MW": value,
                        })

    df = pd.DataFrame(all_data)
    df.to_csv(
        "Data/raw/generation/gen_DK1_2020_2025.csv",
        index=False
    )


# =========================
# Main
# =========================

if __name__ == "__main__":
    print("Downloading LOAD for DK1 (2020–2025)")
    download_load()

    print("Downloading GENERATION for DK1 (2020–2025)")
    download_generation()