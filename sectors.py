"""Index universe for the Sector Rotation Screener.

Each entry describes one NSE index:
    key        short id used inside the app
    label      what the user sees
    nse_name   exact index name used by niftyindices.com / nseindia.com
    yf         Yahoo Finance ticker with long price history (None = not available)
    csv        constituent file name on niftyindices.com/IndexConstituent/
    group      "Sector", "Thematic" or "Size"
    pe_name    short name the PE/PB/DY history uses, when it differs (checked Oct 2026)

Yahoo carries long history only for some NSE indices (checked Oct 2026).
For the rest the app tries niftyindices.com and then builds an equal-weight
proxy from the constituents' own prices.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexDef:
    key: str
    label: str
    nse_name: str
    yf: str | None
    csv: str
    group: str
    pe_name: str | None = None   # name used by the niftyindices PE/PB/DY history, when it differs

    @property
    def pe_key(self) -> str:
        return self.pe_name or self.nse_name


BENCHMARKS = {
    "NIFTY50": IndexDef("NIFTY50", "Nifty 50", "NIFTY 50", "^NSEI", "ind_nifty50list.csv", "Size"),
    "NIFTY500": IndexDef("NIFTY500", "Nifty 500", "NIFTY 500", "^CRSLDX", "ind_nifty500list.csv", "Size"),
}

INDICES: list[IndexDef] = [
    # ---- sectoral ---------------------------------------------------------
    IndexDef("AUTO", "Auto", "NIFTY AUTO", "^CNXAUTO", "ind_niftyautolist.csv", "Sector"),
    IndexDef("BANK", "Bank", "NIFTY BANK", "^NSEBANK", "ind_niftybanklist.csv", "Sector"),
    IndexDef("PVTBANK", "Private Bank", "NIFTY PRIVATE BANK", None, "ind_nifty_privatebanklist.csv", "Sector", "NIFTY PVT BANK"),
    IndexDef("PSUBANK", "PSU Bank", "NIFTY PSU BANK", "^CNXPSUBANK", "ind_niftypsubanklist.csv", "Sector"),
    IndexDef("FIN", "Financial Services", "NIFTY FINANCIAL SERVICES", "NIFTY_FIN_SERVICE.NS", "ind_niftyfinancelist.csv", "Sector", "NIFTY FIN SERVICE"),
    IndexDef("FMCG", "FMCG", "NIFTY FMCG", "^CNXFMCG", "ind_niftyfmcglist.csv", "Sector"),
    IndexDef("IT", "IT", "NIFTY IT", "^CNXIT", "ind_niftyitlist.csv", "Sector"),
    IndexDef("PHARMA", "Pharma", "NIFTY PHARMA", "^CNXPHARMA", "ind_niftypharmalist.csv", "Sector"),
    IndexDef("HEALTH", "Healthcare", "NIFTY HEALTHCARE INDEX", None, "ind_niftyhealthcarelist.csv", "Sector", "NIFTY HEALTHCARE"),
    IndexDef("METAL", "Metal", "NIFTY METAL", "^CNXMETAL", "ind_niftymetallist.csv", "Sector"),
    IndexDef("REALTY", "Realty", "NIFTY REALTY", "^CNXREALTY", "ind_niftyrealtylist.csv", "Sector"),
    IndexDef("MEDIA", "Media", "NIFTY MEDIA", None, "ind_niftymedialist.csv", "Sector"),
    IndexDef("CONSDUR", "Consumer Durables", "NIFTY CONSUMER DURABLES", None, "ind_niftyconsumerdurableslist.csv", "Sector", "NIFTY CONSR DURBL"),
    IndexDef("OILGAS", "Oil & Gas", "NIFTY OIL & GAS", None, "ind_niftyoilgaslist.csv", "Sector", "NIFTY OIL AND GAS"),
    # ---- thematic ---------------------------------------------------------
    IndexDef("ENERGY", "Energy", "NIFTY ENERGY", "^CNXENERGY", "ind_niftyenergylist.csv", "Thematic"),
    IndexDef("INFRA", "Infrastructure", "NIFTY INFRASTRUCTURE", "^CNXINFRA", "ind_niftyinfralist.csv", "Thematic", "NIFTY INFRA"),
    IndexDef("COMMOD", "Commodities", "NIFTY COMMODITIES", None, "ind_niftycommoditieslist.csv", "Thematic"),
    IndexDef("CONSUMP", "India Consumption", "NIFTY INDIA CONSUMPTION", "^CNXCONSUM", "ind_niftyconsumptionlist.csv", "Thematic", "NIFTY CONSUMPTION"),
    IndexDef("PSE", "PSE", "NIFTY PSE", "^CNXPSE", "ind_niftypselist.csv", "Thematic"),
    IndexDef("MNC", "MNC", "NIFTY MNC", "^CNXMNC", "ind_niftymnclist.csv", "Thematic"),
    IndexDef("DEFENCE", "India Defence", "NIFTY INDIA DEFENCE", None, "ind_niftyindiadefence_list.csv", "Thematic", "NIFTY IND DEFENCE"),
    # ---- size / broad -----------------------------------------------------
    IndexDef("NEXT50", "Nifty Next 50", "NIFTY NEXT 50", "^NSMIDCP", "ind_niftynext50list.csv", "Size"),
    IndexDef("MID150", "Midcap 150", "NIFTY MIDCAP 150", "NIFTYMIDCAP150.NS", "ind_niftymidcap150list.csv", "Size"),
    IndexDef("SMALL250", "Smallcap 250", "NIFTY SMALLCAP 250", "NIFTYSMLCAP250.NS", "ind_niftysmallcap250list.csv", "Size", "NIFTY SMLCAP 250"),
    IndexDef("NIFTY500", "Nifty 500", "NIFTY 500", "^CRSLDX", "ind_nifty500list.csv", "Size"),
]

INDEX_BY_KEY = {i.key: i for i in INDICES} | BENCHMARKS
GROUPS = ["Sector", "Thematic", "Size"]
