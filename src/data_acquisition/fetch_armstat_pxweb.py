from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "processed"

HEADERS = {"User-Agent": "Mozilla/5.0"}

CATEGORY_SPECS = [
    {
        "name": "industry_construction_trade_services",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__3%20Industry,%20Construction,%20trade%20and%20services/",
        "event_argument": (
            "pArmStatBank__3 Industry, Construction, trade and services\\"
            "ArmStatBank__3 Industry, Construction, trade and services__31 Construction"
        ),
    },
]


def _request_with_retries(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        try:
            response = session.request(method, url, timeout=60, headers=HEADERS, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _expanded_category_html(session: requests.Session, url: str, event_argument: str) -> str:
    initial = _request_with_retries(session, "GET", url)
    soup = BeautifulSoup(initial.text, "html.parser")
    payload = {item.get("name"): item.get("value", "") for item in soup.select("input[name]") if item.get("name")}
    payload["__EVENTTARGET"] = "ctl00$ContentPlaceHolderMain$TableOfContent1$TableOfContent1$MenuNavigationTree"
    payload["__EVENTARGUMENT"] = event_argument
    expanded = _request_with_retries(session, "POST", url, data=payload)
    return expanded.text


def _extract_table_manifest(html: str) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a href="(?P<href>[^"]+\.px/)">\s*<span class=\'tableofcontent_link\'>(?P<title>.*?)<img.*?'
        r"title='Size: (?P<size>[^']+)'.*?title='Updated: (?P<updated>[^']+)'",
        flags=re.S,
    )
    for match in pattern.finditer(html):
        title = re.sub(r"<.*?>", "", match.group("title"))
        title = re.sub(r"\s+", " ", title).strip()
        href = match.group("href").replace("&amp;", "&")
        records.append(
            {
                "title": title,
                "relative_url": href,
                "url": f"https://statbank.armstat.am{href}",
                "size": match.group("size").strip(),
                "updated": match.group("updated").strip(),
            }
        )
    manifest = pd.DataFrame.from_records(records).drop_duplicates(subset=["url"]).sort_values("title")
    return manifest


def _priority_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    category_mask = manifest["relative_url"].str.contains(r"/ArmStatBank__3(?: |%20)", case=False, na=False, regex=True)
    monthly_mask = manifest["relative_url"].str.contains("Monthly indicators|m-", case=False, na=False)
    keyword_mask = manifest["title"].str.contains(
        "construction|industry|trade|services|retail|employment|wages|employees",
        case=False,
        na=False,
    )
    return manifest[category_mask & (monthly_mask | keyword_mask)].copy()


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    outputs: list[pd.DataFrame] = []

    for spec in CATEGORY_SPECS:
        print(f"Fetching ArmStat category tree: {spec['name']}")
        html = _expanded_category_html(session, spec["url"], spec["event_argument"])
        manifest = _extract_table_manifest(html)
        manifest.insert(0, "category", spec["name"])
        outputs.append(manifest)
        print(f"  discovered {len(manifest)} tables")

    full_manifest = pd.concat(outputs, ignore_index=True).drop_duplicates(subset=["url"])
    priority = _priority_subset(full_manifest)

    full_manifest.to_csv(PROC_DIR / "armstat_pxweb_manifest.csv", index=False)
    priority.to_csv(PROC_DIR / "armstat_pxweb_priority_tables.csv", index=False)

    print(f"Saved manifest to {PROC_DIR / 'armstat_pxweb_manifest.csv'}")
    print(f"Saved priority subset to {PROC_DIR / 'armstat_pxweb_priority_tables.csv'}")


if __name__ == "__main__":
    main()
