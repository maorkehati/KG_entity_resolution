"""Project paths, WDC URLs, and benchmark variant registry."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

WDC_INTERIM_PRODUCTS_DIR = DATA_INTERIM_DIR / "wdc_products"

WDC_PRODUCTS_PAGE_URL = (
    "https://webdatacommons.org/largescaleproductcorpus/wdc-products/"
)
WDC_PRODUCTS_GITHUB_URL = "https://github.com/wbsg-uni-mannheim/wdcproducts"

WDC_EXPLICIT_CANDIDATE_URLS = [
    "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/data.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/50pair.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/20pair.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/50multi.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/20multi.zip",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/sample_pairwise.json",
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/sample_multi.json",
]


@dataclass(frozen=True)
class WDCVariant:
    """Configuration for one WDC Products benchmark variant."""

    variant_id: str
    task: str
    corner_case_ratio: int
    development_size: str
    unseen_test_ratio: int
    description: str
    file_patterns: dict[str, list[str]]


WDC_VARIANTS: dict[str, WDCVariant] = {
    "pairwise_50_medium_unseen100": WDCVariant(
        variant_id="pairwise_50_medium_unseen100",
        task="pairwise",
        corner_case_ratio=50,
        development_size="medium",
        unseen_test_ratio=100,
        description=(
            "Main experiment: WDC Products pairwise binary matching, "
            "50% corner cases, medium development set, 100% unseen test entities."
        ),
        file_patterns={
            "train": [
                "wdcproducts50cc*train_medium.json.gz",
                "*50*train*medium*.json.gz",
                "*pair*50*medium*train*.json.gz",
                "*50*medium*train*.json.gz",
            ],
            "valid": [
                "wdcproducts50cc*valid_medium.json.gz",
                "*50*valid*medium*.json.gz",
                "*50*val*medium*.json.gz",
                "*pair*50*medium*valid*.json.gz",
                "*pair*50*medium*val*.json.gz",
            ],
            "test": [
                "wdcproducts50cc*rnd100un*gs.json.gz",
                "*50*100*un*gs*.json.gz",
                "*rnd100un*gs*.json.gz",
                "*50*medium*100*test*.json.gz",
                "*test*50*medium*100*.json.gz",
            ],
        },
    ),
}

DEFAULT_WDC_VARIANT_ID = "pairwise_50_medium_unseen100"


def get_wdc_variant(variant_id: str | None = None) -> WDCVariant:
    """Return a registered variant or raise KeyError with available IDs."""
    vid = variant_id or DEFAULT_WDC_VARIANT_ID
    if vid not in WDC_VARIANTS:
        available = ", ".join(sorted(WDC_VARIANTS))
        raise KeyError(
            f"Unknown WDC variant_id={vid!r}. Available variants: {available}"
        )
    return WDC_VARIANTS[vid]
