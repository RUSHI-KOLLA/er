from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STUDENT_RESOURCE_ROOT = PACKAGE_ROOT.parents[1]


@dataclass(frozen=True)
class Settings:
    data_root: Path = STUDENT_RESOURCE_ROOT / "dataset"
    output_dir: Path = STUDENT_RESOURCE_ROOT / "output"
    candidate_cap: int = 50
    bucket_limit: int = 5000
    blocking_stage1: int = 2000
    index_batch_size: int = 50_000
    feature_batch_size: int = 100_000
    source_batch_size: int = 256
    query_batch_size: int = 64
    feature_batch_pairs: int = 20_000
    max_in_memory_targets: int = 1_000_000
    max_in_memory_edges: int = 2_000_000
    max_training_entities: int = 100_000
    training_pair_cap: int = 500_000
    validation_fraction: float = 0.2
    seed: int = 2026
    match_threshold: float = 0.72
    country_fallback_limit: int = 10
    model_path: Path = PACKAGE_ROOT / "artifacts" / "matcher.joblib"

    @property
    def train_dir(self) -> Path:
        return self.data_root / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_root / "test"

    @property
    def train_ground_truth_path(self) -> Path:
        return self.train_dir / "train_ground_truth.tsv"

    @property
    def train_source1_path(self) -> Path:
        return self.train_dir / "train_source1.tsv"

    @property
    def test_source1_path(self) -> Path:
        return self.test_dir / "test_source1.tsv"

    def source_path(self, split: str, source: int) -> Path:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train or test: {split}")
        if source not in {1, 2, 3}:
            raise ValueError(f"source must be 1, 2, or 3: {source}")
        prefix = "train" if split == "train" else "test"
        return self.data_root / split / f"{prefix}_source{source}.tsv"

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()
