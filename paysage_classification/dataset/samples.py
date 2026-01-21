from pathlib import Path
from typing import TypedDict

import pandas as pd


class Sample(TypedDict):
    image_name: str
    label: str


def read_samples(
    path_to_data: Path,
) -> list[Sample]:
    df = pd.read_csv(str(path_to_data))
    samples = df.to_dict(orient='records')

    return samples  # type: ignore
