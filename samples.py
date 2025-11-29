import pandas as pd
from pathlib import Path
from typing import TypedDict

class Sample(TypedDict):
    image_name: str
    label: str


def read_samples(
    path_to_data: Path,
) -> list[Sample]:
    df = pd.read_csv(str(path_to_data))
    samples = df.to_dict(orient='records')

    return samples # type: ignore


def create_splits(
    path_to_data: Path,
    ratio: str,
    path_to_save: str
) -> None:
    df = pd.read_csv(str(path_to_data))

    r_train, r_test, r_eval = map(float, ratio.split('-'))
    # Перемешиваем строки для случайной выборки
    df = df.sample(frac=1).reset_index(drop=True)

    n_train = int(len(df) * r_train)
    n_test = int(len(df) * r_test)
    n_eval = len(df) - n_train - n_test

    # Всё, сохраняем каждую выборку в отдельный CSV
    df_train = df.iloc[: n_train]
    df_train.to_csv(f'{path_to_save}train.csv', index=False)
    df_eval = df.iloc[(n_train + n_test) :]
    df_eval.to_csv(f'{path_to_save}eval.csv', index=False)
    df_test = df.iloc[n_train : (n_train + n_test)]
    df_test.to_csv(f'{path_to_save}test.csv', index=False)


def read_splits(
    chosen_splits: list[str],
    path_to_save: str
) -> list[list[Sample]]:
    samples = []

    # Разрешённые названия сплитов -- "train", "eval", "test"
    for split in chosen_splits:
        df = pd.read_csv(f'{path_to_save}{split}.csv')
        # print(df.to_dict(orient='records'))
        samples.append(df.to_dict(orient='records'))

    return samples
