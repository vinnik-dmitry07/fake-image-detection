import itertools
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PREFIX_IN = Path('D:/Downloads/archive')
PREFIX_OUT = Path('./artifact_60k_balanced')
TOTAL_DATASET_LEN = 60_000
REAL_IDX = 0
FAKE_IDX = 1

df = pd.DataFrame()
for i, csv_path in enumerate(PREFIX_IN.glob('*/*.csv')):
    add = pd.read_csv(csv_path, usecols=['image_path', 'target'])
    add.image_path = csv_path.parts[-2] + '/' + add.image_path
    add.target = add.target.clip(upper=1)
    add['dataset_idx'] = i
    df = pd.concat([df, add])


def group_arrange(data_frame: pd.DataFrame, group_name: str, total_elements: int):
    """
    [1, 2, 3, 4, 5, 6]; [a, b, c, d]; [1, 2]
    => 1 a 1; 2 b 2; 3 c 1; 4 d 2; 5 a 1; 6 b 2
    """
    group_indices: dict = data_frame.groupby(group_name).indices
    transposed_generator = zip(*map(itertools.cycle, group_indices.values()))
    tuples_number = total_elements // len(group_indices) + 1
    transposed_generator = itertools.islice(transposed_generator, tuples_number)
    transposed_indices = list(sum(transposed_generator, ())[:total_elements])
    data_frame = data_frame.iloc[transposed_indices]
    return data_frame


df1 = df.groupby('target').apply(
    lambda grp:
    group_arrange(
        data_frame=grp.sample(len(df), replace=True),
        group_name='dataset_idx',
        total_elements=TOTAL_DATASET_LEN // 2
    )
)
assert len(df1) == TOTAL_DATASET_LEN, len(df1)
assert df1.target.sum() * 2 == TOTAL_DATASET_LEN, df1.target.sum()

df1 = df1.reset_index(drop=True)

if PREFIX_OUT.exists():
    shutil.rmtree(PREFIX_OUT)
PREFIX_OUT.mkdir()
(PREFIX_OUT / 'real').mkdir()
(PREFIX_OUT / 'fake').mkdir()

for _, row in tqdm(df1.iterrows(), total=len(df1)):
    new_name = row.image_path.replace('/', '_')
    shutil.copy(
        PREFIX_IN / row.image_path,
        PREFIX_OUT / f'{"fake" if row.target else "real"}/{new_name}'
    )
