"""Index past stories (title + description + story points) into Chroma.

    python -m agent1_scoping.index_stories                 # train split only (default)
    python -m agent1_scoping.index_stories --all           # every row (for production)
    python -m agent1_scoping.index_stories --reset         # drop the collection first

Expected CSV columns (one file per project): issuekey, title, description,
storypoint, split_mark. Only title and storypoint are required.
"""
import argparse
from pathlib import Path
from typing import Optional

# Keep this order: rag imports chromadb, which must load BEFORE pandas (Windows
# onnxruntime access-violation otherwise). Do not let an import sorter reorder it.
from agent1_scoping.rag import COLLECTION, _client, get_collection  # isort: skip

import pandas as pd  # isort: skip

DEFAULT_DIR = "data/raw/storypoints"
MAX_DESCRIPTION_CHARS = 800
BATCH = 256


def load_stories(data_dir: str = DEFAULT_DIR, train_only: bool = True) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = {"title", "storypoint"} - set(df.columns)
        if missing:
            raise ValueError(f"{f.name}: missing columns {sorted(missing)}; found {list(df.columns)}")
        df["project"] = f.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if "description" not in df.columns:
        df["description"] = ""
    if "split_mark" in df.columns:
        print("split_mark counts:\n", df["split_mark"].value_counts(dropna=False).to_string())
        if train_only:
            train = df[df["split_mark"].astype(str).str.lower().str.startswith("train")]
            if train.empty:
                print("WARNING: no 'train' rows found, indexing everything.")
            else:
                df = train

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = (
        df["description"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        .str.slice(0, MAX_DESCRIPTION_CHARS)
    )
    df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
    df = df[(df["title"] != "") & (df["storypoint"] > 0)]
    df = df.drop_duplicates(subset=["project", "title", "description"]).reset_index(drop=True)
    return df


def index_stories(df: pd.DataFrame, collection) -> int:
    for start in range(0, len(df), BATCH):
        chunk = df.iloc[start : start + BATCH]
        collection.upsert(
            ids=[f"{r.project}-{i}" for i, r in zip(chunk.index, chunk.itertuples())],
            documents=[f"{r.title}. {r.description}".strip() for r in chunk.itertuples()],
            metadatas=[
                {"title": r.title, "points": int(round(r.storypoint)), "project": r.project}
                for r in chunk.itertuples()
            ],
        )
        print(f"indexed {min(start + BATCH, len(df))}/{len(df)}", flush=True)
    return len(df)


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DIR)
    p.add_argument("--all", action="store_true", help="index every split, not just train")
    p.add_argument("--reset", action="store_true", help="delete the collection first")
    args = p.parse_args(argv)

    df = load_stories(args.data_dir, train_only=not args.all)
    print(f"{len(df)} stories from {df['project'].nunique()} projects; "
          f"points range {df['storypoint'].min():g}-{df['storypoint'].max():g}")
    if args.reset:
        try:
            _client().delete_collection(COLLECTION)
        except Exception:
            pass
    index_stories(df, get_collection(create=True))
    print("done.")


if __name__ == "__main__":
    main()
