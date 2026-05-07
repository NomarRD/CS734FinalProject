"""Dataset loading utilities for the MS MARCO passage project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd


@dataclass
class Passage:
    """
    Represents a single passage.

    @param pid: Passage id.
    @param text: Passage text.
    """

    pid: str
    text: str


@dataclass
class QueryExample:
    """
    Represents a single query.

    @param qid: Query id.
    @param text: Query text.
    """

    qid: str
    text: str


class MSMarcoDataLoader:
    """Loads MS MARCO passages, queries, and qrels from TSV files."""

    @staticmethod
    def load_queries(file_path: str) -> List[QueryExample]:
        """
        Loads queries from a TSV file.

        @param file_path: Path to the query file.
        @return: List of QueryExample objects.
        @raises FileNotFoundError: If the file does not exist.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {file_path}")

        df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"], dtype=str)
        return [QueryExample(qid=row.qid, text=row.query) for row in df.itertuples(index=False)]

    @staticmethod
    def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
        """
        Loads qrels from a TSV file.

        MS MARCO dev qrels usually have 4 columns:
        qid, unused-column, pid, relevance

        @param file_path: Path to the qrels file.
        @return: Dictionary of the form {qid: {pid: relevance}}.
        @raises FileNotFoundError: If the file does not exist.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Qrels file not found: {file_path}")

        df = pd.read_csv(path, sep="\t", header=None, dtype=str)
        if df.shape[1] == 4:
            df.columns = ["qid", "unused", "pid", "relevance"]
        elif df.shape[1] == 3:
            df.columns = ["qid", "pid", "relevance"]
        else:
            raise ValueError(f"Unexpected qrels format with {df.shape[1]} columns.")

        qrels: Dict[str, Dict[str, int]] = {}
        for row in df.itertuples(index=False):
            qrels.setdefault(str(row.qid), {})[str(row.pid)] = int(row.relevance)
        return qrels

    @staticmethod
    def iterate_collection(file_path: str) -> Iterator[Passage]:
        """
        Streams the MS MARCO collection TSV row by row.

        The expected collection format is:
        pid<TAB>passage_text

        @param file_path: Path to the collection TSV file.
        @return: Iterator of Passage objects.
        @raises FileNotFoundError: If the file does not exist.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Collection file not found: {file_path}")

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                yield Passage(pid=parts[0], text=parts[1])

    @staticmethod
    def load_passages(file_path: str) -> List[Passage]:
        """
        Loads all passages from a TSV file into memory.

        @param file_path: Path to the collection or subset TSV file.
        @return: List of Passage objects.
        """

        return list(MSMarcoDataLoader.iterate_collection(file_path))

    @staticmethod
    def save_queries(file_path: str, queries: List[QueryExample]) -> None:
        """
        Saves query examples to a TSV file.

        @param file_path: Output path.
        @param queries: Query examples to write.
        @return: None.
        """

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for query in queries:
                handle.write(f"{query.qid}\t{query.text}\n")

    @staticmethod
    def save_qrels(file_path: str, qrels: Dict[str, Dict[str, int]]) -> None:
        """
        Saves qrels to a TSV file using the 4-column format.

        @param file_path: Output path.
        @param qrels: Qrels dictionary.
        @return: None.
        """

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for qid, pid_map in qrels.items():
                for pid, relevance in pid_map.items():
                    handle.write(f"{qid}\t0\t{pid}\t{relevance}\n")
