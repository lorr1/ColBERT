import os
import sys
import argh
import numpy as np
import faiss
import pickle
from pathlib import Path
sys.path.insert(0, '../../colbert')

from colbert.infra import Run, RunConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from rich.console import Console

console = Console(soft_wrap=True)

@argh.arg("--dataset_name", help="Name to give to experiment/index")
@argh.arg("--log_dir", help="Log directory")
@argh.arg("--query_file", help="Path to query file in tsv")
@argh.arg("--passage_file", help="Path to passage file in tsv")
@argh.arg("--index_file", help="Path to saved index file")
@argh.arg("--colbert_checkpoint", help="Path to colbert checkpoint")
@argh.arg("--num_gpus", type=int, help="Num gpus")
@argh.arg("--batch_size", type=int, help="Batch size")
def main(
    dataset_name="amber",
    log_dir="log_dir",
    query_file=None,
    passage_file=None,
    index_file=None,
    colbert_checkpoint="/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000/",
    num_gpus=1,
    batch_size=32,
):
    log_dir = Path(log_dir)
    if passage_file:
        collection = Collection(path=passage_file)
        console.print(f"Loaded {len(collection)} passages to index")
        index_name = f'{dataset_name}.index'

        with Run().context(RunConfig(nranks=num_gpus, experiment=dataset_name)):  # nranks specifies the number of GPUs to use.
            indexer = Indexer(checkpoint=colbert_checkpoint)  # MS MARCO ColBERT checkpoint
            indexer.index(name=index_name, collection=collection, overwrite=True)
            index_file = indexer.get_index()
            print("INDEX FILE", index_file)

    if query_file:
        assert index_file is not None, f"Must provide index file if pass in query file"
        searcher = Searcher(index=index_file)

        queries = Queries(path=query_file)
        console.print(f"Loaded {len(queries)} queries")
        rankings = searcher.search_all(queries, k=5).todict()
        log_rankings_file = log_dir / dataset_name / "rankings.pkl"
        log_rankings_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(rankings, open(str(log_rankings_file), "wb"))


if __name__ == "__main__":
    argh.dispatch_command(main)
