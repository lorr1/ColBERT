import os
import sys
import argh
import numpy as np
import faiss
from pathlib import Path
sys.path.insert(0, '/dfs/scratch0/lorr1/projects/colbert')

from colbert.infra import Run, RunConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from rich.console import Console

console = Console(soft_wrap=True)

def test_clustering():
    d = 64
    n = 1000
    rs = np.random.RandomState(123)
    x = rs.uniform(size=(n, d)).astype('float32')

    x *= 10

    km = faiss.Kmeans(d, 32, niter=10)
    err32 = km.train(x)

    # check that objective is decreasing
    prev = 1e50
    for o in km.obj:
        assert prev > o
        prev = o

    km = faiss.Kmeans(d, 64, niter=10)
    err64 = km.train(x)

    # check that 64 centroids give a lower quantization error than 32
    assert err32 > err64

    km = faiss.Kmeans(d, 32, niter=10, int_centroids=True)
    err_int = km.train(x)

    # check that integer centoids are not as good as float ones
    assert err_int > err32
    assert np.all(km.centroids == np.floor(km.centroids))

@argh.arg("--data_path")
@argh.arg("--dataset_name")
@argh.arg("--query_file")
@argh.arg("--passage_file")
@argh.arg("--colbert_checkpoint")
@argh.arg("--num_gpus", type=int)
def main(
    data_path="/dfs/scratch0/lorr1/projects",
    dataset_name="amber",
    query_file="colbert_amber_queries.tsv",
    passage_file="colbert_wikipedia_collection_subset.tsv",
    colbert_checkpoint="/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000/",
    num_gpus=1,
):
    # test_clustering()
    data_path = Path(data_path)
    queries = data_path / query_file
    collection = data_path / passage_file
    console.print(f"Loading queries from {queries} and passages from {collection}")

    queries = Queries(path=str(queries))
    collection = Collection(path=str(collection))
    console.print(f"Loaded {len(queries)} queries and {len(collection)} passages")

    with Run().context(RunConfig(nranks=num_gpus, experiment=dataset_name)):  # nranks specifies the number of GPUs to use.
        index_name = f'{dataset_name}.index'

        indexer = Indexer(checkpoint=colbert_checkpoint)  # MS MARCO ColBERT checkpoint
        indexer.index(name=index_name, collection=collection, overwrite=True)

        searcher = Searcher(index=index_name)

    query = queries[0]   # or supply your own query

    print(f"#> {query}")

    # Find the top-3 passages for this query
    results = searcher.search(query, k=3)

    # Print out the top-k retrieved passages
    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")


if __name__ == "__main__":
    argh.dispatch_command(main)
