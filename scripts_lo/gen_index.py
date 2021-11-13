import os
import sys
import argh
import numpy as np
import ujson
import pickle
from pathlib import Path
sys.path.insert(0, '../../colbert')

from colbert.infra import Run, RunConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from rich.console import Console

console = Console(soft_wrap=True)

def get_int_id_filepath(filepath, cache_dir):
    # Load index from cache dir
    filepath = Path(filepath)
    all_data = [line.strip().split("\t") for line in open(filepath)]
    rowidx2origidx = {i: line[0] for i, line in enumerate(all_data)}
    new_data = [[str(i), line[1]] for i, line in enumerate(all_data)]
    print(filepath.stem)
    outfile_path = cache_dir / f"{filepath.stem}_adj.tsv"
    rowidx2origidx_path = cache_dir / f"{filepath.stem}_idxmapping.json"
    ujson.dump(rowidx2origidx, open(rowidx2origidx_path, "w"))
    with open(outfile_path, "w") as out_f:
        for line in new_data:
            out_f.write("\t".join(line) + "\n")
    console.print(f"Index Mapping Path {rowidx2origidx_path}")
    return str(outfile_path), rowidx2origidx



@argh.arg("--dataset_name", help="Name to give to experiment/index")
@argh.arg("--log_dir", help="Log directory")
@argh.arg("--cache_dir", help="Cache directory")
@argh.arg("--idxmapping_path", help="Path to json from row pid to orig pid. If none, will be created.")
@argh.arg("--query_file", help="Path to query file in tsv")
@argh.arg("--passage_file", help="Path to passage file in tsv")
@argh.arg("--index_file", help="Path to saved index file")
@argh.arg("--colbert_checkpoint", help="Path to colbert checkpoint")
@argh.arg("--num_gpus", type=int, help="Num gpus")
@argh.arg("--topk", type=int, help="Batch size")
def main(
    dataset_name="amber",
    log_dir="log_dir",
    cache_dir="_colbert_cache",
    idxmapping_path=None,
    query_file=None,
    passage_file=None,
    index_file=None,
    colbert_checkpoint="/dfs/scratch0/okhattab/share/2021/checkpoints/msmarco.psg.kldR2.nway64.ib__colbert-400000/",
    num_gpus=1,
    topk=100,
):
    log_dir = Path(log_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if passage_file:
        passage_file, rowidx2origidx = get_int_id_filepath(passage_file, cache_dir)
        collection = Collection(path=passage_file)
        console.print(f"Loaded {len(collection)} passages to index")
        index_name = f'{dataset_name}.index'

        with Run().context(RunConfig(nranks=num_gpus, experiment=dataset_name)):  # nranks specifies the number of GPUs to use.
            indexer = Indexer(checkpoint=colbert_checkpoint)  # MS MARCO ColBERT checkpoint
            indexer.index(name=index_name, collection=collection, overwrite=True)
            index_file = indexer.get_index()
            console.print(f"Index File {index_file}")
    else:
        if idxmapping_path is None:
            raise ValueError("If loading an existing index, must pass in idxmapping_path.")
        rowidx2origidx = ujson.load(open(idxmapping_path))

    if query_file:
        assert index_file is not None, f"Must provide index file if pass in query file"
        searcher = Searcher(index=index_file)

        queries = Queries(path=query_file)
        console.print(f"Loaded {len(queries)} queries")
        rankings = searcher.search_all(queries, k=topk).todict()
        rankings_with_origid = {qid: {"orig_id": rowidx2origidx[pid], "pid": pid, "rank": rank, "score": score}
                                for qid in rankings for pid, rank, score in rankings[qid]}
        log_rankings_file = log_dir / dataset_name / "rankings.pkl"
        log_rankings_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(rankings_with_origid, open(str(log_rankings_file), "wb"))


if __name__ == "__main__":
    argh.dispatch_command(main)
