import json
from pathlib import Path
import pytest

from autoslm_v3.artifacts import write_adapter_manifest, verify_adapter_manifest
from autoslm_v3.benchmark import load_locked_benchmark, evaluate_callable, compare_reports
from autoslm_v3.cpt import mix_domain_and_replay, load_cpt_corpus
from autoslm_v3.preference import validate_preference_pairs


def test_adapter_manifest_detects_tampering(tmp_path):
    root=tmp_path/"adapter"; root.mkdir()
    (root/"adapter_config.json").write_text('{"r":8}',encoding="utf-8")
    write_adapter_manifest(str(root),{"base":"tiny"})
    assert verify_adapter_manifest(str(root))["valid"]
    (root/"adapter_config.json").write_text('{"r":16}',encoding="utf-8")
    result=verify_adapter_manifest(str(root))
    assert not result["valid"] and result["changed"]==["adapter_config.json"]


def _benchmark(path):
    rows=[{"id":str(i),"task":"qa" if i%2 else "math","prompt":f"q{i}",
           "reference":f"a{i}","source":"human-reviewed-set","license":"internal-consent",
           "split":"test"} for i in range(10)]
    path.write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf-8")


def test_benchmark_requires_minimum_and_explicit_provenance(tmp_path):
    path=tmp_path/"benchmark.jsonl"; _benchmark(path)
    assert len(load_locked_benchmark(str(path)))==10
    rows=path.read_text().splitlines()
    item=json.loads(rows[0]); item.pop("license"); rows[0]=json.dumps(item)
    path.write_text("\n".join(rows),encoding="utf-8")
    with pytest.raises(ValueError,match="license"):
        load_locked_benchmark(str(path))


def test_base_candidate_must_use_same_locked_benchmark(tmp_path):
    path=tmp_path/"benchmark.jsonl"; _benchmark(path)
    base=evaluate_callable("base",lambda prompt:"wrong",str(path),str(tmp_path/"base.jsonl"))
    candidate=evaluate_callable("candidate",lambda prompt:prompt.replace("q","a"),str(path),str(tmp_path/"candidate.jsonl"))
    report=compare_reports(base,candidate)
    assert report["exact_match_delta"]==1.0
    altered=tmp_path/"other.jsonl"; _benchmark(altered)
    rows=altered.read_text().splitlines(); item=json.loads(rows[0]); item["reference"]="different"; rows[0]=json.dumps(item)
    altered.write_text("\n".join(rows),encoding="utf-8")
    other=evaluate_callable("other",lambda prompt:"x",str(altered),str(tmp_path/"other-predictions.jsonl"))
    with pytest.raises(ValueError,match="identical locked benchmark"):
        compare_reports(base,other)


def test_cpt_replay_mix_is_deterministic_and_weighted():
    domain=[{"text":"domain","source":"d1"} for _ in range(8)]
    replay=[{"text":"general","source":"r1"}]
    a=mix_domain_and_replay(domain,replay,0.25,seed=7)
    b=mix_domain_and_replay(domain,replay,0.25,seed=7)
    assert a==b
    assert sum(x["corpus_role"]=="general_replay" for x in a)==3


def test_cpt_corpus_fails_closed_on_unapproved_license(tmp_path):
    path=tmp_path/"corpus.jsonl"
    path.write_text(json.dumps({"text":"valid body","source":"doc-1","license":"unknown"})+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="allowed license"):
        load_cpt_corpus(str(path),["cc-by-4.0"])


def test_preference_pairs_require_provenance_and_distinct_answers():
    good={"prompt":"question","chosen":"good","rejected":"bad","source":"reviewer","license":"internal-consent"}
    assert len(validate_preference_pairs([good,good]))==1
    with pytest.raises(ValueError,match="identical"):
        validate_preference_pairs([{**good,"rejected":"good"}])
    with pytest.raises(ValueError,match="provenance"):
        validate_preference_pairs([{k:v for k,v in good.items() if k!="source"}])


def test_new_stage_cli_help_is_cpu_safe():
    import subprocess, sys
    root=Path(__file__).resolve().parents[1]
    for name,expected in (("train_cpt.py","--domain-corpus"),("train_dpo.py","--allowed-licenses"),("evaluate_benchmark.py","--benchmark"),("profile_gpu.py","--load-model")):
        result=subprocess.run([sys.executable,str(root/"scripts"/name),"--help"],cwd=root,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert expected in result.stdout

 
def test_cpt_chunker_keeps_short_final_blocks():
    from autoslm_v3.cpt import chunk_texts
    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": list(range(34))}
    chunks=chunk_texts([{"text":"short"}],TinyTokenizer(),block_size=32)
    assert [len(x["input_ids"]) for x in chunks]==[32,2]
    assert chunks[0]["labels"]==chunks[0]["input_ids"]


def test_benchmark_rejects_non_object_and_output_overwrite(tmp_path):
    bad=tmp_path/"bad.jsonl"
    bad.write_text("[]"+"\n"+"\n".join(json.dumps({"id":str(i),"task":"qa","prompt":"p","reference":"r","source":"review","license":"consent","split":"test"}) for i in range(10)),encoding="utf-8")
    with pytest.raises(ValueError,match="JSON object"):
        load_locked_benchmark(str(bad))
    good=tmp_path/"good.jsonl"; _benchmark(good)
    with pytest.raises(ValueError,match="must not overwrite"):
        evaluate_callable("base",lambda x:x,str(good),str(good))

 
def test_sft_split_keeps_nested_source_documents_together():
    from autoslm_v3.data import split_dataset
    rows=[]
    for doc in range(12):
        for variant in range(2):
            rows.append({
                "instruction":f"unique prompt {doc}-{variant}",
                "response":f"response {doc}-{variant}",
                "provenance":{"source_document_id":f"source-{doc}"},
            })
    split=split_dataset(rows,test_fraction=.2,validation_fraction=.2,seed=3)
    partitions=[split.train,split.validation,split.test]
    assignment={}
    for partition_index,partition in enumerate(partitions):
        for row in partition:
            source=row["provenance"]["source_document_id"]
            assert source not in assignment or assignment[source]==partition_index
            assignment[source]=partition_index
    assert len(assignment)==12
