"""Sequential paired navigation experiments; proposals precede source edits."""
import hashlib,json
from pathlib import Path
import numpy as np
from warm_navigation_cases import generate,KINDS
ROOT=Path(__file__).parent
DEST=ROOT/"results/research-warm-hundred/navigation"
DEST.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/"warm_navigation.py"
def evaluate(source,seeds):
    namespace={};exec(compile(source,"archived_navigation","exec"),namespace)
    rows=[];all_errors=[]
    for seed in seeds:
        for kind in KINDS:
            z,truth,attack=generate(seed,kind);result=namespace["estimate"](z)
            e=np.linalg.norm(result["position"][attack:,:2]-truth[attack:,:2],axis=1);valid=np.isfinite(e)
            all_errors.extend(e[valid].tolist())
            gap=longest=0
            for ok in valid:
                gap=0 if ok else gap+1;longest=max(longest,gap)
            initialized=result["initialized_epoch"]
            rows.append(dict(seed=seed,kind=kind,antennas=[2,3,4],paired_identical=True,initialization_epoch=initialized,initialization_time_s=None if initialized is None else float(z["time"][initialized]),attack_time_s=float(z["time"][attack]),warm_gap_s=None if initialized is None else float(z["time"][attack]-z["time"][initialized]),initialization_failed=initialized is None or initialized>=attack,epochs=len(e),outputs=int(valid.sum()),le20=int(np.sum(e<=20)),ge30=int(np.sum(e>=30)),ge50=int(np.sum(e>=50)),p95_m=float(np.percentile(e[valid],95)) if valid.any() else None,longest_gap_s=longest*.5))
    total=sum(r["epochs"] for r in rows);outputs=sum(r["outputs"] for r in rows)
    return dict(rows=rows,summary=dict(epochs=total,outputs=outputs,availability=outputs/total,unavailable=total-outputs,p95_m=float(np.percentile(all_errors,95)) if all_errors else None,conditional_le20=float(np.mean(np.array(all_errors)<=20)) if all_errors else None,le20=sum(r["le20"] for r in rows),ge30=sum(r["ge30"] for r in rows),ge50=sum(r["ge50"] for r in rows),initialization_failures=sum(r["initialization_failed"] for r in rows),longest_gap_s=max(r["longest_gap_s"] for r in rows)))
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
def round(number,hypothesis,marker,insertion,replace=False):
    folder=DEST/f"round-{number:03d}";folder.mkdir(exist_ok=True)
    proposal=dict(round=number,hypothesis=hypothesis,development_seeds=[111001,111013],input_domain="linearized code/carrier observations; no RF",selection="minimize >=50 then >=30; maximize <=20 outputs")
    write(folder/"proposal.json",proposal)
    before=SOURCE.read_text(encoding="utf-8");(folder/"before.py").write_text(before,encoding="utf-8")
    after=before.replace(marker,insertion) if replace else before.replace("        "+marker,insertion+"\n        "+marker)
    assert before!=after
    SOURCE.write_text(after,encoding="utf-8");(folder/"after.py").write_text(after,encoding="utf-8")
    a=evaluate(before,[111001,111013]);b=evaluate(after,[111001,111013]);write(folder/"before.json",a);write(folder/"after.json",b)
    score=lambda r:(r["summary"]["ge50"],r["summary"]["ge30"],-r["summary"]["le20"])
    keep=score(b)<score(a)
    if not keep:SOURCE.write_text(before,encoding="utf-8")
    entry=dict(**proposal,accepted=keep,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),before=a["summary"],after=b["summary"],conclusion="Retain: improves preregistered safety/usable-output ordering" if keep else "Reject: no strict improvement in preregistered safety/usable-output ordering",next_step="Inspect remaining errors and continuity; test a distinct mechanism next")
    write(folder/"conclusion.json",entry)
    with (DEST/"ledger.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(entry)+"\n")
    print(json.dumps(dict(round=number,accepted=keep,before=a["summary"],after=b["summary"])))


def verify():
    ledger=[json.loads(line) for line in (DEST/"ledger.jsonl").read_text().splitlines()]
    assert [row["round"] for row in ledger]==list(range(76,101))
    for entry in ledger:
        folder=DEST/f"round-{entry['round']:03d}"
        for side in ("before","after"):
            source=(folder/f"{side}.py").read_text(encoding="utf-8")
            assert hashlib.sha256(source.encode()).hexdigest()==entry[f"{side}_sha256"]
            repeated=evaluate(source,entry["development_seeds"])
            saved=json.loads((folder/f"{side}.json").read_text())
            assert all(repeated["summary"][key]==value for key,value in saved["summary"].items())
        print("verified",entry["round"],flush=True)
    return len(ledger)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true");parser.add_argument("--holdout",action="store_true");args=parser.parse_args()
    if args.verify:verify()
    if args.holdout:
        frozen=json.loads((DEST/"freeze.json").read_text())
        for name,digest in frozen["sha256"].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        result=evaluate(SOURCE.read_text(encoding="utf-8"),[112001,112013,112037])
        write(DEST/"holdout.json",result);print(json.dumps(result["summary"]))
