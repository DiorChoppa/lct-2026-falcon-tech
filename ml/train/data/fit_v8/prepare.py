"""Stage exactly two reviewed bbox exclusions; never activate or edit a trainer."""
import csv, hashlib, io, json, math
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIELDS=('image_id','x','y','w','h','vehicle_id','camera_id')
EXCLUDED={8009:('2223367bff1d4237a34212ee3df0f0df','728','627','753','453','967','90'),
          8012:('8f28abd7a6104e9d84068eb01ea04363','728','625','753','455','967','90')}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def rows(path):return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
def key(row):return tuple(row[field] for field in FIELDS)
def save(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def derive(before):return [row for row in before if key(row) not in set(EXCLUDED.values())]

parent=HERE.parent/'v7/train.csv'
sourcepath=ROOT/'Data/dataset/train.csv'
original_fit=ROOT/'artifacts/episodes/v1/train.csv'
evidence=ROOT/'artifacts/fit_eda_round7'
fixed={parent:'75425adee5a118e3853f47cef8a53572d4079be93a7e2a41b6771f9d750ea4c9',
 parent.with_name('manifest.json'):'f11e5c1fcb9ef7cbd62fb00f2985c90f5ff26c02ae233fd5d216d290dc71e111',
 sourcepath:'bd1df45b052ae9aabb7fd356898e244875f8cad2f111a3f76977c1b90bce9268',
 evidence/'selection.json':'8b3aa85d2deb07c1e4dc4c5d9d24910ffa0efa243bc9d7236deadc4b7047badb',
 evidence/'final_review.json':'d8bb7781ace8ffd4b59eaca3639391a65372a9ed769c0318275af3d95da940d3',
 evidence/'root_visual_review.json':'04f6ae9c11ba28293cf9fb1b9025364a5f52b352507f9bec19a5bc0f0b9c45ba'}
assert not (HERE/'train.csv').exists() and not (HERE/'manifest.json').exists()
assert all(sha(path)==expected for path,expected in fixed.items())
root_review=json.loads((evidence/'root_visual_review.json').read_text())
assert set(root_review['decision'])=={'8009','8012','8010'}
assert all(root_review['decision'][str(n)].startswith('confirmed displaced') for n in EXCLUDED)
assert root_review['decision']['8010'].startswith('retain correct')
for relative,expected in root_review['files'].items():
    path=evidence/relative
    assert sha(path)==expected
    fixed[path]=expected
for path in [Path(__file__),original_fit,*[p for v in range(2,8) for p in (HERE.parent/f'v{v}').glob('*.json')]]:
    fixed[path]=sha(path)
source,before,initial=rows(sourcepath),rows(parent),rows(original_fit)
assert tuple(before[0])==FIELDS
assert len(before)==5724 and len(initial)==5746
assert len({key(r) for r in before})==len(before)
for number,excluded in EXCLUDED.items():
    assert key(source[number-2])==excluded
    assert sum(key(r)==excluded for r in before)==1
after=derive(before)
assert len(after)==5722 and len(before)-len(after)==2
assert Counter(map(key,before))-Counter(map(key,after))==Counter(EXCLUDED.values())
assert not Counter(map(key,after))-Counter(map(key,before))
assert {r['vehicle_id'] for r in after}=={r['vehicle_id'] for r in before} and len({r['vehicle_id'] for r in after})==928
assert {r['camera_id'] for r in after}=={r['camera_id'] for r in before} and len({r['camera_id'] for r in after})==96
remaining=[r for r in after if r['vehicle_id']=='967']
assert len(remaining)==4 and Counter(r['camera_id'] for r in remaining)=={'89':3,'90':1}
assert source[8010-2] in after
assert [r for r in after if r['vehicle_id']!='967']==[r for r in before if r['vehicle_id']!='967']
# Exact full-row matching must not discard another identity that shares an image key.
foreign=dict(zip(FIELDS,EXCLUDED[8009]));foreign['vehicle_id']='foreign_test_only'
assert derive([foreign])==[foreign]
initial_keys=set(map(key,initial));before_keys=set(map(key,before));after_keys=set(map(key,after))
assert after_keys<=before_keys<=initial_keys
inherited=initial_keys-before_keys
assert len(inherited)==22 and len(initial_keys-after_keys)==24 and not inherited & after_keys
lines=parent.read_bytes().splitlines(keepends=True)
assert len(lines)==len(before)+1
(HERE/'train.csv').write_bytes(lines[0]+b''.join(line for row,line in zip(before,lines[1:],strict=True) if key(row) not in EXCLUDED.values()))
assert rows(HERE/'train.csv')==after
assert all(sha(path)==expected for path,expected in fixed.items())
exclusions=[]
for number,excluded in EXCLUDED.items():
    context=evidence/f'contexts/source_row_{number}.jpg'
    exclusions.append({'row':dict(zip(FIELDS,excluded)),'source_csv_line_including_header':number,
      'parent_fit_csv_line_including_header':next(i+2 for i,r in enumerate(before) if key(r)==excluded),
      'context':context.relative_to(ROOT).as_posix(),'context_sha256':sha(context)})
manifest={'version':8,'status':'STAGED_PENDING_INDEPENDENT_REVIEW_NOT_ACTIVE',
 'parent_csv_sha256':sha(parent),'parent_manifest_sha256':sha(parent.with_name('manifest.json')),
 'source_csv_sha256':sha(sourcepath),'csv_sha256':sha(HERE/'train.csv'),
 'exclusion_key_fields':FIELDS,'exclusions':exclusions,
 'evidence_pins':{p.relative_to(ROOT).as_posix():v for p,v in fixed.items()},
 'before':{'images':5724,'identities':928,'cameras':96},'after':{'images':5722,'identities':928,'cameras':96},
 'affected_identities_remaining':{'967':{'rows':4,'cameras':{'89':3,'90':1}}},
 'retained_hard_or_ambiguous_source_rows':[8010],
 'inherited_original_FIT_exclusions':22,'cumulative_original_FIT_exclusions':24,
 'inherited_quarantines':'Exact subset of v7; every prior omission remains absent, including pre-FIT quarantines. No role assignments, source rows or labels changed.',
 'batch32_steps_per_epoch':math.ceil(len(after)/32),'e12_updates':12*math.ceil(len(after)/32),
 'reason':'Two independently viewed displaced target bboxes. No plate basis, no invented replacement bbox, no relabeling.',
 'activation':'None: campaign, STATE and trainer unchanged. Existing checkpoints remain FIT-v7.',
 'split_roles':'Only delete two observations within FIT; retain exact identity/camera sets. No protected-role files opened.'}
save(HERE/'manifest.json',manifest)
checks={'status':'passed','ordered_exact_two_row_diff':True,'source_and_parent_hashes_unchanged':True,
 'retained_control_8010':True,'foreign_identity_rows_unchanged':True,'synthetic_foreign_identity_exact_key_check':True,
 'all_prior_exclusions_inherited':True,'role_reassignment':False,'protected_roles_opened':False,
 'csv_sha256':sha(HERE/'train.csv'),'manifest_sha256':sha(HERE/'manifest.json'),'prepare_sha256':sha(Path(__file__))}
save(HERE/'checks.json',checks)
print(json.dumps(checks))
