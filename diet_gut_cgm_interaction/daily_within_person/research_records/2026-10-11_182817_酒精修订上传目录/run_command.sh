conda activate pheno &&
cd /home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/daily_within_person/alcohol_revision &&
python -B -c 'import json; v=json.load(open("protocol.json"))["version"]; print("REVISION_VERSION="+v); assert v=="2026-10-11.alcohol.1"' &&
PYTHON=python bash run_revision.sh \
  --reference-run ../outputs/daily_20261011T055621_479640Z_666d40 \
  --data-root /home/ec2-user/Desktop/HPP/Data
