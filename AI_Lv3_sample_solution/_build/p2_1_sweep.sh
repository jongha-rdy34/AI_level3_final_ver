cd /c/Users/SSAFY/Desktop/AI_Lv3_sample_solution
PY=.venv/Scripts/python
for cfg in "64 0.0003" "64 0.0001" "32 0.0001" "32 0.0002"; do
  set -- $cfg
  HP=$(printf '# 수정 가능\nbatch_size = %s\nlearning_rate = %s\nepochs = 20' $1 $2)
  JS=$($PY -c "import json,sys;print(json.dumps({'# [정답안] 비교한 설정': sys.argv[1]}))" "$HP")
  T=$($PY _build/run_variant.py P2_1_answer.ipynb _build/sweep_$1_$2.ipynb "$JS")
  $PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=lv3venv "$T" >/dev/null 2>&1
  mv "$T" _build/sweep_$1_$2.ipynb
  echo "done $1 $2"
done
$PY _build/collect_sweep.py
