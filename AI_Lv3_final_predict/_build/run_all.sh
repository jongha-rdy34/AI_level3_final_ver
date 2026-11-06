cd /c/Users/SSAFY/Desktop/AI_Lv3_final_predict
PY=.venv/Scripts/python
NB="$PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=lv3venv"
mkdir -p _build/baseline
for n in F1_1 F1_2 F2_2; do
  echo "== baseline $n"; T=$($PY _build/run_variant.py ${n}_problem.ipynb _build/baseline/${n}_baseline.ipynb); $NB "$T" >/dev/null 2>&1; echo $?; mv "$T" _build/baseline/${n}_baseline.ipynb
done
cp _build/test_blank.csv test.csv
for n in F1_1 F1_2 F2_1 F2_2 F3; do
  echo "== answer $n"; $NB ${n}_answer.ipynb >/dev/null 2>&1; echo $?
done
echo ALLDONE
