cd /c/Users/SSAFY/Desktop/AI_Lv3_sample_solution
PY=.venv/Scripts/python
NB="$PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=lv3venv"
echo "== P1_2 answer"; $NB P1_2_answer.ipynb >/dev/null 2>&1; echo $?
echo "== baseline P1_2"; T=$($PY _build/run_variant.py P1_2_problem.ipynb _build/baseline/P1_2_baseline.ipynb); $NB "$T" >/dev/null 2>&1; echo $?; mv "$T" _build/baseline/P1_2_baseline.ipynb
echo "== P2_1 sweep"; bash _build/p2_1_sweep.sh
cp P1_1_answer.ipynb P1_2_answer.ipynb _build/baseline/ && $PY _build/build_answers.py && cp _build/baseline/P1_1_answer.ipynb _build/baseline/P1_2_answer.ipynb . && rm _build/baseline/P1_?_answer.ipynb
echo "== P2_1 answer"; $NB P2_1_answer.ipynb >/dev/null 2>&1; echo $?
echo "== P2_2 answer"; $NB P2_2_answer.ipynb >/dev/null 2>&1; echo $?
echo "== P3 answer"; $NB P3_answer.ipynb >/dev/null 2>&1; echo $?
echo ALLDONE
