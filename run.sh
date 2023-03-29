source /disk/scratch1/ramons/myenvs/p3_herman/bin/activate
export KALDI_ROOT=/disk/scratch1/ramons/myapps/kaldi/

number_centroids=896
language=mandarin
speaker=A08

python run.py --speaker ${speaker} --language ${language} --centroids ${number_centroids} --min_duration 20
