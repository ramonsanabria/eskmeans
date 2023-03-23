source /disk/scratch1/ramons/myenvs/p3_herman/bin/activate

number_centroids=5
language=mandarin
speaker=A08

python run.py --speaker ${speaker} --language ${language} --centroids ${number_centroids} --min_duration 20
