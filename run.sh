if [ $(hostname) = "banff.inf.ed.ac.uk" ]
then
    source /disk/scratch_fast/ramons/myenvs/p3_herman/bin/activate
else
    source /disk/scratch1/ramons/myenvs/p3_herman/bin/activate
fi

number_centroids=5
language=mandarin
speaker=A0

python run.py --speaker ${speaker} --language ${language} --centroids ${number_centroids} --min_duration 20
