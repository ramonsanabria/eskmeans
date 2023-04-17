if [ $(hostname) = "banff.inf.ed.ac.uk" ]
then
    source /disk/scratch_fast/ramons/myenvs/p3_herman/bin/activate
else
    source /disk/scratch1/ramons/myenvs/p3_herman/bin/activate
fi

language=buckeye

cat data/file_list/buckeye_spk | parallel  python run.py --speaker {} --language ${language} --feature_type hubert_base_ls960  --pooling_type average --kmeans_type em
