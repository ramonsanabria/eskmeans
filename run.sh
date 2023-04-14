source /disk/scratch1/ramons/myenvs/p3_herman/bin/activate
export KALDI_ROOT=/disk/scratch1/ramons/myapps/kaldi/

language=mandarin

cat ./${language}_spk | parallel python run.py --speaker {} --language ${language} --feature_type hubert_base_ls960 --pooling_type average
