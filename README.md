# Environment

conda env create --name rift --file environment.yml

# Data

- Spacy
python -m spacy download en

mkdir data_set
mkdir temp

- IMDB
cd data_set
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvzf aclImdb_v1.tar.gz
rm -f aclImdb_v1.tar.gz
cd ..

# Run

rift_train_imdb.sh
