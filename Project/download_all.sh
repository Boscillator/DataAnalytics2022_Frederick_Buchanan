rm -r data/processed

for month in {1..12}
do
    python -m FirePrediction.generate --year 2020 --month $month --n 40 --seed $month --size 64
done

python -m FirePrediction.enrich
python -m FirePrediction.test_train_split
