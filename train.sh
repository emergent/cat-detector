# 学習用のデータの特徴量をとり、SVMで学習する
python lbp.py positive negative ser_lbp
python learn_svm.py ser_lbp ser_svm

# テスト用のデータの特徴量を取る
python lbp.py positive_test negative_test ser_lbp_test

# 学習済みモデルの精度を測定する
python test_svm.py ser_svm ser_lbp_test
