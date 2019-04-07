# 学習用のデータの特徴量をとり、SVMで学習する
# その後、テスト用データで学習済みモデルの精度を測定し、
# 分類器を`pickle`でシリアライズする
python lbp.py positive negative positive_test negative_test ser_svm
