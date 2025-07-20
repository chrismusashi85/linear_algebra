# linear_algebra
Explain what is linear algebra

# 説明
csv_matrix_reader.pyは、
sample_matrix.csv内の
１．調味料ベクタ（どの調味料をどれだけの割合で混ぜるかを表す。csvの行列＝1,5から縦に3個（列ベクタ）と
２．「調味料→味空間 変換行列」（調味料を混ぜる割合（＝調味料ベクタ）をこの行列にかけると（内積）、どんな味になるか（甘い、しょっぱい、酸っぱい）に変換してくれる行列。csvの行列＝1,1から横に3個、縦に3個の3x3行列）
を読みだして内積を計算する。（変換行列に後ろから調味料ベクタ（列ベクタ）をかける）

調味料ベクタと変換後の味空間上での点を3Dベクタで表示する。

# 準備
使っているライブラリのバージョンはrequirements.txtに書かれている。
必要ならば実行前に
pip install -r requirements.txt
を実行する。

# 実行方法
sample_matrix.csvの調味料ベクタや変換行列を適当に修正した後、
python csv_matrix_reader.py



