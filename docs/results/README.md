# 保存した実測結果

`evaluation-run6.tar.gz` はGitHub Actionsで生成されたJSON/Markdownを変更せずまとめたものです。元画像や学習済み重みは含みません。

- アプリ、手、人物検出、未使用32画像、幾何、実録動作: [Actions run 41378750685](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41378750685)
- 対応する推論・評価コード: `7e3bf03021d7b2131ec7fc7b3b4583508b8ad6e9`
- 足の4画像比較・JavaScriptとの数値比較: [Actions run 41377683638](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41377683638)
- 足比較のコード: `781a023ec2b11dabedd87215face73db2e07df88`

展開すると`holdout-report.json`が32画像の全集計と画像別結果、`holdout-selection.json`が推論前に固定した対象と取得ハッシュを含みます。`report.json`は調整に使用した4画像であり、32画像の検証とは分けて読んでください。

`panoptic-report.json`は取得元と試験条件を記録します。このrunではCMUの実録ASF/AMCを使用し、観測ラベルを人工的に欠測させています。実カメラの遮蔽検出や指・足指の時系列精度ではありません。

`foot-report.json`は足専用モデルの比較実験です。4画像で改善しても未使用32画像では回帰したため、最終アプリの足補正は既定OFFです。モデルの生スコアは可視性・正解の確率として校正されていません。

推論経路の閾値や採点対象は32画像の結果を見て変更していません。アプリの実測パネルにも、同じ結果を読み込んで表示します。
