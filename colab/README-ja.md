# Colab GPU 学習キット / 7fe4fac1

入力ZIP: `VRC_Compact59_Colab_7fe4fac1.zip`

完了時の返却ZIP: **`VRC_Compact59_Colab_Result_7fe4fac1.zip`**

未完了時の返却ZIP: **`VRC_Compact59_Colab_Diagnostic_7fe4fac1.zip`**

## 操作

1. `VRC_Compact59_GPU_7fe4fac1.ipynb` をColabで開く。
2. 「ランタイム」→「ランタイムのタイプを変更」でGPUを選択する。
3. 「すべてのセルを実行」を選び、最初のファイル選択で入力ZIPをアップロードする。
4. 学習セルがダウンロードする返却ZIPを、元のChatGPTチャットへその名前のまま添付する。

Google Driveのマウント、GitHubログイン、HF_TOKEN入力は不要。GPUが割り当てられなければ、CPUへ黙って切り替えずエラーを記録する。失敗時は、原因と保存済みモデルを別名のDiagnostic ZIPへ回収してからセルをエラーにする。Resultという名前では返さない。画面には実行状態と実際に含まれるONNXの一覧を表示する。

依存関係は `.python-packages` にインストールし、学習用の子プロセスにだけ適用する。ColabのCUDA版PyTorchとノートブック本体のパッケージは置き換えない。`venv` / `ensurepip` への依存はない。

## 何を実行するか

公開COCO-WholeBodyの768画像を学習、128画像を候補選択、192画像を最後の評価に使用する。過去の標準59点実験と失敗した軽量化実験の画像IDは、新しいvalidation/testから除外する。testはcheckpoint選択に使わない。

元の学習済み59点headを保持し、144×192と168×224入力の2候補をCUDA GPUで4000 stepずつ学習する。最初の400 stepはhead側を調整し、その後backboneも微調整する。出力は384/512 binのSimCCを保持し、`decodeWH=[192,256]` を使って元画像座標へ戻す。

標準設定は `BATCH_SIZE=32`、`MICRO_BATCH_SIZE=8`。1回の更新に32枚を使い、GPUには8枚ずつ載せて4回の勾配を蓄積する。特徴抽出・候補評価も同じ上限で実行する。BATCH_SIZEは2〜256、MICRO_BATCH_SIZEは1〜16に制限し、開始前に実効バッチと蓄積回数を明示する。2048枚を一度にGPUへ転送する旧処理は使わない。

初期化直後、1 step目、100 stepごとに、実際の重みとONNXを `fast59/recovery/` へ原子的に保存する。これらは途中結果であり、`production_eligible=false`。返却された重みから再微調整はできるが、optimizer状態は含まないため完全な途中再開とは扱わない。

続いて、標準59点をINT8へ量子化する2候補（全Conv、backboneのConvのみ）を作る。校正には同じ学習集合の128枚だけを使い、validation/testの画像は使わない。分割IDをGPU学習の前に確定・保存するので、GPU学習が失敗してもINT8実験は独立して実行する。元の解像度を保って、重みとCPU負荷を減らせるか比較する。

教師推論・学習・候補のPyTorch推論はGPUを使用する。ONNX書き出し後の数値照合・CPU推論時間・INT8校正はCPUで行い、別に記録する。ColabのGPU時間・CPU時間を、利用端末のWASM性能やActionsのNode性能とは扱わない。精度基準未達の候補も失敗結果として返すが、標準モデルを自動で置き換えない。最終testが基準未達なら、候補を選び直さず標準を保持する。

## 同梱物と返却物

入力ZIPにはソース、既存の学習済みCompact59、過去の分割ID、SHA-256の一覧を入れる。公開画像と教師の元checkpointは実行時に固定ハッシュ付きで取得する。学習画像のキャッシュはColabの作業領域だけに置く。

返却ZIPには選択されたONNX、再学習用state_dict、精度・速度・環境情報、分割ID、学習曲線とハッシュを入れる。画像、動画、切り抜き、特徴キャッシュ、フレームごとの座標列は入れない。`IMG_0122_cropped.*` はこの学習の入力ではなく、別途非公開で扱う。

Diagnostic ZIPには、その時点までに実在する途中モデル・独立INT8結果・具体的な例外情報を入れる。チェックポイントが生成される前の失敗ではONNXが無いことを明示する。削除済みランタイムから未保存の重みを復元する機能はない。任意のファイルを一括収集せず、固定したファイル一覧だけを返す。

返却後、こちらでZIPを取得して内容とハッシュを確認し、GitHub ActionsからHF_TOKENでHugging Faceへ公開する。公開APIから認証なしで再取得し、ヘッドレスNode.jsの推論・VRM安定性を測定したうえで、専用Sitesへ反映する。

## 出典・条件

初期モデル: DWPose-t / MMPose（Apache-2.0の実装）。公開COCO-WholeBody注釈は研究・非商用条件を持ち、画像には元の各ライセンスがある。詳細は `training/sources.json` とbaselineの評価JSONを参照。

Colab公式: https://research.google.com/colaboratory/faq.html?hl=ja

ランタイム情報: https://research.google.com/colaboratory/runtime-version-faq.html
