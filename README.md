# Wholebody Video Tracking
動画1〜2入力から身体・手・足のキーポイントを推論し、相対3Dと人物全体の移動を別々に推定するブラウザアプリです。任意のVRM 0/1に動きを反映します。元映像、VRM、姿勢記録は端末内で扱います。
## 実行
公開アプリ: https://wholebody-video-20261102-b7d2.daisukedaisuke389137.chatgpt.site
Node.js 20以降で同じアプリをローカル実行できます。npmインストールは不要です。
```sh
node bridge/server.mjs
```
表示された `http://127.0.0.1:8765/` をChrome/Edge等で開きます。初回は固定バージョンの公開JSとHugging Faceモデルを取得するので、完全オフラインではありません。モデルは取得後にブラウザのCache Storageへ保存します。
## 操作
MOV/MP4/WebMを選び、「トラッキング開始」を押します。分割TARは `.001` から最後のファイルまで一度に選択してください。TAR内の2本の映像を両スロットに読み込めます。動画・分割ファイルをサーバーへアップロードする処理はありません。
2視点は同時撮影された別角度の映像を使います。開始時刻が違う場合は「視点2の時刻補正」で調整します。視点2の時刻 = 視点1の時刻 + 補正秒です。両方のピクセルを先に保存してから順番に推論するため、1台目の推論を待つ間に2台目だけ別時刻へ進むことを防ぎます。
人物の自動追跡が不適切な場合は「範囲を指定」で映像上の人物を囲みます。MediaPipeは内部で人体検出を行います。全身が見切れる場合には、人体検出器を必須にしないDWPose/RTMPoseの全身2D経路も利用できます。
「VRMを選択」で任意のVRMを読み込みます。アバターの身長を本人の実寸として距離計算へ流用しません。ドラッグ・ホイールで3D表示を操作できます。推定対象を見失った場合、古い姿勢を新しい認識成功として表示しません。
## モデルと実行方式
| 構成 | 推定内容 | ブラウザ実行 |
|---|---|---|
| GHUM Full + Hands | 身体33点・手21点×2、骨盤相対3D | MediaPipe Tasks GPU / WebGL |
| GHUM Lite | 省電力比較用の身体33点3D | MediaPipe Tasks GPU / WebGL |
| DWPose-m / RTMPose-s | COCO-WholeBody 133点の2D | ONNX Runtime WebGPU |
| Hybrid | 身体3Dと必要時の全身2D再取得 | 上記の部位別組み合わせ |
既存の768枚学習モデルを再学習したものではありません。各モデルの元の学習済み重みを保持し、モデル本体は公開Hugging Faceから取得します。元のデータセットを768枚だけに切り詰めた学習を、この実装では行いません。
HF: https://huggingface.co/RabbitDaisuke/Wholebody-Video-Tracking
固定manifest: https://huggingface.co/RabbitDaisuke/Wholebody-Video-Tracking/resolve/48774ee3dc0612f5de170c7a6fbd1a86f017a3b7/manifest.json
モデルURLはimmutable asset commitへ固定し、サイズとSHA-256を照合します。コードのUnlicenseとモデル・外部ライブラリのライセンスは別です。モデルのApache-2.0ライセンスと取得元をHF内に掲載しています。
## 奥行きの意味
`relativeXYZm` / `relativeZm` は骨盤相対の学習済み3D推定です。`root.cameraXYZm` は元動画の2D画素と相対3Dをロバスト最小二乗で合わせたカメラ並進、`root.worldXYZm` は設定した世界座標への変換です。BBox高さだけで距離を決めたり、固定の5.14388mを現在距離として使ったりしません。
校正なしでは距離を `estimatedUncalibrated` と表示します。身長・K・R・tが適切に設定された場合も単眼3Dの曖昧さは残ります。校正済み2カメラで同期・再投影誤差・視線交角を通過した点だけを `measuredMultiView` と呼びます。入力したK/R/tが推測値なら必ず `calibrated:false` にしてください。
「現在の2視点を推定整列」はGHUMの共通関節から推定した相対カメラ姿勢を使います。これは `estimatedMultiView` であり、実測校正の代わりではありません。共通する関節が少ない、同期が悪い、推定が退化している場合は採用しません。
2Dしか観測できない単眼フレームから、根拠のない3Dを生成しません。保持・幾何補完・単眼推定・三角測量・欠測は、出力のsourceとdiagnosticsで区別します。
## 計算負荷と記録
古いフレームを待ち行列に溜めず、最新の映像を推論します。推論上限と計算時間の予算で実際の推論間隔を制御します。計算時間の予算はGPU使用率の実測値ではありません。指推論の頻度と3D描画の頻度も抑えています。GPUが使えないときにCPUへ黙って切り替えません。
推論fpsは実際に推論した回数から求め、動画の再生fpsと分けます。p95推論時間、部位別の観測点数、人体未検出、同期差を画面で確認できます。「座標ログを記録」を選ぶと現在の推定を記録し、「結果を書き出す」でJSONを保存します。
検出率・可視点数・骨格の滑らかさだけを精度の合格条件にはしません。学習モデルは、見切れた人体や強い短縮投影、足裏がカメラへ向いた姿勢で誤推定することがあります。複数モデルが同じ誤りをする可能性もあり、モデル間合意は正解アノテーションを代替しません。対象GPUでの負荷と正解座標に対する3D精度は別に測定する必要があります。
## 検証と再現
`.github/workflows/video-tracking.yml` は各ジョブ10〜60分上限で実行します。公開の公式画像でモデルを実推論し、出力形状と有限値を確認してからHFへモデルのみを公開します。JS/Python構文、公開CDNの取得、相対importとimportmapも確認します。クラウドブラウザは使用しません。
`scripts/evaluate_video.py` は私有動画をローカルで推論し、欠測率・部位別可視点・相対Z・根位置・再投影誤差を分けて集計します。元動画やフレームを外部へアップロードしません。姿勢ラベルを伴わない診断なので、MPJPEや足先の正解率を測ったことにはしません。
```sh
python scripts/evaluate_video.py --input YOUR_VIDEO.mp4 --sample-hz 2 --max-seconds 300 --hands --output artifacts/evaluation.json
```
## SteamVR
腰と左右足のVMT送出には `bridge/server.mjs` を使います。接続トークンを画面へ入力すると、ブラウザの姿勢JSONをlocalhostのVMT/OSCへ送れます。通信停止・欠測ではトラッカーを無効化します。設定と座標契約は [bridge/README.md](bridge/README.md) を参照してください。SteamVR/VMTの導入、VRChatのFBT校正、実機での一致確認はこのクラウド環境では実施していません。
