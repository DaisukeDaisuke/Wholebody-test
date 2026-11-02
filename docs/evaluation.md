# 実画像評価の範囲

GitHub Actionsの`Actual-image pose evaluation`で公開画像を実際にONNXへ入力し、正解注釈との2D位置誤差を測る。ローカルやクラウドブラウザの手動テストは行わない。

初期セットはCOCO-WholeBody作者が公開した`evaluation/example_gt.json`と、対応する4枚のCOCO検証画像である。画像IDは785、40083、196141、197388。人物ID順で最大12人を選ぶ。小規模な実画像比較であり、論文の全検証セットAPや実利用一般の精度とは異なる。`workflow_dispatch`の`dataset=val-subset`と`people`で正式検証注釈から対象を増やせる。`val-subset`では初期4画像を除外し、人物IDのSHA-256順でサンプルを固定する。モデルの学習や閾値調整には使用しない。

## 比較

RTMPose-S FP32、RTMW-L FP32の全身推論を同じ人物ROIで比較する。さらにOpenCV ZooのMediaPipe手専用ONNXを追加し、同じ画像の手位置が改善するか悪化するかを測る。手の回転ROIは全身モデルの予測した掌7点だけから作る。正解の手位置・手BBoxは手推論へ入力しない。正解の人物BBoxは利用するため、人物検出器の性能はこの評価に含まれない。

元画像と、手または足を灰色矩形で隠した画像を比較する。矩形実験は既知の正解を保持した静止画の入力劣化実験であり、現実の布・人体による遮蔽や時系列追跡の精度とは別の結果として扱う。

## 指標

Body、feet、handsを別々に集計し、平均・中央値・p95画素誤差、人物BBox長辺の5%と10%を閾値とするPCK、手BBox長辺の20%を閾値とするHand PCKを出す。正解注釈が有効であれば、モデルが低信頼でも位置誤差の分母から外さない。信頼度0.25以上の被覆率も独立表示する。Bodyのみ、公式の可視`v=2`と遮蔽`v=1`を分ける。手足の`v`には小数の信頼値が認められているため、整数1/2を遮蔽の定義として流用しない。

モデルURL、revision、SHA-256、実ONNX入力・出力名とshape、選択した人物ID、画像SHA-256、推論バックエンドを`report.json`へ保存する。予測値JSONと、正解・全身モデル・手refinementを重ねた証拠画像も保存する。モデルの巨大なバイナリをActions成果物へ再アップロードしない。

## 3Dと実機の境界

`verify-geometry.mjs`は、異なるカメラ中心・校正・時刻を使った三角測量の数学契約と、予測の失効を検査する。これは既知の合成3D入力による実装検証であり、実カメラでのセンチメートル精度の証拠ではない。実機の2視点動画、本人骨長、撮影位置の正解、LAN端末間通信、RTXのWebGPU速度は、それぞれの実入力があるときに評価する。

`--panoptic-input`を指定した場合は、CMU Panopticの実画像推論から得た2視点観測と既知カメラ校正を、アプリと同じ`dist/geometry.js`の`triangulateJoint`と`dist/tracking.js`の`OcclusionTracker`へ入力する。正解3D座標は評価にだけ使用し、公開レポートには残さない。身体17点と左右の手42点について、利用可能な推定のMPJPE、推定が存在する割合、20/50/100mm以内のPCKを測る。PCKの分母は欠測を含む全有効正解点とし、難しい点を失効させただけで精度が良く見える集計を避ける。

中央8フレームで両手の全視点観測を除去する実験は、画像モデルの遮蔽性能とは分ける。これは実動作の3D正解に対し、時系列補完が観測欠測中にどこまで追従できるかを測る。元の観測・観測除去の両方について、直接三角測量と時系列追跡、全フレームと欠測区間の指標を個別に出す。

## 初回4画像での実測

[Actions run 41374788866](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41374788866)は、コミット`935b396ad3684e12edc4f9ceca059215a755a871`で全job成功した。4画像12人、身体181点・足38点・手210点が有効だった。

| 構成 | 身体平均誤差px | 足平均誤差px | 手平均誤差px | Hand PCK(.20) |
|---|---:|---:|---:|---:|
| RTMPose-S | 6.460 | 8.713 | 6.544 | 32.86% |
| RTMPose-S＋手ROI | 6.460 | 8.713 | 6.119 | 40.48% |
| RTMW-L | 4.380 | 5.183 | 4.084 | 64.29% |
| RTMW-L＋手ROI | 4.380 | 5.183 | 3.949 | 66.19% |

手ROIの採用はSで24手候補中2手、Lで4手だった。灰色で隠した手157点のHand PCK(.20)はSで1.91%、Lで12.74%となり、手専用推論の追加では改善しなかった。Lでは隠した手点の信頼スコア0.25以上の割合が100%だったため、そのスコアを「見えている確率」とみなせない。初回セットでの改善を全ての映像へ一般化しない。後続の64人評価はこの4画像を除外する。

## 出典

- [COCO-WholeBody公式](https://github.com/jin-s13/COCO-WholeBody)
- [注釈形式とvisibilityの定義](https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md)
- [公式評価例](https://github.com/jin-s13/COCO-WholeBody/blob/master/evaluation/example_gt.json)
- [OpenCV Zoo MediaPipe Hand](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe)
- [RTMPose公式](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)

COCO画像は各撮影者のライセンス、COCO-WholeBody注釈は作者の研究・非商用条件に従う。評価用画像・注釈は取得して比較するだけで、アプリのUnlicenseの対象として再配布しない。
