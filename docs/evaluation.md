# 実画像評価の範囲

GitHub Actionsの`Actual-image pose evaluation`で公開画像を実際にONNXへ入力し、正解注釈との2D位置誤差を測る。ローカルやクラウドブラウザの手動テストは行わない。

初期セットはCOCO-WholeBody作者が公開した`evaluation/example_gt.json`と、対応する4枚のCOCO検証画像である。画像IDは785、40083、196141、197388。人物ID順で最大12人を選ぶ。小規模な実画像比較であり、論文の全検証セットAPや実利用一般の精度とは異なる。`workflow_dispatch`の`dataset=val-subset`と`people`で正式検証注釈から対象を増やせる。

## 比較

RTMPose-S FP32、RTMW-L FP32の全身推論を同じ人物ROIで比較する。さらにOpenCV ZooのMediaPipe手専用ONNXを追加し、同じ画像の手位置が改善するか悪化するかを測る。手の回転ROIは全身モデルの予測した掌7点だけから作る。正解の手位置・手BBoxは手推論へ入力しない。正解の人物BBoxは利用するため、人物検出器の性能はこの評価に含まれない。

元画像と、手または足を灰色矩形で隠した画像を比較する。矩形実験は既知の正解を保持した静止画の入力劣化実験であり、現実の布・人体による遮蔽や時系列追跡の精度とは別の結果として扱う。

## 指標

Body、feet、handsを別々に集計し、平均・中央値・p95画素誤差、人物BBox長辺の5%と10%を閾値とするPCK、手BBox長辺の20%を閾値とするHand PCKを出す。正解注釈が有効であれば、モデルが低信頼でも位置誤差の分母から外さない。信頼度0.25以上の被覆率も独立表示する。Bodyのみ、公式の可視`v=2`と遮蔽`v=1`を分ける。手足の`v`には小数の信頼値が認められているため、整数1/2を遮蔽の定義として流用しない。

モデルURL、revision、SHA-256、実ONNX入力・出力名とshape、選択した人物ID、画像SHA-256、推論バックエンドを`report.json`へ保存する。予測値JSONと、正解・全身モデル・手refinementを重ねた証拠画像も保存する。モデルの巨大なバイナリをActions成果物へ再アップロードしない。

## 3Dと実機の境界

`verify-geometry.mjs`は、異なるカメラ中心・校正・時刻を使った三角測量の数学契約と、予測の失効を検査する。これは既知の合成3D入力による実装検証であり、実カメラでのセンチメートル精度の証拠ではない。実機の2視点動画、本人骨長、撮影位置の正解、LAN端末間通信、RTXのWebGPU速度は、それぞれの実入力があるときに評価する。

## 出典

- [COCO-WholeBody公式](https://github.com/jin-s13/COCO-WholeBody)
- [注釈形式とvisibilityの定義](https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md)
- [公式評価例](https://github.com/jin-s13/COCO-WholeBody/blob/master/evaluation/example_gt.json)
- [OpenCV Zoo MediaPipe Hand](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe)
- [RTMPose公式](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)

COCO画像は各撮影者のライセンス、COCO-WholeBody注釈は作者の研究・非商用条件に従う。評価用画像・注釈は取得して比較するだけで、アプリのUnlicenseの対象として再配布しない。
