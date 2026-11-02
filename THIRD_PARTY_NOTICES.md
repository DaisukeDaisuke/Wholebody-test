# 第三者のコード・モデル・評価データ
## 再利用コード
`dist/runtime.js` と `dist/ort-patch.js` は DaisukeDaisuke/VRC-mmpose の実装を継承し、runtime.jsに取得済みデータ限定モードの確認を追加した。元のApache-2.0本文は `third_party/VRC-mmpose-LICENSE` に保持する。元ファイルblob SHAはruntime.js=`34578f5e1f846ec9c712ca2bbcb213bf89f3c366`、ort-patch.js=`d5d0130dc93f1321e95559c439d1dd767729adf1`。
ONNX Runtime Web 1.27.0はMITライセンス。https://github.com/microsoft/onnxruntime/blob/main/LICENSE 。バージョン固定CDNから取得し、SHA-256を照合する。WebGPU FP32累積パッチを適用する構成を継承する。
## モデル
- RTMPose/RTMW/RTMW3D: OpenMMLab MMPose、Apache-2.0。https://github.com/open-mmlab/mmpose 。ユーザーのHugging Faceリポジトリから固定revisionで取得する。
- 高解像度RTMW-L 384×288: 同じテストリポジトリの別ProがActionsで取得・公開した公式ONNXを利用する。Hugging Face `RabbitDaisuke/Wholebody-Refinement-onnx` のrevision `0a5519d19e52477307916e99ef22024ac0faa241`、SHA-256 `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054`。Apache-2.0。公開元のrun https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41376834905 。
- MediaPipe手ONNX: OpenCV Zoo `models/handpose_estimation_mediapipe`、Apache-2.0。https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/handpose_estimation_mediapipe 。このディレクトリのライセンスとモデルハッシュを使用する。
- RTMDet人物検出: OpenMMLab系の公開変換モデル。取得先とhashは `dist/models.js` に記録する。
## 評価画像
COCO-WholeBodyの注釈とCOCO画像を評価に使用する。各画像の元URL・取得URL・hash・ライセンスIDはActionsの `report.json` に記録する。全画像が同じライセンスではない。画像はモデルの学習には使用しない。
画面の例示画像 `000000000785.jpg` はCOCOのlicense ID4（CC BY 2.0）、元Flickr画像URLは http://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg 。取得先はOpenMMLab MMPoseテストデータであり、画像の所有者ではない。ページは元画像へのリンクと評価記録を保持する。
## 3D評価データ
CMU Panopticは別途の研究データ利用条件を持つ。動画・生GT・校正ファイルはActionsの作業用キャッシュだけで扱い、サイトと公開成果物には集計値とモデル推論記録を保存する。データの条件は公式提供元を参照する。https://domedb.perception.cs.cmu.edu/ 。

HanCo Testerは研究用途限定・商用利用不可の公開データ。MANO適合由来の手3D注釈と同期した手画像を部品評価に使用する。元画像、参照3D、校正配列は再配布しない。出典: Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, Thomas Brox, *FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images*, ICCV 2019; Christian Zimmermann, Max Argus, Thomas Brox, *Contrastive Representation Learning for Hand Shape Estimation*, 2021。利用条件: https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html 。

FreeMoCap Validation Dataset v1.0（DOI 10.5281/zenodo.23041223）はCC BY 4.0。既に同期・整列・補間・フィルタ処理されたRTMPoseの3D系列とQualisys参照を、製品の因果的な補完処理の評価に使用する。生映像は含まれない。出典: *FreeMoCap Validation Dataset: Multi-Camera Markerless Motion Capture of Treadmill Gait and Standing Balance*, Version 1.0。https://doi.org/10.5281/zenodo.23041223 および https://github.com/freemocap/validation 。
