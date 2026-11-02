# フェーズ2までのモデル選定と精度改善
調査日: 2026-11-02。対象は映像による全身の位置、手指、足の向き・つま先、遮蔽時の推定。実行済みの測定結果はGitHub Actionsが生成する`evaluation/results/report.json`で確認し、この文書の公表値と区別する。
## 採用する構成
全身133点をRTMW-lで推論し、元映像の手領域を手専用RTMPose-m Hand5へ渡す。PCブラウザ内で一つずつ実行し、カメラごとに時刻を持つ観測として保存する。校正した二視点で共通関節を三角測量し、片方しか見えない関節は過去の位置・骨長・関節制約から推定する。
全身モデルへ渡す384×288画像では手が数十画素以下になるため、手の細部が足りない問題を全身モデルの大型化だけで解決しない。手専用推論は元映像から切り出し、全身推論用の縮小画像を再拡大しない。改善量は正解注釈との比較で判定する。
## 確認したモデル
|モデル|入力H×W|点数|公式の公表値|用途|
|---|---:|---:|---|---|
|RTMW-l Cocktail14|384×288|133|COCO-WholeBody Whole AP 70.1、17.7 GFLOPs|全身・手の初期位置・足先|
|RTMW-l Cocktail14|256×192|133|Whole AP 66.0、7.9 GFLOPs|低遅延の比較候補|
|RTMPose-m/DWPose COCO+UBody|256×192|133|Whole AP 60.6、2.22 GFLOPs|軽量の比較候補|
|RTMPose-m Hand5|256×256|21/手|Hand5 PCK@0.2 96.4、2.581 GFLOPs|手の細部再推論|
|MediaPipe Hand Landmarker|検出192×192・関節224×224|21/手|手中心の相対3Dも出力|ONNX以外の代替候補|
これらのAP/PCKは異なる評価条件を含み、そのまま当PoCの精度や4070 Ti Superの速度には置き換えられない。手モデルのHand5成績と全身モデルのCOCO Whole APは同じ指標ではない。[MMPose公式モデル一覧](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md)、[MediaPipe公式仕様](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)。
## 実ファイルと再現性
RTMW-l 384×288公式ONNX ZIP: https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip
Hand5公式ONNX ZIP: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip
RTMPose-m/DWPose公式ONNX ZIP: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip
公式ファイルは日付付きURLであり、Git revisionと同等の不変性は仮定しない。Actionsが取得したZIPとONNXのSHA-256、実際の入力・出力名・shapeを保存する。Hugging Faceミラーは`RabbitDaisuke/Wholebody-Refinement-onnx`を使用し、モデルカードに元モデル・変換手順・SHA-256を残す。上流コードはApache-2.0、データの利用条件は独立して扱う。既製重みを変換したものを自作学習済みモデルとは呼ばない。
## 前処理と後処理の契約
画像はRGB、`float32`のNCHW。各チャンネルで`(value-mean)/std`を使い、meanは`[123.675,116.28,103.53]`、stdは`[58.395,57.12,57.375]`。BBoxを中心を保って1.25倍にし、モデルの縦横比に合わせて不足側を広げ、黒パディングのアフィン変換を行う。
SimCC座標は各軸のargmaxを2で割る。元画像への逆変換は`inputXY/[inputW,inputH]*cropScale+cropCenter-cropScale/2`。2Dの公式MMPose codecに合わせてscoreは`min(max(x),max(y))`とし、scoreが0以下なら両軸のargmax indexを-1にしてから2で割る。scoreは校正済みの確率ではない。任意の`/5`やsoftmaxは入れない。RTMLibは独自に平均scoreを使用しているため、その閾値を無条件に流用しない。
アプリは公式ピーク値を`rawScore`として保存し、互換性のため`score`は0〜1へclipする。このscoreも未校正のモデル内の重みで、可視確率・正解確率やモデル間共通の品質尺度ではない。RTMW-lの[公式config](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-l_8xb320-270e_cocktail14-384x288.py)はKL損失の予測温度beta=1、Hand5はbeta=10で、[損失実装](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/classification_loss.py)が予測logitへこの倍率を掛ける。実測でもRTMWとHand5のピーク尺度は異なる。RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42`の[2D後処理](https://github.com/Tau-J/rtmlib/blob/03a1693e59e4f7cd84582c0fb30459b3bf18ad42/rtmlib/tools/pose_estimation/post_processings.py)にもRTMW専用の`/5`はなく、当PoCにも根拠のないモデル間のraw score比較や確率換算は加えない。
正方形のHand5出力は両方512要素なのでshapeだけではx/yを判別できない。`simcc_x`・`simcc_y`の出力名を優先する。RTMW-l 384×288はxが576、yが768。出力shapeを実ファイルで検証し、133点と21点を取り違えない。
公式ONNX例のBGR無変換で結果が変わる問題は[公式issue #2888](https://github.com/open-mmlab/mmpose/issues/2888)にも記録されている。[Hand5 config](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/rtmpose/hand_2d_keypoint/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py)、[SimCC decoder](https://github.com/open-mmlab/mmpose/blob/main/mmpose/codecs/simcc_label.py)を契約の根拠にする。
## 指と足で実際に得られる情報
COCO-WholeBodyの手は左右21点ずつで、手首と5指の関節を含む。足は左右それぞれ親指先、小指先、踵の3点。身体側の足首と組み合わせて足の向き・甲を表す面を推定できるが、5本の足指すべての関節が学習対象ではない。甲の法線や補間した中間指を表示する場合は幾何推定として区別する。[公式の点定義](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py)。
足の詳細についてはFOCUS/FINDも調査した。FOCUSは複数画像から密な対応点・COLMAP・最適化で足の形状を復元する研究実装で、既成の高速全身トラッカーの置換ではない。Foot3Dの多視点データは474画像・14足形状で、取得は申請フォーム経由。一般RGB映像から裸足の5指関節をリアルタイム追跡する軽量ONNXとしてそのまま採用できる配布物は、この調査では確認できなかった。[FOCUS](https://github.com/OllieBoyne/FOCUS)、[FIND](https://github.com/OllieBoyne/FIND)、[Foot3D](https://github.com/OllieBoyne/Foot3D)。
SAM 3D Bodyは手・足を含む3Dメッシュの候補だが、MHR70の足キーポイントも親指先・小指先・踵である。これに交換するだけでは5本の足指の独立した観測点は増えない。[公式SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)、[MHR70定義](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py)。
2026年の追加候補も確認した。CIGPose-x+UBodyは18.7 GFLOPs・Whole AP 67.5で、遮蔽への頑健性を狙ったモデルだが、RTMW-lより軽く同等精度とは公表値から言えない。NVIDIA GEM-Xは10月更新で手と時間方向の全身推定に対応する。Body Motion本体46.8Mパラメータに加え、Body PoseのDINOv3 ViT-H/16+と2D検出器を必要とし、軽量なブラウザ推論への即時置換にはならない。重みはNVIDIA側のモデル条件、手はEgoHand側の取得条件に従い、Apache-2.0の本ミラーへ一括再ライセンスしない。[CIGPose公式](https://github.com/53mins/CIGPose)、[GEM-X公式](https://github.com/NVlabs/GEM-X)、[GEM-Xモデルカード](https://huggingface.co/nvidia/GEM-X)。
FootMRと2026年10月公開のExMRは、全身姿勢の初期値へ手・足の時間方向の補正を追加する具体的な候補。FootMRは2D足先・膝・足首の文脈から足首回転の残差を推定し、ExMRはSAM 3D Bodyの身体特徴とHaMeRの手画像特徴も使う。5本の足指を独立観測するものではなく、現在の2D ONNXをそのまま差し替える構成でもない。ExMRは研究・教育・非営利条件で、SMPL-X・MANOの登録やSAM 3D Bodyのアクセス申請が別途ある。MOOFは41動画・14,589フレームの実動画評価候補だが、注釈は足ごとに3点、データは署名した利用契約の提出後に取得する。現PoCの試験済みモデルや公開評価データには含めない。[FootMR](https://github.com/twehrbein/FootMR)、[ExMR](https://github.com/twehrbein/ExMR)、[公式MOOF説明](https://www.tnt.uni-hannover.de/en/project/HumanMotionDatabases/)。
別プロジェクトの[Logimade/FOCUS](https://github.com/Logimade/FOCUS)には、COCO-WholeBody由来の足8点（左右の母趾・小趾・踵・足首）を検出するYOLO26n-poseの小型重みがある。`weights/best.pt`は6,396,455bytes、Git blob `aaab7daf96fd2253a64268c3f897391156e5864a`。足ROI再推論の比較候補だが未ONNX・当PoC未実測で、5指の独立関節を増やすモデルではない。独自コードのMIT表記とYOLO由来モデルの利用条件は分けて確認し、Apache-2.0ミラーへ条件を上書きして配置しない。これは形状復元のOllieBoyne/FOCUSとは別研究である。
## 遮蔽への対応
別カメラから見える関節を統合する処理を優先する。時刻差が大きい観測、再投影誤差が大きい対応、平行に近い視線は3D測定として採用しない。片視点の位置に過去の骨長・速度を加えた補完と、両視点の三角測量を区別する。
全カメラから隠れた指は、映像に存在しない正解を確定できない。過去の関節角・可動域・手首運動から推定はできるが、遮蔽が長いほど不確かさを増やす。再観測したら観測へ滑らかに戻す。長時間の完全遮蔽を確実な観測として保持しない。
自作モデルを追加する場合は画像認識器全体の再学習より、欠損マスク付き2D/3D座標から姿勢を補う小さなモデルが先候補。H3WBには133点の2D→3D・不完全2D→3Dタスクがあり、遮蔽補完の定量比較が可能。ただし現在の評価スクリプトはこの学習・精度達成を含まない。[H3WB公式データとタスク](https://github.com/wholebody3d/wholebody3d)、[公表ベンチマーク](https://github.com/wholebody3d/wholebody3d/blob/main/benchmark.md)。
## Actionsでの実画像評価
`evaluation/benchmark.py`は実ONNXをCPU推論し、MMPose公式repoのCOCO-WholeBody小標本を使用する。データ元をcommit `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`に固定。4画像・最大14人物なので統計的な精度保証には使わず、前処理不整合、手の再推論の改善・悪化、遮蔽時の破綻を確認する。
身体・手・足を別々に採点し、低信頼点も分母へ含める。モデル単体の基準比較は正解人物BBoxを使うため検出器の評価ではない。手のROIは予測座標のみから作り、正解手BBoxは採点尺度にだけ使用する。手・足・胴体を人工的に隠した入力でも同じ推論を行い、隠した点だけの成績を別集計する。静止画の人工遮蔽から、時間補完・実際の多視点・利用者環境の精度達成は主張しない。
加えて、アプリの入力経路を測るため、画像ごとの最大人物を採点対象として固定し、正解BBoxを入力しない全画面推論→予測人物ROI再推論を行う。全画面と予測ROIのそれぞれでHand5なし・従来採用・選択採用を採点し、6経路を比較する。ROIは身体・足23点のうちscore>0.4を8点以上採れた場合だけ作り、幅は画像幅の8%以上、高さは画像高の18%以上、外接範囲から`[left-0.25w,top-0.15h,1.5w,1.3h]`へ広げる。この式はアプリの初回tracker更新と同じ。採点後の人物対応診断はターゲットや入力ROIを変更しない。`initialization-predictions.json`に実座標、`report.json`の`initializationExperiment`に部位別数値・各画像の結果を保存する。同じ静止画を2回使う初期化診断であり、動画追跡性能とは区別する。
手の選択採用`hand_selective`は、同じHand5候補を追加推論せずに判定する。従来の平均score条件に加え、手首自身のraw scoreが0.2以上、身体側の手首からの移動が手ROI辺の0.35以下、21点中11点以上のraw scoreが0.2以上、モデルcropの1px以内の境界に張り付く点が10点以下、を要求する。境界はBBoxを1.25倍してモデル縦横比へ合わせたアフィンcrop上で判定する。これは[Actions run 41375521773](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41375521773)で、平均scoreが0.2を超えていても全指がcrop隅へ集中する失敗が確認されたために設計したルール。画像IDやGTによる例外は入れないが、同じ診断データを見て設計したため、改善結果は探索的な再測定として扱い、未知映像への汎化実証にはしない。旧方式も数値に残し、悪化を隠さない。
注釈はCOCO-WholeBody公式の研究・非商用条件に従い、画像は個別のFlickrライセンスを記録する。公開する重ね描き画像はCC BYの画像785に限定する。NC-ND画像の派生画像は生成成果へ含めない。[注釈利用条件](https://github.com/jin-s13/COCO-WholeBody#terms-of-use)。
実行例: `python evaluation/benchmark.py --output evaluation/results --cache .cache/wholebody --max-people 14`。モデル差替えは`--body-model URL_or_path --body-model-id identifier`、手モデルは`--hand-model URL_or_path`。実行場所はGitHub Actionsとし、ローカル推論やクラウドブラウザの手動テストは行わない。
## ブラウザ用前後処理との数値一致
Actionsで上記コマンドに`--emit-codec-fixture`を加えると、実画像1枚のRGBA、OpenCVで作った正規化テンソル、実ONNX出力のSimCC、復元した座標を保存する。続けて`node evaluation/codec-parity.mjs --fixture evaluation/results/codec-fixture --output evaluation/results/codec-parity.json`を実行し、アプリで使う`dist/pose-codec.mjs`の前処理とデコードを同じ入力で比較する。
テンソルの最大絶対差0.02・平均絶対差0.0001、座標差0.0001px、score差0.000001を判定基準とする。最大差には8bit補間時の1階調の丸め差を許容するが、平均差でも判定し、RGB/BGRの取り違えや切り出し位置のずれを隠さない。この比較は精度の代用やWebGPU実行の確認ではなく、Python側で測ったモデル結果をアプリが再現するための前後処理の検証である。
