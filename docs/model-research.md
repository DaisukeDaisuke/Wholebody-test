# フェーズ2までのモデル選定と精度改善

更新日: 2026-11-02。対象は映像による全身の位置、手指、足の向き・つま先、遮蔽時の推定。既製モデルの組合せと採用判定をフェーズ2まで実装し、新たな画像認識モデルの学習は行っていない。この文書は上流の公表値と開発用4画像の診断を記録する。完了した未使用32画像の成績と最終判断は[固定標本検証](holdout-results.md)にまとめた。足専用補正は未使用画像で回帰したため既定OFFを確定した。

## 採用する構成

YOLOX-tiny HumanArtで人物を検出し、初回は最大のperson領域を選ぶ。その領域でRTMW-lが全身133点を推論し、元映像の手領域をRTMPose-m Hand5へ渡す。手は身体側の手首との整合性などを満たした候補だけ採用する。手動人物ROIが指定された場合はそちらを優先し、人物未検出時は全画面の推論結果で埋めず`missing`として扱う。実装は`dist/inference.mjs`、判定契約は`dist/pose-codec.mjs`と`dist/detector-codec.mjs`にある。

足はLogimade/FOCUSのone2many出力を追加比較できる。人物領域を1.25倍した範囲で検出し、同側の膝・足首に整合する足先だけ元の位置と50%ずつ混ぜる。4画像の診断で候補に残ったが、未使用32画像では通常・遮蔽の両方で悪化し、既定OFFとした。FOCUSの足だけを小さく切り出す経路とone2one出力も測定結果には残す。

2D推論はPCブラウザ内で順番に実行し、カメラごとの時刻を持つ観測として保存する。人物検出のONNXに含まれるNMSはWASMで実行し、身体・手・足は選択したbackendを使う。校正した二視点からの3D推定、単眼モデルの相対3D、過去の位置や骨長による補完は出力の由来を分ける。RTMW-l、Hand5、FOCUS自体の出力は2Dであり、これらが直接3Dを返すとは扱わない。

全身モデルへ渡す384×288画像では手が数十画素以下になるため、手の細部が足りない問題を全身モデルの大型化だけで解決しない。手専用推論は元映像から切り出し、全身推論用の縮小画像を再拡大しない。改善量は正解注釈との比較で判定する。

## 確認したモデル

|モデル|入力H×W|点数|公式の公表値|用途|
|---|---:|---:|---|---|
|YOLOX-tiny HumanArt|416×416|person検出BBox|本PoCでは下記4画像の人物選択で評価|初回の人物特定・再捕捉|
|RTMW-l Cocktail14|384×288|133|COCO-WholeBody Whole AP 70.1、17.7 GFLOPs|全身・手の初期位置・足先|
|RTMW-l Cocktail14|256×192|133|Whole AP 66.0、7.9 GFLOPs|低遅延の比較候補|
|RTMPose-m/DWPose COCO+UBody|256×192|133|Whole AP 60.6、2.22 GFLOPs|軽量の比較候補|
|RTMPose-m Hand5|256×256|21/手|Hand5 PCK@0.2 96.4、2.581 GFLOPs|手の細部再推論|
|Logimade/FOCUS YOLO26n-pose|640×640|8、うち足先6・足首2|本PoCでは下記のPCK/画素誤差で評価|既製モデルによる足の補正候補|
|MediaPipe Hand Landmarker|検出192×192・関節224×224|21/手|手中心の相対3Dも出力|ONNX以外の代替候補|

これらのAP/PCKは異なる評価条件を含み、そのまま当PoCの精度や4070 Ti Superの速度には置き換えられない。手モデルのHand5成績と全身モデルのCOCO Whole APは同じ指標ではない。[MMPose公式モデル一覧](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md)、[MediaPipe公式仕様](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)。

## 実ファイルと再現性

OpenMMLabの[RTMW-l公式ONNX](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip)、[Hand5公式ONNX](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip)、[YOLOX-tiny公式ONNX](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip)は変換・量子化せず取得した。[Wholebody-Refinement-onnx](https://huggingface.co/RabbitDaisuke/Wholebody-Refinement-onnx)で上流Apache-2.0条件と由来を明記し、アプリはrevision固定URLとSHA-256を使用する。公式の日付付きURLだけに不変性を仮定せず、取得ZIPと展開ONNX双方のhashをActionsへ記録する。

|実測したONNX|bytes|SHA-256|
|---|---:|---|
|YOLOX-tiny HumanArt|20,283,006|`ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`|
|RTMW-l 384×288|229,320,930|`bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054`|
|Hand5 256×256|55,080,248|`39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29`|
|FOCUS one2many、32画像検証用pin|10,687,517|`f4f58eac45f40870fe2c6bc774400423c11f8e9a64ae1d9daa186a72e7142f7b`|
|FOCUS one2one、32画像検証用pin|10,700,493|`7569eddfa65c943b9a2fe0e1d6e13a57fbd7c9f69678c59863daf67d9dd7988b`|

FOCUSは[上流commit](https://github.com/Logimade/FOCUS/tree/ed2ee8e394162b823b575f8d2fd5b7d625dcd74c)の既製`weights/best.pt`をActionsでONNXへexportした。再学習はしていない。YOLO由来のAGPL-3.0条件を保持し、Apacheモデルとは別の[Foot-Refinement-onnx](https://huggingface.co/RabbitDaisuke/Foot-Refinement-onnx)へ公開した。32画像の検証はrevision `c7a40271d743a6cef447ee63c6246399418aa96f`に固定する。アプリ側one2manyのrevision `8c514d9f9cde4a596bad28b7a7fbbfdb2e6aa562`は再export時のmetadataを含む別ファイルで、SHA-256は`e975d01f6c9af4c06c0168893a9485f501ec6c78801f9db57aad7225fa60eb6f`。両方の実ファイル識別子を混同しない。hash・shapeの一次記録は[モデル評価run 41377683651](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41377683651)と[足評価run 41377683638](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41377683638)の成果物にある。データの利用条件はモデルコードのライセンスと独立して扱う。

## 前処理と後処理の契約

### 人物検出

YOLOX-tinyはBGR、0〜255の`float32` NCHWで入力し、平均・標準偏差による正規化をしない。416×416へ縦横比を保って縮小し、寸法は切り捨て、画像を左上へ配置して右・下を114で埋める。実ONNXは`dets=[1,N,5]`のxyxy・scoreと`labels=[1,N]`のint64を返す。Nは画像依存で、最初の実画像では9だった。class 0、score 0.3以上へ絞り、画像範囲へclipしたBBoxにIoU 0.45のNMSを行ってから面積順で選ぶ。[公式HumanArt設定](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/yolox/humanart/yolox_tiny_8xb8-300e_humanart.py)、[RTMLibの同配布モデルの前後処理](https://github.com/Tau-J/rtmlib/blob/03a1693e59e4f7cd84582c0fb30459b3bf18ad42/rtmlib/tools/object_detection/yolox.py)、上記モデル評価runの`detector-report.json`を根拠にする。

### 身体と手

画像はRGB、`float32`のNCHW。各チャンネルで`(value-mean)/std`を使い、meanは`[123.675,116.28,103.53]`、stdは`[58.395,57.12,57.375]`。BBoxを中心を保って1.25倍にし、モデルの縦横比に合わせて不足側を広げ、黒パディングのアフィン変換を行う。
SimCC座標は各軸のargmaxを2で割る。元画像への逆変換は`inputXY/[inputW,inputH]*cropScale+cropCenter-cropScale/2`。2Dの公式MMPose codecに合わせてscoreは`min(max(x),max(y))`とし、scoreが0以下なら両軸のargmax indexを-1にしてから2で割る。scoreは校正済みの確率ではない。任意の`/5`やsoftmaxは入れない。RTMLibは独自に平均scoreを使用しているため、その閾値を無条件に流用しない。
アプリは公式ピーク値を`rawScore`として保存し、互換性のため`score`は0〜1へclipする。このscoreも未校正のモデル内の重みで、可視確率・正解確率やモデル間共通の品質尺度ではない。RTMW-lの[公式config](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-l_8xb320-270e_cocktail14-384x288.py)はKL損失の予測温度beta=1、Hand5はbeta=10で、[損失実装](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/classification_loss.py)が予測logitへこの倍率を掛ける。実測でもRTMWとHand5のピーク尺度は異なる。RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42`の[2D後処理](https://github.com/Tau-J/rtmlib/blob/03a1693e59e4f7cd84582c0fb30459b3bf18ad42/rtmlib/tools/pose_estimation/post_processings.py)にもRTMW専用の`/5`はなく、当PoCにも根拠のないモデル間のraw score比較や確率換算は加えない。
正方形のHand5出力は両方512要素なのでshapeだけではx/yを判別できない。`simcc_x`・`simcc_y`の出力名を優先する。RTMW-l 384×288はxが576、yが768。出力shapeを実ファイルで検証し、133点と21点を取り違えない。
公式ONNX例のBGR無変換で結果が変わる問題は[公式issue #2888](https://github.com/open-mmlab/mmpose/issues/2888)にも記録されている。[Hand5 config](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/rtmpose/hand_2d_keypoint/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py)、[SimCC decoder](https://github.com/open-mmlab/mmpose/blob/main/mmpose/codecs/simcc_label.py)を契約の根拠にする。

### FOCUSの足補正

FOCUSの入力は人物contextを画像範囲へclipしたcropを、中央114 paddingの640×640 RGBへletterboxした`float32` NCHW、値域0〜1。one2manyは`[1,29,8400]`、one2oneは`[1,300,30]`で、8点の順序はCOCO index `[17,18,19,15,20,21,22,16]`。足首は候補対応の検証に使い、補正する点は足先6点だけである。detector score 0.25以上の上位64候補を対象に、同側足首の距離が`max(3px,下腿長×0.30)`以内、足先の半径とconfidenceが整合する点が2個以上、左右足首が潰れていないことを確認する。採用候補を固定0.5で混ぜ、blendではRTMWのscoreを保持する。FOCUSのraw scoreとRTMWのraw scoreを大小比較しない。全定数は`evaluation/foot-evaluate.py`の`ASSOCIATION_POLICY`、同じアプリ側契約は`dist/foot-codec.mjs`にある。

## 指と足で実際に得られる情報

COCO-WholeBodyの手は左右21点ずつで、手首と5指の関節を含む。足は左右それぞれ親指先、小指先、踵の3点。身体側の足首と組み合わせて足の向き・甲を表す面を推定できるが、5本の足指すべての関節が学習対象ではない。甲の法線や補間した中間指を表示する場合は幾何推定として区別する。[公式の点定義](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py)。
足の詳細についてはFOCUS/FINDも調査した。FOCUSは複数画像から密な対応点・COLMAP・最適化で足の形状を復元する研究実装で、既成の高速全身トラッカーの置換ではない。Foot3Dの多視点データは474画像・14足形状で、取得は申請フォーム経由。一般RGB映像から裸足の5指関節をリアルタイム追跡する軽量ONNXとしてそのまま採用できる配布物は、この調査では確認できなかった。[FOCUS](https://github.com/OllieBoyne/FOCUS)、[FIND](https://github.com/OllieBoyne/FIND)、[Foot3D](https://github.com/OllieBoyne/Foot3D)。
SAM 3D Bodyは手・足を含む3Dメッシュの候補だが、MHR70の足キーポイントも親指先・小指先・踵である。これに交換するだけでは5本の足指の独立した観測点は増えない。[公式SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)、[MHR70定義](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py)。
2026年の追加候補も確認した。CIGPose-x+UBodyは18.7 GFLOPs・Whole AP 67.5で、遮蔽への頑健性を狙ったモデルだが、RTMW-lより軽く同等精度とは公表値から言えない。NVIDIA GEM-Xは10月更新で手と時間方向の全身推定に対応する。Body Motion本体46.8Mパラメータに加え、Body PoseのDINOv3 ViT-H/16+と2D検出器を必要とし、軽量なブラウザ推論への即時置換にはならない。重みはNVIDIA側のモデル条件、手はEgoHand側の取得条件に従い、Apache-2.0の本ミラーへ一括再ライセンスしない。[CIGPose公式](https://github.com/53mins/CIGPose)、[GEM-X公式](https://github.com/NVlabs/GEM-X)、[GEM-Xモデルカード](https://huggingface.co/nvidia/GEM-X)。
FootMRと2026年10月公開のExMRは、全身姿勢の初期値へ手・足の時間方向の補正を追加する具体的な候補。FootMRは2D足先・膝・足首の文脈から足首回転の残差を推定し、ExMRはSAM 3D Bodyの身体特徴とHaMeRの手画像特徴も使う。5本の足指を独立観測するものではなく、現在の2D ONNXをそのまま差し替える構成でもない。ExMRは研究・教育・非営利条件で、SMPL-X・MANOの登録やSAM 3D Bodyのアクセス申請が別途ある。MOOFは41動画・14,589フレームの実動画評価候補だが、注釈は足ごとに3点、データは署名した利用契約の提出後に取得する。現PoCの試験済みモデルや公開評価データには含めない。[FootMR](https://github.com/twehrbein/FootMR)、[ExMR](https://github.com/twehrbein/ExMR)、[公式MOOF説明](https://www.tnt.uni-hannover.de/en/project/HumanMotionDatabases/)。
今回実装したLogimade/FOCUSは上記の形状復元用OllieBoyne/FOCUSとは別プロジェクトであり、足の観測点を5本の足指関節へ増やすモデルではない。出力の甲面や中間の足指関節列は`inferredAnatomy`、足指の独立した曲げは`articulationStatus='unobserved'`として扱う。2Dの足3点から作った5指の形状を、映像から実測した独立関節として数えない。

## 遮蔽への対応

別カメラから見える関節を統合する処理を優先する。時刻差が大きい観測、再投影誤差が大きい対応、平行に近い視線は3D測定として採用しない。片視点の位置に過去の骨長・速度を加えた補完と、両視点の三角測量を区別する。
全カメラから隠れた指は、映像に存在しない正解を確定できない。過去の関節角・可動域・手首運動から推定はできるが、遮蔽が長いほど不確かさを増やす。再観測したら観測へ滑らかに戻す。長時間の完全遮蔽を確実な観測として保持しない。
自作モデルを追加する場合は画像認識器全体の再学習より、欠損マスク付き2D/3D座標から姿勢を補う小さなモデルが先候補。H3WBには133点の2D→3D・不完全2D→3Dタスクがあり、遮蔽補完の定量比較が可能。ただし現在の評価スクリプトはこの学習・精度達成を含まない。[H3WB公式データとタスク](https://github.com/wholebody3d/wholebody3d)、[公表ベンチマーク](https://github.com/wholebody3d/wholebody3d/blob/main/benchmark.md)。
## 実行済みの4画像診断

### 人物検出を入れた入力経路

同じ4画像の最大人物を採点対象へ事前固定した比較。採点点数は身体62、手126、足17で、正解BBoxを渡した経路を除く人物選択にGTは使わない。PCK@0.2は、身体では人物BBoxの長辺、手では注釈手BBoxの長辺、足では注釈足先の最大広がりの20%以内を正解とする。部位間で許容する画素誤差は同じではない。

|入力経路|身体PCK@.2|手PCK@.2|足PCK@.2|手の平均誤差px|
|---|---:|---:|---:|---:|
|全画面RTMW|59.7%|19.0%|29.4%|143.484|
|人物検出→RTMW|100.0%|77.8%|76.5%|3.551|
|人物検出→RTMW→Hand5選択採用|100.0%|81.7%|76.5%|3.025|
|正解人物BBox→RTMW→Hand5選択採用、reference|100.0%|81.0%|70.6%|2.825|

人物検出では4画像とも事前固定対象が選ばれ、正解BBoxとのIoUは0.937〜0.958だった。全画面入力で別人物を追っていた問題がこの標本では解消した。一方、手の従来採用もPCK81.7%で、平均誤差2.974pxは選択採用より小さい。全ての指標で選択採用が優れるとは言わない。この表は4対象の初期化評価であり、12人物のモデル単体評価や動画追跡の成績とは別物である。[モデル評価run 41377683651](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41377683651)の`detector-report.json`と`report.json`。

### 手と足の再推論

正解人物BBoxを用いる12人物の診断では、手210点のclean PCK@.2はRTMWの66.7%から選択採用69.5%へ改善した。胴体を覆った条件では従来採用が54.3%から47.6%へ悪化したが、手首とcrop境界を確認する選択採用は55.7%だった。同じ4画像の失敗を見て設計したルールなので、この再測定を未知映像への汎化証明には使わない。覆った点だけの数値は`maskedPointMetrics`へ分けて保存している。[同runのreport.json](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41377683651)。

FOCUS one2many・person context・50% blendの足37点の測定は次の通り。こちらも正解人物BBoxを使った専門モデルの比較であり、上の4対象・足17点の検出器経路と数値を直接混ぜない。

|足の条件|RTMW PCK@.2|FOCUS blend PCK@.2|RTMW平均誤差px|FOCUS blend平均誤差px|
|---|---:|---:|---:|---:|
|clean、37点|40.5%（15/37）|48.6%（18/37）|4.691|4.398|
|足mask、全37点|35.1%（13/37）|40.5%（15/37）|6.217|6.397|
|足maskで覆った18点だけ|33.3%（6/18）|38.9%（7/18）|7.249|7.817|

足maskではPCKが改善する一方で平均誤差とp95誤差は悪化する。one2oneや足だけcropにも回帰があり、全方式の結果を隠さず保存した。現時点ではone2many blendを追加検証する候補として扱い、一般的な精度改善や完全遮蔽への有効性は断定しない。[足評価run 41377683638](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41377683638)の`foot-report.json`。

## Actionsでの実画像評価

`evaluation/benchmark.py`は実ONNXをCPU推論し、MMPose公式repoのCOCO-WholeBody小標本を使用する。データ元をcommit `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`に固定。4画像・14注釈中、身体ラベル数とBBox条件を満たす12人物を採点する。統計的な精度保証には使わず、前処理不整合、手の再推論の改善・悪化、遮蔽時の破綻を確認する。
身体・手・足を別々に採点し、低信頼点も分母へ含める。モデル単体の基準比較は正解人物BBoxを使うため検出器の評価ではない。手のROIは予測座標のみから作り、正解手BBoxは採点尺度にだけ使用する。手・足・胴体を人工的に隠した入力でも同じ推論を行い、隠した点だけの成績を別集計する。静止画の人工遮蔽から、時間補完・実際の多視点・利用者環境の精度達成は主張しない。
加えて、アプリの入力経路を測るため、画像ごとの最大人物を採点対象として固定し、正解BBoxを入力しない全画面推論→予測人物ROI再推論を行う。全画面と予測ROIのそれぞれでHand5なし・従来採用・選択採用を採点し、6経路を比較する。ROIは身体・足23点のうちscore>0.4を8点以上採れた場合だけ作り、幅は画像幅の8%以上、高さは画像高の18%以上、外接範囲から`[left-0.25w,top-0.15h,1.5w,1.3h]`へ広げる。この式はアプリの初回tracker更新と同じ。採点後の人物対応診断はターゲットや入力ROIを変更しない。`initialization-predictions.json`に実座標、`report.json`の`initializationExperiment`に部位別数値・各画像の結果を保存する。同じ静止画を2回使う初期化診断であり、動画追跡性能とは区別する。
手の選択採用`hand_selective`は、同じHand5候補を追加推論せずに判定する。従来の平均score条件に加え、手首自身のraw scoreが0.2以上、身体側の手首からの移動が手ROI辺の0.35以下、21点中11点以上のraw scoreが0.2以上、モデルcropの1px以内の境界に張り付く点が10点以下、を要求する。境界はBBoxを1.25倍してモデル縦横比へ合わせたアフィンcrop上で判定する。これは[Actions run 41375521773](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41375521773)で、平均scoreが0.2を超えていても全指がcrop隅へ集中する失敗が確認されたために設計したルール。画像IDやGTによる例外は入れないが、同じ診断データを見て設計したため、改善結果は探索的な再測定として扱い、未知映像への汎化実証にはしない。旧方式も数値に残し、悪化を隠さない。
注釈はCOCO-WholeBody公式の研究・非商用条件に従い、画像は個別のFlickrライセンスを記録する。公開する重ね描き画像はCC BYの画像785に限定する。NC-ND画像の派生画像は生成成果へ含めない。[注釈利用条件](https://github.com/jin-s13/COCO-WholeBody#terms-of-use)。
実行例: `python evaluation/benchmark.py --output evaluation/results --cache .cache/wholebody --max-people 14`。モデル差替えは`--body-model URL_or_path --body-model-id identifier`、手モデルは`--hand-model URL_or_path`。実行場所はGitHub Actionsとし、ローカル推論やクラウドブラウザの手動テストは行わない。

## 開発例から分離した32画像検証の固定規則

`evaluation/holdout-evaluate.py`は上記4画像を全て除外したCOCO-WholeBody val v1.0を使う。公式READMEが指定する[Google Driveのval注釈](https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb/view)を取得し、[revision固定の公開ミラー](https://huggingface.co/datasets/msdkhairi/coco2017/blob/159fa91cbeccbb944e666a399e43b4cf25e2d3e9/coco_wholebody_val_v1.0.json)と同じ132,970,317 bytes、SHA-256 `f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e`であることを確認した。全注釈は5,000画像・11,004注釈で、全画像ZIPのダウンロードは必要ない。

選択規則はモデル実行前に固定する。

1. 画像ID `785,40083,196141,197388`を除外する。
2. 各画像で、非crowd・身体ラベル1点以上・BBox両辺20px以上を満たす人物のうち、BBox面積が最大の人を先に確定する。同面積なら小さい注釈IDを選ぶ。
3. その人物に採点可能な手注釈と、2個以上の異なる足先点がある画像を候補とする。対象を別人へ変更して条件を満たすことはしない。
4. seed `wholebody-phase2-4fc7ff60-holdout-v1`と画像ID・注釈IDから`SHA256(seed:imageId:annotationId)`を計算し、昇順の32画像を選ぶ。
5. 選択ID・規則・注釈hashを`holdout-selection.json`へ保存してから、対象JPEGだけを公式COCO S3バケットから取得する。取得不能な画像は別候補へ入れ替えず、検証未完了として報告する。

この規則は655候補から32対象を選ぶ。選択列のSHA-256は`023d9faa25098ee0c7d03facec9f610161e8f0e4b6bd931c9e70d8c7ded3817d`、採点可能な注釈は身体477点・手1,132点・足157点。これは注釈の集計結果であり、モデルの精度結果ではない。

clean・手mask・足maskの3条件について、人物検出cropと正解人物cropのreferenceを別々に比較する。RTMW、Hand5選択採用、FOCUS両headのassociation・固定50% blendを全て残し、覆った点も別集計する。足の閾値・手の判定・seed・採用重みをこの32画像の成績から調整しない。人物未検出は分母へ含むPCK失敗とし、元画像・加工画像は成果物へ含めない。

この32画像は当PoCの判定規則を設計した4画像から分離されている。ただし、上流モデルの学習・モデル選択にCOCO valが使われなかったことは保証できず、`trainingOverlapExcluded=false`とする。また手足注釈のある最大人物を選ぶ標本であり、利用者の全映像を代表する抽出ではない。実行はActionsの`python evaluation/holdout-evaluate.py --output evaluation/results/holdout-report.json --cache .cache/wholebody --max-images 32 --threads 2`。run 41378750685で全32画像の評価を完了し、[詳細結果](holdout-results.md)に失敗・回帰も含めて保存した。

## ブラウザ用前後処理との数値一致

Actionsで上記コマンドに`--emit-codec-fixture`を加えると、実画像1枚のRGBA、OpenCVで作った正規化テンソル、実ONNX出力のSimCC、復元した座標を保存する。続けて`node evaluation/codec-parity.mjs --fixture evaluation/results/codec-fixture --output evaluation/results/codec-parity.json`を実行し、アプリで使う`dist/pose-codec.mjs`の前処理とデコードを同じ入力で比較する。
テンソルの最大絶対差0.02・平均絶対差0.0001、座標差0.0001px、score差0.000001を判定基準とする。最大差には8bit補間時の1階調の丸め差を許容するが、平均差でも判定し、RGB/BGRの取り違えや切り出し位置のずれを隠さない。この比較は精度の代用やWebGPU実行の確認ではなく、Python側で測ったモデル結果をアプリが再現するための前後処理の検証である。

run 41377683651では身体の入力テンソル最大差0、復元座標の最大差`1.14e-13`pxだった。実際のHand5出力から保存した採用・手首不整合・境界集中の3例でもPython/JavaScriptの採否が一致した。人物検出とFOCUSはそれぞれの実ONNX出力を使って別codecを比較する。ここで検証するのは前後処理と採用判定であり、同じモデルをブラウザのWebGPUで動かした精度・速度や実動画の追跡成功を確認したものではない。前後処理以外の新規のブラウザ手動試験や大量の単体テストで、精度実測の代わりをしない。
