# VRC Compact 59

VRC向けに必要な関節と表情へ絞った軽量モデルの研究PoC。独立ブランチ compact-vrc-20261102-7fe4fac1 だけで開発する。

## 現行版の構成

採用モデルは精度を維持した標準Compact59。最初の低解像度・縮小head実験はCPU時間を減らしたが、独立testでPCK@0.05が約9.9ポイント低下したため不採用とした。実験モデルは比較用の記録として残し、既定モデルへ自動昇格させない。

次の学習はColabのCUDA GPUで行う。既存の学習済みheadを維持した小さい入力の候補と、標準59点のINT8候補を比較する。過去実験の画像IDをvalidation/testから除外し、画像キャッシュは返却ZIPへ入れない。GPUでのPyTorch推論とCPUでのONNX推論を別の測定として記録する。

Hugging Faceへの公開とヘッドレスNode.js推論は `compact.yml` で行う。`HF_TOKEN` はActionsの公開ステップだけに渡す。ブラウザとNodeは、公開された重みの固定commit SHAを指定したHFの `/resolve/` URLから認証なしで取得し、SHA-256を検証する。Sitesには重みを同梱しない。`fast59.yml` の長時間CPU学習は自動起動を止めた。

動画とVRMは常に横並び。単眼では胴体の画像平面上の傾きを反映し、観測できないyaw/pitch・絶対root移動を抑制する。範囲外・不自然な骨長・時間跳躍を除外し、腰や肩が見えなくても可視の前腕・下腿は残す。部位単位の短い保持と腕を下げた待機姿勢への復帰、再検出の連続確認、角速度制限を行う。観測がない点をトラッカーの新しい観測として送らない。

推論はCPU/WASM・全身15Hzが既定で、全身の頻度とWebGPU利用を変更できる。顔はCPU/WASMで最大10Hz、足はCPU/WASMで最大5Hz、有効な領域があるときだけ推論する。VRMは30Hzで1回だけ更新し、一時停止中は視点の変更時だけ再描画する。15Hzは実Node検証と同じ頻度であり、1回のモデル推論時間を短縮したこととは区別する。

## Colab GPU 学習

[GPUノートブックを開く](https://colab.research.google.com/github/DaisukeDaisuke/Wholebody-test/blob/8cae0c0b9243628a3910060a0bdc09d8b04b13e8/colab/VRC_Compact59_GPU_7fe4fac1.ipynb)。学習用ZIPを選択し、GPUランタイムですべてのセルを実行する。2種類の低解像度backbone候補を4000 stepずつ学習し、元解像度を保つINT8候補も2種類比較する。詳しい構成は[colab/README-ja.md](colab/README-ja.md)。返却ファイル名は `VRC_Compact59_Colab_Result_7fe4fac1.zip`。

[Actions run 41394188181](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41394188181) で、Python 3.13の依存導入、実画像での短い学習更新、ONNX書き出しと数値照合、INT8校正と推論まで確認済み。これはCPUでの実行確認であり、本番のCUDA学習や精度達成を意味しない。

## 実動画とVRMの検証

[Actions run 41394715364](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41394715364) で、公式公開動画10秒・150フレームを実際に推論し、実VRMを299回更新した。同じ観測からの旧版比較で、骨ローカル回転変化のp95は19.485°/frameから1.753°/frame、最大は180°から30°となった。149描画フレームで観測骨の駆動を保持している。詳細・測定範囲は[RESULTS.md](RESULTS.md)。

非公開動画はActionsの入力secretが未設定のため未実行。公開動画の結果をその代わりとは扱わない。新しいGPU候補もColabの本学習結果が戻るまで採用しない。

## 先行Compact59の実測結果

2026-11-02の[GitHub Actions](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41385649138)で、学習と画像IDが重複しない公開128画像を評価した。元の133点モデルと同じ59点を比較すると、PCK@0.05は85.76%から85.66%（−0.10ポイント）、CPU推論中央値は10.03msから7.77ms（22.5%短縮）となった。ONNXは22.10MBから16.53MB、パラメータは552万から413万、MACsは5.48億から4.23億へ減った。MBは10進表記。

この評価で蒸留自体のPCK改善は確認できなかった。再学習前の構造縮小モデルは85.87%であり、主な効果は出力畳み込みと関節間注意を実際に小さくしたことにある。蒸留後は元モデルの精度を概ね維持し、平均座標誤差は9.414から9.370入力pxへ小幅に改善した。評価画像でチェックポイントを選び直していない。

PCK@0.05は192×256人物切り抜き上の距離12.8px以内を正解とする値。公式COCO AP、まばたき認識率、GPU使用率、動画全体のfpsとは異なる。詳しい条件と部位別結果は[RESULTS.md](RESULTS.md)を参照。

## 実装

身体13点、足6点、両手30点、顔10点の59点。DWPose-tの学習済みCSPNeXtを保持し、133チャネルの7×7出力畳み込みと133 tokenのGAUを59へ縮め、PyTorchで教師分布と座標ラベルから再学習する。出力をGatherで間引くだけの変換ではない。

先行Compact59では公開COCO画像512枚を学習、画像IDが重複しない128枚を評価に使用した。元の133点、構造縮小した直後、蒸留後を同じ入力で比較し、PCK・部位別誤差・重み数・MACs・ONNXサイズ・CPU中央値/p95を記録した。精度は人物の正解矩形を使用する条件付き評価であり、画像全体からの人物検出精度ではない。

足の向きはFootNet公開191画像の足裏・甲・側面ラベルで別途学習する。MobileNetV3-smallの固定特徴から独自分類層を学習し、validationで温度と閾値を設定した後、独立testで精度を測る。近接画像を分割間で分離するが人物IDがないため、人物が独立した評価とは断言しない。

表情はPyTorch顔サンプルの小型PFLD68を使用する。目の閉じ、口の開き、鼻の位置、頬の輪郭幅変化を算出する。頬膨らみは未学習であり、単なる幅を頬膨らみの正解としない。

## 使い方

Sitesの専用URLでローカル動画を選ぶ。全身が小さい映像は人物の範囲をドラッグで囲む。「足のアップ」では片足を囲んで足裏・甲・側面を分類する。「顔のアップ」では顔を囲んで表情を読む。モデルを読み込んで開始し、画面下の実測fps・全身推論ms・全処理p95を確認する。

端末内のVRMを選ぶと、検出骨格と目閉じ・口開きを反映する。付属アバターやダミー動作は同梱しない。ずんだもん公式モデルは https://booth.pm/ja/items/3733351 から利用規約を確認して取得できる。

カメラは最大3入力を共通のモデルセッションで順に処理する。推論頻度は全入力合計の上限であり、各カメラ60fpsやGPUの使用率保証ではない。顔は最大10回/秒、足の向きは最大5回/秒に抑える。ブラウザでは元映像の解像度を落とさず、必要な部位だけモデル入力サイズへ切り出す。

## 非公開動画の扱い

添付動画、そのフレーム、切り抜き、特徴キャッシュ、動画由来ランドマークはGitHub・Sites・Hugging Face・Actions artifactsに一切含めない。学習には公開画像だけを使う。非公開動画の検証にはActionsの `PRIVATE_VIDEO_URL` secretを用い、runnerの一時領域へ取得してheadless Nodeへ渡した後に削除する。secretが未設定なら未実行として記録し、公開fixtureの結果で代用しない。URLをworkflow inputやソースへ書かない。ブラウザ入力はFile APIとObject URL、Canvasを使い、外部へ送信しない。

同一サイトの自己所有閲覧設定を維持する。private入力のディレクトリはGit管理対象から分離し、配布ファイルを列挙して動画・画像・学習キャッシュがないことを確認する。

## 3D・SteamVR連携

59点は2Dモデルである。未校正時の表示は画像平面上で観測できる動きが基準であり、メートル単位の実測ではない。2D座標だけからの不定な前後向き最適化は既定で無効。単眼の絶対root移動は現在時刻の独立したdepth計測がない限り抑制し、時刻同期済み複数視点では三角測量する。カメラ校正JSONは元のgeometry/calibrationの契約を用い、入力IDはcamera-1、camera-2、camera-3。

複数入力の時刻が一致するとの仮定を置かない。校正JSONのsourceSyncに入力ごとのoffsetMs、uncertaintyMs、confirmedを明示した場合だけ同期候補にする。別々に撮影した3本の動画は多視点の実測証拠ではない。

ローカルブリッジはserver/bridge.mjs。Node.js22以上でserver/の依存をnpm ci --ignore-scriptsで取得し、node server/bridge.mjsを起動する。表示された接続トークンを画面へ入力する。Sitesから接続する場合は起動引数--originへ専用Sitesの正確なOriginを渡す。ブラウザのloopback接続制限がある場合はブリッジが配信するローカルページを使う。

ルーム校正を読み込んだ有効な腰・両足の位置と回転のみ、VMTの127.0.0.1:39570へOSC送出する。未校正・欠測・古い観測は無効化する。目と口はVRM表示へ反映するが、任意のVRChatアバターへのOSC表情マッピングは本PoCでは自動設定しない。SteamVR/HMD/VRChat実機動作は環境での確認が必要であり、CIのUDP送信を実機達成としない。

## 再現

先行モデルはGitHub Actionsでtorch2.6.0 CPU、torchvision0.21.0、onnx1.17.0、onnxruntime1.20.1を使って学習・評価した。新しい学習用のColabキットはCUDA対応PyTorchを維持し、実際のバージョン・GPU型番を結果へ記録する。配布前の実行確認とNode推論は引き続きActionsで行う。ブラウザ配信のONNX Runtime WebとNodeは1.22.0。公開チェックポイントと元ONNXをSHA-256で確認し、PyTorch→ONNXの実画像比較を行う。

配布モデルは同じActions内のheadless Nodeで実動画をデコードし、実ONNX推論、姿勢ガード、実VRMのhumanoid・spring bone更新を実行する。旧133点・旧59点・新59点は同じフレーム・同じ人物範囲で比較し、速度とモデル同士の一致度を分けて記録する。一致度は正解精度ではない。欠測・画像外の点を注入したchallengeも実動画の集計から分離する。精度評価には独立の公開画像を用い、動画・スクリーンショット・フレーム座標列を成果物へ入れない。

training/が全身モデル、parts/が顔・足、dist/がブラウザ、server/がローカルVMT送出、ci/が配布準備。元のphase3ブランチd3917f976394eea95451800dde4c9e219ad3e3c5から幾何、VRM、ローカル送出の共通モジュールを独立コピーして再利用している。

## ライセンスと出典

新規コードはリポジトリのUnlicense。元モデルはApache-2.0のDWPose/MMPose。COCO-WholeBody注釈はCC BY-NC4.0の研究・非商用条件を持つ。FootNetはCC BY4.0、ImageNet pretrained MobileNetV3はtorchvision配布。PFLDの原配布元・固定ハッシュ・権利確認の範囲はHFのmodel cardに記録し、独自学習物やMIT許諾済みとは表示しない。詳細はtraining/sources.json、公開マニフェストとmodel cardに記録する。
