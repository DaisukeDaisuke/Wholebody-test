# VRC Compact 59

VRC向けに必要な関節と表情へ絞った軽量モデルの研究PoC。独立ブランチ compact-vrc-20261102-7fe4fac1 だけで開発する。

## 現行版の構成

既定構成は標準Compact59のXY認識と、低頻度の学習済み3D奥行き。単独3DモデルのXYが元の59点より不正確だったため、信頼できる現在の59点XYを維持する。最初の低解像度・縮小head実験は独立testでPCK@0.05が約9.9ポイント低下し、今回のINT8も手が約5.33ポイント低下したため不採用。実験候補は既定へ昇格させない。

返却されたColab ZIPは学習失敗のJSONだけで、ONNX・checkpointがなく、破棄された環境の重みは復元できない。返却処理を修正し、途中の回復ONNX・checkpoint、失敗診断、非0終了をActionsで確認した。新しいGPU学習に成功したとは扱わない。

Hugging Faceへの公開とヘッドレスNode.js推論は `compact.yml` で行う。`HF_TOKEN` はActionsの公開ステップだけに渡す。ブラウザとNodeは、公開された重みの固定commit SHAを指定したHFの `/resolve/` URLから認証なしで取得し、SHA-256を検証する。Sitesには重みを同梱しない。`fast59.yml` の長時間CPU学習は自動起動を止めた。

動画とVRMは常に横並び。単眼の身体には学習済み相対XYZを反映し、現在のXYと合わせた透視投影から相対root移動を推定する。未校正の絶対位置を計測済みトラッカーとして送らない。観測できた点、消えた点、駆動できたボーンを別に数える。短い表示保持や角速度制限で回転量が減っても、それを認識精度の改善とは扱わない。

既定構成はCPU/WASMで、59点最大15Hz、奥行き最大5Hz、ロスト時の人物再検出最大2Hz。顔は最大10Hz、足は最大5Hzで有効な領域があるときだけ推論する。VRMは30Hzで更新する。2D比較モードではWebGPUも選べる。頻度制限と1回の推論時間、CPU時間とGPU使用率は区別する。

## Colab GPU 学習

[修正版GPUノートブックを開く](https://colab.research.google.com/github/DaisukeDaisuke/Wholebody-test/blob/8cda5c69dcef8c1dde62f3ef91b2fa50ceac3696/colab/VRC_Compact59_GPU_7fe4fac1.ipynb)。修正版学習用ZIPを選び、表示されたstep・batch・microbatchで実行する。途中のONNXとcheckpointも回収し、完了条件を満たす返却は `VRC_Compact59_Colab_Result_7fe4fac1.zip`、失敗時は `VRC_Compact59_Colab_Diagnostic_7fe4fac1.zip` と区別する。詳しい構成は[colab/README-ja.md](colab/README-ja.md)。

[Actions run 41398850065](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41398850065) で、CPUの実学習・ONNX書き出し・INT8校正と、失敗時の回復ONNX回収・非0終了まで確認した。これは完全なCUDA学習や新モデルの精度達成を意味しない。

## 実動画とVRMの検証

以前の骨ローカル回転p95の減少を「安定性・追従の改善」とした評価は撤回した。認識できずに骨が動かなくなっても回転量は減る。新しい評価は、現在の可視関節、実際に駆動した四肢、ロスト継続、動作の保持を全身・部分表示で分け、正解ラベルに対する2D精度も別に測る。実行がsuccessでも品質の `deploymentReadiness.passed` がfalseなら合格とはしない。最新の数値・未達条件は[RESULTS.md](RESULTS.md)に記載する。

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

Sitesの専用URLでローカル動画を選ぶ。全身が小さい映像は人物の範囲をドラッグで囲む。「足のアップ」では片足を囲んで足裏・甲・側面を分類する。「顔のアップ」では顔を囲んで表情を読む。モデルを読み込んで開始し、画面下の実測fps・全身推論ms・推論処理p95を確認する。GPU描画を含む総遅延とは区別する。

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
