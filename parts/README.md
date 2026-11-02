# 顔と足の軽量部位モデル

`python parts/prepare.py --mode all --output artifacts/parts --report artifacts/parts --cache .cache/parts` をGitHub Actionsで実行する。学習・評価・推論はActions環境でだけ有効であり、動画を読み込む処理は持たない。

## 足の向き

FootNet v1.0の公開191画像にある`sole/top/side`フォルダを足裏・足の甲・側面の3クラスとして使う。TorchVisionのImageNet学習済みMobileNetV3-Smallの特徴抽出器を固定し、576→96→3の独自MLPを新規学習する。入力は既知の足領域を正方形に切り出した128×128 RGB、ImageNet正規化。出力は3クラス確率。

完全一致画像と近いdHashの画像を同じ群にまとめ、元の左右×3方向フォルダ内で画像群を60/20/20に分割する。学習済みhead、温度、採用閾値の選択にtest集合を使わない。test集合は固定したモデルの最終評価に一度だけ用いる。subject IDは配布されていないため、人物単位の独立性や未知人物への一般化を達成したとはしない。

評価の足領域は正解maskの外接矩形であり、足検出の精度ではない。ブラウザでは手動ROIまたはCompact59の足首・踵・つま先の予測からROIを作る。背景クラスは学習していないため、一般画像から足を発見するモデルとして使わない。3方向ラベルは足指の屈曲や足首の3D角度を教師として含まない。

データ原典：[FootNet, DOI 10.5281/zenodo.20457253](https://zenodo.org/records/20457253)。Adith Vijay、Akash Prabhune、Vinay R Srihari、Akhila Rayampalli、CC BY 4.0。ZIPのMD5と画像・maskの配布SHA-256を検証する。画像・crop・既存の重いsegmentation checkpointを成果物へ含めない。

## 表情

[Cunjian ChenのPyTorch Face Landmark](https://github.com/cunjian/pytorch_face_landmark)のPFLD68 ONNXを作者の固定revisionから取得する。112×112 RGBを255で割った入力を使う。これは作者のONNXカメラサンプルに従う。別のPyTorchバッチサンプルではBGRを使うため、採用した前処理を評価記録にも明示する。

重みを新規学習したとは表示しない。上流モデルの明確な再配布ライセンスは確認できないため、このリポジトリやSitesに重みをコピーせず、ブラウザが作者の固定URLを取得する。実際に使うbytesのGit blob SHAとSHA-256を検証する。新規コードと上流重みの権利を混同しない。

目は6点のEARから閉じ具合、口は内唇8点から開き具合、頬は左右輪郭の幅の変化、鼻は鼻先点を取り出す。頬幅は2Dの代理指標であり、頬の膨らみを学習した判定ではない。`cheekPuff`は未推定の`null`、`cheekWidthChange`はニュートラル校正時のみ返す。顔検出は含まないので顔ROIが必要。

公開COCO-WholeBody診断画像の実ラベルに対してランドマークNMEとEAR・口開き比率の誤差を測る。目の開閉や頬膨らみの分類ラベルはないため、その分類精度は測定済みとしない。顔に見えない範囲を入力した時にPFLDが返すランドマークを有効な表情と扱わない。

## ブラウザ契約

`PartModels(ort, manifest, {executionProviders:['wasm']})`は標準でCPU側のWASMを使い、GPU負荷を抑える。`load('face')`と`load('foot')`で必要なモデルだけ初期化する。`runFace(source,[x,y,w,h],options)`、`runFoot(source,[x,y,w,h],options)`へ元動画の画素座標ROIを渡す。`options.roiConfidence`には自動ROIの関節信頼度を渡す。手動で既知の顔・足を選んだ時のみ1を使う。

`faceRoiFrom59(points)`と`footRoiFrom59(points,side)`は`{x,y,score}`配列からROIを計算する。戻り値は`{roi:[x,y,w,h],confidence}`、可視点が不足すると`null`。モデルの分類確率とランドマークの観測信頼度は別に保持する。

動画・画像ピクセル・座標を外部へ送る処理はない。外部通信は公開モデルとmanifestのGETのみ。モデル取得時のSHA-256検証とCache Storageは、プライベート動画を保存・送信しない。
