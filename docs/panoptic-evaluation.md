# Panoptic実動画による2視点・時系列評価

対象はCMU Panoptic Studioの`171204_pose1_sample`。取得と推論はGitHub Actionsで実行する。最初に到達性・配布容量・注釈形式を小さなHTTP Rangeリクエストで確認し、動画本体を丸ごと取得する前に結果を記録する。

## 実行状況：配布元へ到達できず、精度評価は未実行

[Actions run 41376282352](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41376282352)と[run 41376400694](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41376400694)ではprobe段階で失敗した。CMUのHTTPSは証明書の鍵強度エラーまたは接続リセット、HTTPはタイムアウト、SNUのHTTPは502、HTTPSは接続リセットとなった。注釈と同期動画を取得できていないため、このスクリプトから実測したPanopticの精度値はない。証明書検証を緩和して取得する処理は追加していない。

以下は再実行可能な評価の設計であり、実測成功の報告ではない。配布元への接続が回復した場合に明示起動する別workflowへ分離し、現在の実画像手部評価には[HanCo Tester](model-research.md)を使う。

## 根拠とデータ契約

[公式toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)は同期HD動画、キャリブレーション、19点の全身3Dと左右各21点の手3Dを案内している。CMUの配布元が応答しない場合のSNU公式ミラーも同リポジトリで案内されている。

[座標仕様](https://domedb.perception.cs.cmu.edu/develop/tools.html)では`x=K(RX+t)`、歪みはOpenCV形式`[k1,k2,p1,p2,k3]`、1単位は1cm。同じHDフレーム番号は同期した撮影時刻を表す。動画の先頭フレームは番号0とし、GTファイル名の番号と一致させる。

身体19点の定義はCOCO17と順番が異なる。neck/body-centerを除外し、nose/eyes/ears/shoulders/elbows/wrists/hips/knees/anklesを明示的に写像する。手の21点はOpenPose/MediaPipeの手首から5指へ続く順序で対応させる。足の甲と足指のGTはなく、この評価から足指精度を主張しない。

| 変換 | 実装 |
|---|---|
| COCO17からPanoptic19への対応 | `[1,15,17,16,18,3,9,4,10,5,11,6,12,7,13,8,14]` |
| 手 | 左手は133点中91〜111、右手は112〜132 |
| 世界座標 | `XYZ_m = XYZ_cm / 100`。GTに対する平行移動・回転・スケール合わせはしない |
| アプリのカメラ回転 | `worldFromCamera.rotation = transpose(R)` |
| アプリのカメラ位置 | `worldFromCamera.translation = -transpose(R) * t / 100` |
| 時刻 | `frameIndex * 1000 / (30000 / 1001)` ms |

## Probe

`python scripts/evaluate-panoptic.py --probe --out artifacts/panoptic`

標準ライブラリだけで実行できる。各動画は先頭4096 bytes、各GT tarは最大4MiBだけ読み、Content-Length/Content-Range・prefix hash・JSONの型/配列長を`probe.json`へ記録する。取得や形式確認が失敗すれば終了コード1を返す。probeの成功は推論精度の成功を意味しない。

## 実動画の評価

```sh
python scripts/evaluate-panoptic.py --models rtmw-l --frames 40 --out artifacts/panoptic
```

既定はHDカメラ`00_00`と`00_08`の連続40フレーム。比較する条件は同じモデルの全身出力と手ROI再推論後の出力で、両条件は同じデコード画像を使う。`--models rtmpose-s,rtmw-l`ならモデル間も比較できる。30〜60フレームを指定でき、フレーム開始位置、人物ID、配布元の明示指定にも対応する。フレーム・人物の既定選択は推論誤差を見て変更しない。

Python側は既存の`evaluate.py`からONNXの読込、前後処理、手ROI補正を再利用する。必要な依存は同じNumPy、OpenCV、ONNX Runtimeと、動画デコード用ffmpeg。動画はフレーム0から読み、必要な枚数でデコードを止める。ffmpegはMP4索引を取得するためHTTP Rangeで後方へアクセスする場合がある。元動画の全ファイルを保存せず、選んだフレームだけ実行環境のキャッシュへ置く。GT tarは順に読み、必要なJSON以外を展開しない。各アーカイブの読込上限は256MiBで、上限超過や不足データを成功扱いにしない。

得た2D推論を、実行環境内の`evaluation-input.json`を通して`verify-geometry.mjs`へ渡す。同スクリプトがアプリ本体の`triangulateJoint`と`OcclusionTracker`を呼ぶため、Pythonで別実装した3D復元器の結果ではない。通常観測と、中央8フレームだけ両手の観測を取り除いた条件を、独立したトラッカーで測定する。人工欠落は画像推論の後に行う。人物画像への自然な遮蔽を再現した実験ではなく、同じ実運動で観測を失った時の短期補完を測る実験である。

### 指標

身体17点と手42点を分けて集計する。3D GTは身体の再構成confidenceが0.1超の点、手は有限で非ゼロの3D点を対象とし、21点のscore配列が存在する場合はさらに0.1超を要求する。この手GT条件は実行レポートへ記録する。GTのqualityと推論器のconfidenceは別の値として扱う。

| 指標 | 分母と意味 |
|---|---|
| Coverage | 有効GT点のうち3D出力が存在する割合 |
| MPJPE | 3D出力があるGT点に対する世界座標での平均誤差、mm。必ずcoverageを併記する |
| PCK20 / 50 / 100mm | 有効GT点すべてを分母とし、欠測を失敗に数えた閾値内の割合 |
| 生の三角測量と因果トラッカー | 補完前と補完後を両方掲載する。全区間と中央の欠落区間も分ける |
| 2D位置誤差 | 画像範囲内に投影されるGT点の平均・中央値・95%点と5/10px以内の割合。低い推論scoreを理由に除外しない |
| CPU推論時間 | 実行したONNXモデル単体の実測値。ブラウザやGPUのFPSには換算しない |

## 実測の判定範囲

人物の関連付けと人物ROIをGTの投影点から与える場合はoracle条件として記載する。実モデルへ渡すのは画像ROIだけで、手ROIは全身モデルの予測点から作る。GTの手・身体座標を推論入力へ渡さない。GTから得たROIを使うため、人物検出や複数人物の自動対応付けの精度は評価しない。

自然動画の全関節3D誤差と、観測を意図的に欠落させた区間の回復誤差は別々に扱う。推定器のscoreを自然遮蔽の正解ラベルへ読み替えない。欠測を除いた誤差だけでなく、GTが有効な全関節に対する出力率と閾値内の割合を併記する。

このGTはPanopticの多数カメラから再構成した点であり、光学マーカーの測定値ではない。1本の短い動作だけで一般的な利用条件の精度は確定しない。また、本評価は校正済み同期2視点の能力を測る。単眼や未校正カメラで同じ精度が得られるとは主張しない。

## 保存

[データ提供条件](https://domedb.perception.cs.cmu.edu/)は研究目的での使用を許可し、元データの再配布を制限している。CIの公開成果物には元動画、画像、GT座標列、キャリブレーション本体を含めず、取得元・hash・評価条件・誤差統計・推定値を記録する。データキャッシュは実行環境内で使用する。

公開する`artifacts/panoptic/report.json`は取得情報、2D誤差と3D集計、`geometry.json`は製品JavaScriptでの3D評価、`observations.json`はモデルの2D推定列を収録する。実行前には成功した精度値として扱わず、Actionsの完了結果に合わせて記載する。
