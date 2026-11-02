# 実運動系列での遮蔽補完評価 — FreeMoCap / Qualisys

評価日: 2026-11-02。対象コード: `e9bc84483c88884c7b88ed9589eb069a6070a903`。

**記録済み身体系列で、左右の肘と手首の観測を最大 266.7 ms 消した場合、製品の因果補完は最後の座標を保持する方法より平均誤差を小さくした。** RTMPose 由来の入力を独立に収録された Qualisys マーカー基準へ比較した欠測区間 160 点では、MPJPE は **57.98 mm → 33.35 mm（42.48% 減）**、50 mm 以内の割合は **47.50% → 81.25%** だった。両方法とも 160 / 160 点を出力した。[1]

一方、1.5 m/s 条件の手首、2.5 m/s 条件の欠測末尾には保持法より大きな誤差がある。平均改善をすべての部位・時刻の改善とは扱わない。この結果を受けた製品の閾値変更や同じ系列での再調整は行っていない。

## この評価で通した経路

`公開された aligned 3d_xyz → dist/tracking.js の OcclusionTracker → Qualisys 参照との誤差集計`

画像推論、カメラ校正、`dist/geometry.js` の三角測量、ブラウザ映像入力は、この身体系列の評価経路に入っていない。FreeMoCap の配布物には原動画が含まれず、RTMPose 由来の三次元系列と Qualisys 由来の参照系列を使用した。今回の PoC のカメラから出力までの精度とは区別する。[2]

Qualisys は別のマーカーシステムで収録された参照だが、関節中心はマーカーから計算された解剖学的近似であり、関節位置の完全な真値ではない。配布系列には、系列全体を使う剛体整列、時刻整合、補間、フィルタ処理が既に入っている。評価対象の `OcclusionTracker` は未来フレームを読まないが、配布前処理を含む全体をオンラインの因果経路とは呼ばない。[3]

## データと固定した評価条件

| 項目 | 内容 |
|---|---|
| 配布物 | FreeMoCap Validation Dataset v1.0、DOI `10.5281/zenodo.23041223`、CC BY 4.0 |
| 被験者・試行 | `sub-001 / task-treadmill_trial-01` のみ |
| 歩行速度 | 0.5、1.0、1.5、2.0、2.5 m/s。公式設定の各速度区間から先頭 300 連続フレームを選択 |
| 時間 | 30 Hz、各速度 10 秒、計 1,500 フレーム。誤差を見て区間を選んでいない |
| 身体点 | 左右の肩・肘・手首・股・膝・足首の 12 点、COCO ID 5–16 |
| 元の座標 | `model=rtmpose.body` または `qualisys.body`、`trajectory=3d_xyz`。`rigid_3d_xyz` は選択しない |
| 単位・軸 | mm から m へ一律に 0.001 倍し、全系列へ同じ回転 `[x,y,z] → [x,z,-y]` を適用 |
| 欠測 | 各 300 フレームの中央 8 フレームで左右の肘・手首、ID 7–10 の入力を除去 |
| 最後の観測からの経過 | 33.3、66.7、100.0、133.3、166.7、200.0、233.3、266.7 ms |
| 身体全体の分母 | 1,500 フレーム × 12 点 = 18,000 点。入力・参照とも選択点の非有限値は 0 |
| 欠測腕の分母 | **5 速度 × 8 フレーム × 4 点 = 160 点**。条件ごとには 32 点 |

速度と 30 Hz は固定した公式試行設定、mm 単位は公式の軌跡・RMSE 表示コードに従った。[4] 個別フレームの平行移動合わせ、回転合わせ、スケール合わせは追加していない。入力の `confidence=1` は「座標が存在する」という印であり、校正された可視確率ではない。

比較した方法は次のとおり。

- **製品補完:** 現行 `OcclusionTracker` を既定値のまま実行。観測がある肩から、過去に観測した骨の相対変位と長さ、減衰する相対速度を使って肘・手首を補完する。今回の欠測腕の出力 source は全 160 点とも `kinematicPrior`。
- **最後の座標を保持:** 関節ごとに最後に観測された三次元座標を保持する。未来参照や速度外挿をせず、身体点は 600 ms、手指は 250 ms、足点は 450 ms の期限で終了する。今回の ID 7–10 は身体点であり、比較する全欠測区間が 600 ms 以内に収まる。

MPJPE は利用可能な点と参照の三次元ユークリッド距離の平均。PCK50 は参照がある全点を分母にした「誤差 50 mm 以内」の割合で、欠測出力は成功点に数えない。今回は両方法の coverage が 100% のため、平均誤差の比較対象は同じ 160 点になる。推論・評価の実行は GitHub Actions のみで行い、結果 ZIP の SHA-256 照合と JSON の集計だけを手元で行った。

## 主結果: RTMPose 入力 → Qualisys 参照

人工的に観測を除去した腕 4 点だけの結果。[1]

| 歩行速度 | 製品補完 MPJPE | 保持 MPJPE | 製品補完 PCK50 | 保持 PCK50 | 両方法の出力数 |
|---|---:|---:|---:|---:|---:|
| 0.5 m/s | 26.77 mm | 46.42 mm | 93.75% | 62.50% | 32 / 32 |
| 1.0 m/s | 24.73 mm | 49.91 mm | 90.625% | 59.375% | 32 / 32 |
| 1.5 m/s | 33.78 mm | 39.06 mm | 81.25% | 75.00% | 32 / 32 |
| 2.0 m/s | 36.02 mm | 82.36 mm | 78.125% | 18.75% | 32 / 32 |
| 2.5 m/s | 45.46 mm | 72.17 mm | 62.50% | 21.875% | 32 / 32 |
| **全 160 点** | **33.35 mm** | **57.98 mm** | **81.25%** | **47.50%** | **160 / 160** |

各速度区間を通じた腕の平均では 5 / 5 条件で改善した。欠測を作らない場合、公開 RTMPose 身体 12 点と Qualisys の差は速度別に 23.07–27.92 mm だった。この元の誤差も残った状態での欠測区間比較である。製品が写真から 23–28 mm の身体精度を達成したという意味ではない。

## 局所的に保持法より悪化した部分

### 1.5 m/s: 手首の外挿が 8 フレームすべてで悪化

この区間の肘 2 点は 43.32 mm → 24.65 mm と改善したが、手首 2 点は **34.80 mm → 42.91 mm** へ悪化した。手首の PCK50 は 75.00% → 68.75%、誤差の 95 パーセンタイルは 91.67 mm → 123.09 mm だった。腕全体の平均だけでは、この末端の悪化が隠れる。

| 最終観測からの経過 | 製品補完: 手首平均 | 保持: 手首平均 |
|---:|---:|---:|
| 33.3 ms | 12.84 mm | 7.20 mm |
| 66.7 ms | 17.15 mm | 9.88 mm |
| 100.0 ms | 24.21 mm | 13.57 mm |
| 133.3 ms | 33.75 mm | 21.70 mm |
| 166.7 ms | 44.84 mm | 34.10 mm |
| 200.0 ms | 57.67 mm | 49.62 mm |
| 233.3 ms | 70.62 mm | 64.60 mm |
| 266.7 ms | 82.23 mm | 77.74 mm |

同じ時刻で入力を欠測させなかった場合の手首誤差は 10.70–14.10 mm。大きな誤差増加は観測除去後に発生している。一方、誤差集計だけでは、肘から手首への誤差伝播、相対速度の推定誤差、腕の曲率変化の寄与を分離できない。

### 2.5 m/s: 欠測末尾で優劣が逆転

33.3–200.0 ms の腕平均では製品補完の誤差が小さく、233.3 ms では 76.84 mm / 77.02 mm とほぼ同じだった。266.7 ms では **製品補完 96.88 mm、保持 57.84 mm** となり、優劣が逆転した。

Qualisys 自体を入力にした診断条件でも、同じ末尾で **110.49 mm / 64.66 mm** と逆転した。元の RTMPose 推定誤差だけが原因とは考えにくく、最後の運動方向を延長する補完が実際の運動の変化に合わなくなる可能性を示している。これは集計からの推論であり、運動の向きや曲率を直接確認した結論ではない。

今回の実装では観測が戻るまで誤差を検出できない。表示される confidence や出力点があることを、センチメートル単位の正確さの保証として扱わない。

## 診断条件: Qualisys 入力 → 同じ Qualisys 参照

同じ人工欠測を参照系列そのものへ入れ、入力推定器の誤差を外した場合の補完誤差を測った。この値はマーカーレス推定精度を示すものではない。[1]

| 歩行速度 | 製品補完 MPJPE | 保持 MPJPE |
|---|---:|---:|
| 0.5 m/s | 9.43 mm | 37.78 mm |
| 1.0 m/s | 10.83 mm | 38.44 mm |
| 1.5 m/s | 31.64 mm | 39.12 mm |
| 2.0 m/s | 28.60 mm | 86.14 mm |
| 2.5 m/s | 50.85 mm | 75.21 mm |
| **全 160 点** | **26.27 mm** | **55.34 mm** |

全 160 点で平均誤差は 52.53% 減り、PCK50 は 52.50% → 83.75% となった。欠測なしの保持法は同じ座標を返すため誤差 0、製品の平滑化を通した身体平均は 0.02–0.06 mm だった。

## 判断と証拠の範囲

今回の比較は、短い身体観測欠損に対する骨長・親関節を使う補完に、単純保持を上回る場合があることを支持する。平均改善があり、局所失敗もあるため、この 1 人・1 試行へ合わせて減衰時定数などを変更する理由にはしない。製品の既定値は維持した。

改善する場合の次の論点は、腕の運動変化を過去の観測から検出して外挿の強さを制限できるか、肘と手首の相対運動の不確かさを扱えるかである。それを判断するには別被験者・別試行を残した評価が必要であり、今回の 160 点を調整用と検証用の両方には使わない。

この評価からは、手指 21 点、足の甲、各足指、実画像での遮蔽、全身が同時に隠れた場合、266.7 ms より長い欠測、ユーザーのカメラ条件、実時間レイテンシは判断できない。腕の元入力を削除する実験であり、映像を遮って再推論した実験とは区別する。

## 再現記録

- GitHub Actions [run 41378383880 / job 125063561669](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41378383880/job/125063561669): 成功。
- [評価 artifact 13111389565](https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/41378383880/artifacts/13111389565)。保持期間は実行から 7 日。集計 JSON 5 本と取得来歴のみを含み、画像・元座標は含まない。
- Artifact ZIP SHA-256: `a4e19b83d5c663d15fcc4b02d2713b60a84f8b8d6dadab122f5a69a5ebc8bbc4`。ダウンロード後に GitHub の digest と一致を確認。
- 元データ ZIP: `freemocap_validation_dataset_v1.0_sub-001.zip`、252,005,355 bytes。SHA-256: `cd1c13799af9272a29b977f47d6ec7c9dd754644fb370774a78c10bcd78a9766`。配布元の `SHA256SUMS.txt` と Actions 内で照合。
- 配布設定・実装参照 commit: `freemocap/validation@39207e112a9fb79ee71e15f061bd98b6ac4cbdce`。
- Actions の入力作成: `python scripts/fetch_freemocap.py --subjects 1 --frames-per-condition 300 --out-dir .evaluation-cache/freemocap --report artifacts/freemocap-fetch.json`。
- Actions の集計: `node scripts/verify-geometry.mjs --sequence-input-report artifacts/freemocap-fetch.json --sequence-out-dir artifacts/freemocap`。

## 一次資料

[1] 上記 Actions artifact 内 `freemocap/*-metrics.json` および `freemocap-fetch.json`。本書の測定数値はこの実行結果から集計。

[2] [FreeMoCap Validation README](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/README.md)、[公開データ DOI](https://doi.org/10.5281/zenodo.23041223)。原動画を含まない aligned 3D trajectory 配布と CC BY 4.0 を明記。

[3] 公式の [剛体整列](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/validation/steps/spatial_alignment/core/alignment_utils.py)、[Qualisys 時間再標本化と関節中心](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/validation/steps/temporal_alignment/core/qualisys_processing.py)、[補間・フィルタ処理](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/validation/steps/temporal_alignment/core/postprocessing.py)。

[4] [試行の速度区間と 30 Hz 設定](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/configs/sub-001/task-treadmill_trial-01.yaml)、[公式軌跡表示の m/s・mm 表記](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/plotting/gait/joint_trajectories/joint_trajectories_xyz_plots.py)、[Qualisys の関節名定義](https://github.com/freemocap/validation/blob/39207e112a9fb79ee71e15f061bd98b6ac4cbdce/qualisys_markerset/qualisys_model_info.yaml)。
