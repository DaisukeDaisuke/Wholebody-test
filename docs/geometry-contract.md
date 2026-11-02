# フェーズ2 幾何・遮蔽推定の契約
作業ID: `732e048e-dba4-468e-bdd2-6026afc8218d`
## 座標と校正
カメラ座標は x=右、y=下、z=前、距離単位はメートル。世界座標は `worldFromCamera` が定める右手系で、回転は row-major の3×3、translation はカメラ中心の世界座標。画像左右反転やリサイズは観測点と内部行列の両方に反映させる。
```json
{"id":"camera-a","width":1920,"height":1080,"fx":1200,"fy":1200,"cx":960,"cy":540,"skew":0,"distortion":[0,0,0,0,0],"worldFromCamera":{"rotation":[1,0,0,0,1,0,0,0,1],"translation":[0,0,0]},"calibrated":true,"intrinsicsCalibrated":true}
```
この数値はAPIの例であり実測校正ではない。`calibrated:true` は内部・外部パラメータが測定済みの場合だけ設定する。歪み係数順は `[k1,k2,p1,p2,k3]`。魚眼モデルは非対応で、事前に正しい投影モデルで補正する必要がある。
## geometry.js
`projectPoint(xyz,camera)` は `{u,v,depthM}`、カメラ後方点は `null` を返す。`undistortPixel(u,v,camera)` は正規化画像座標、`pixelToWorldRay(u,v,camera)` は世界射線を返す。
`triangulateJoint(observations,cameras,options)` の観測は `{sourceId,timeMs,u,v,score,observable?}`。同じ関節の観測だけを渡す。cameras は id キーのオブジェクト、Map、配列を許容する。出力は `{valid,xyz,confidence,source,reprojectionErrorPx,rayAngleDeg,sources,timeMs,timeSpreadMs}`。無効時は `xyz:null,source:'missing',reason`。
同じカメラの重複を除き、既定で撮影時刻差25ms以下、視線交角2度以上、基線長2cm以上、再投影誤差8px以下の観測だけを用いる。3視点以上は観測対から候補を生成して外れ値を除く。信頼度は幾何品質の指標であり、校正済みの正解確率ではない。
`allowApproximateCalibration:true` を明示すると、`calibrationStatus:'approximate'` のカメラも利用する。この場合のsourceは `estimatedMultiView`、distanceStatusも `estimatedMultiView`、信頼度は半減する。FOVから仮定した内部行列を実測校正と扱わない。
`triangulateFrame([{sourceId,timeMs,keypoints}],cameras,options)` は同じidを集約し、各関節の三角測量結果を返す。時刻は受信時刻ではなく、共通時計に対応づけた撮影時刻を渡す。映像の転送遅延やデコード遅延が未知なら実測同時刻と扱わない。
`estimateRootDepth(keypoints,camera,boneLengths,options)` の点は `{id,u,v,score,relativeZm?}`。boneLengths は `{'5-7':0.30,'7-9':0.25,...}`。複数長骨の長さ残差をHuber損失で最小化し、出力は `{valid,depthM,joints,distanceStatus,fitErrorM,confidence,boneCount,coordinateFrame,boneLengthSource}`。BBox高さを距離計算に使わない。最低3骨が必要で、探索範囲境界・矛盾の大きい骨長・深度不足は無効にする。
`relativeZm` がない2Dモデルは既定で単眼3Dに変換しない。明示的に `allowPlanarPrior:true` を指定した場合のみ平面事前分布を使用し、`source:'planarPrior'`、`distanceStatus:'estimatedPlanarPrior'`、低い信頼度を返す。これは学習済み3D姿勢推定の代替精度を保証しない。
`anthropometricBoneLengths(heightM)` は身長からの初期比率を生成する。本人の実測骨長ではないため、`boneLengthSource:'measured'` を指定してはならない。未実測内部パラメータでの距離は `estimatedLowConfidence`。
`BODY_BONES`、`SKELETON_EDGES` は COCO-WholeBody 133点の身体・足・手の接続を公開する。顔の詳細接続は含めない。
`calibratePlanarCamera({imagePoints,worldPoints,intrinsics,id,maximumReprojectionErrorPx:8})` は既知の床矩形等の対応4点以上からカメラ外部姿勢を求める。imagePoints は `[[u,v],...]`、worldPoints は `[[x,0,z],...]`、距離単位m、床Y上。4点の順は両配列で対応させる。例は `[[0,0,0],[幅,0,0],[幅,0,奥行],[0,0,奥行]]`。出力は `{valid,camera,reprojectionErrorPx,maximumPointErrorPx,calibrationStatus,pointCount}`。失敗は `{valid:false,camera:null,reason}`。
歪み補正後の対応から正規化ホモグラフィを解き、回転を正規直交化した後、原画像再投影誤差を最小化する。共線点・床以外の点・カメラ後方・床より下のカメラ・再投影残差超過は拒否する。カメラは床より上にあることを前提とする。4点の再投影が一致しても未知Kの誤差が消えるわけではない。
intrinsics に `intrinsicsCalibrated:true` を指定した場合だけ、返すcameraの `calibrated:true` とする。FOV由来の暫定Kなら `intrinsicsCalibrated:false` とし、返るcameraも `calibrated:false,calibrationStatus:'approximate'`。床矩形の測定によりスケールの基準は得られるが、暫定Kによる3D誤差は残る。
## tracking.js
`new OcclusionTracker(options)` の `update({timeMs,joints})` は `{timeMs,joints,root,feet,stats}` を返す。入力関節は `{id,xyz:[x,y,z],confidence,source,valid?}`。許可する現在入力sourceは `triangulated`、`estimatedMultiView`、`monocular`、`planarPrior`、`model3d`。過去の予測を再入力して観測寿命を延ばさない。
出力jointsは133点。`source:'predicted'` は減衰する速度外挿、`source:'kinematicPrior'` は直近の骨長・親関節位置・掌座標系による補完。入力がある関節はそれぞれのsourceを維持する。`observed:true` は三角測量された3Dだけに付く。`ageMs` と `lastObservedTimeMs` は予測フレームでは更新しない。
既定の自由外挿期限は身体600ms、手250ms、足450ms。観測できる親や掌で拘束する補完も最大1200msで終了し、信頼度は時間で減衰する。最初から一度も推定できない関節や、期限後の関節は `missing` とする。完全遮蔽中の未知の指曲げを実測として出さない。
逆方向の動画シークで履歴を初期化する。入力を切り替える場合にも `reset()` を呼ぶ。未来フレームを参照する平滑化は行わない。
`deriveFootFrames(joints)` は左右の踵・母趾・小趾・足首から足の向きと足の甲の代理位置 `instepProxy` を出す。標準133点には各足指の関節がないため、`toeArticulationAvailable:false` を必ず返す。足の甲の表面形状や5本の足指の曲げ角を測れたという意味ではない。
`mergeJointEstimates(primary,fallback)` は有効なprimaryを優先し、欠測点にfallbackを使う。両方は同一の座標系に変換してから渡す。
## view3d.js
`new PoseView(canvas)` は外部依存なしのCanvas骨格表示。`render(result,cameras,{coordinateFrame:'world'|'camera'|'rootRelative',statusText})` にtracker結果とカメラ群を渡す。`clear(text)` は未推論等の空表示、`resetView()` は現在姿勢へ視点を戻し、`dispose()` はイベントを解除する。
world表示は床Y上で、初回だけ自動中心を決める。その後は床と視点中心を固定して人物の移動を表示する。camera表示は入力のYを描画投影時だけ反転する。rootRelative表示だけは骨盤へ追従するが、入力自体は世界Y上を想定する。元の関節座標は変更しない。
ドラッグで回転、ホイールで拡大、ダブルクリックで再中心。キーボード矢印・±・Homeにも対応する。未推論と、有効3D点がない欠測状態は別表示。足の甲の菱形と矢印は `instepProxy` と足方向であり、足指関節の推定表示ではない。
## 検証境界
実機LAN映像・実測校正・正解3Dとの比較が済むまで、二端末での精度達成とはしない。Actionsの既知3D投影による検査は幾何実装の整合性検証であり、実映像上の推論精度と分ける。
