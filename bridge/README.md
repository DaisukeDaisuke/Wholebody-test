# ローカルSteamVR送出
Node.js 20以降の組み込み機能だけを使います。インストール作業・npm依存はありません。動画・画像は送信せず、ブラウザが生成した姿勢JSONをローカルのVMTへ渡します。
```sh
node bridge/server.mjs
```
表示された `http://127.0.0.1:8765/` で同じアプリが開きます。画面のブリッジURLに `http://127.0.0.1:8765`、トークン欄にターミナルの `Bridge token` を入力します。トークンを固定する場合は環境変数 `TRACKER_TOKEN` を設定してください。
公開サイトから接続するときは、そのURLのオリジンを明示します。Chromeのローカルネットワーク許可やHTTPS→HTTP制限がある環境では、上記ローカルページを使用してください。
```sh
node bridge/server.mjs --origin https://YOUR-PUBLISHED-SITE
```
UDP送出先は `127.0.0.1:39570`、VMTメッセージは `/VMT/Room/Unity` です。トラッカー番号は腰0・左足1・右足2。Three.js右手系の位置・回転をUnity左手系へ変換します。500ms姿勢が届かない場合、欠測、停止時にはトラッカーを無効化します。VMT Driver/ManagerとSteamVRの実機導入、部屋原点合わせ、VRChatのFBT校正は必要です。ソフトウェアだけで実機合格とはしません。
既定では、カメラと人物尺度が校正済みの有効トラッカーだけを送ります。未校正単眼推定を試す場合は明示して起動します。
```sh
node bridge/server.mjs --allow-estimated
```
## HTTP契約
`POST /pose` に `Content-Type: application/json` と `Authorization: Bearer TOKEN` を付け、`PoseFusion.update()` の返り値を送ります。任意の `clientId` をセッションごとに追加できます。巻き戻し・シーク・再開始の最初のフレームには `reset:true` を追加するか `clientId` を更新します。並行送信で順番を入れ替えず、前回POST完了後に最新フレームを送ってください。`GET /health` は送出状態を返します。
## 校正JSON
`K` は元動画画素に対する内部行列、`R` と `t` はOpenCV形式の `Pcamera = R * Pworld + t` です。行列はrow-majorで指定します。カメラ・世界座標ともx右/y下/z前向きで記録し、ブラウザ内部でx右/y上/z後ろへ変換します。`imageSize` があれば動画の解像度に合わせてKを拡縮します。`D` は任意の `[k1,k2,p1,p2,k3]`。推測画角や自動整列を実測校正として保存しないでください。
```json
{
  "heightM": 1.70,
  "heightCalibrated": true,
  "cameras": [
    {"sourceId":"video-1","K":[1000,0,960,0,1000,540,0,0,1],"R":[1,0,0,0,1,0,0,0,1],"t":[0,0,0],"imageSize":{"width":1920,"height":1080},"calibrated":false}
  ]
}
```
上記数値は入力形式の例で、実測値ではないため `calibrated:false` です。カメラの床に対する姿勢が未校正なら床原点も未確定です。プレビューの初期表示位置合わせを床校正として利用しないでください。
## 2映像の推定整列
`fusion.estimateCameraAlignment([observation1, observation2])` は、同時刻の共通する相対3D関節が8点以上ある場合に推定剛体整列を行います。採用残差の上限は18cmで、戻り値の `residualM` を記録します。設定も内部で反映されます。以後は `estimatedMultiView` と表示され、`calibrated:false` を維持します。未校正の画角と学習済み3D骨格に依存するため、実測三角測量の合格とは区別します。別の姿勢・時刻で残差と左右対応を再確認してください。
## 部分2D観測の骨長制約
有効な3D根位置と隣接3D関節がある場合にだけ、同じ骨グラフで腕・脚・胴体・指・足の2D観測を補完します。画像射線と既知骨長の球の交点を求め、複数の隣接骨、信頼できる同時刻の相対3D、または短時間の深度履歴で候補を選びます。2解を区別できない場合は `position:null` と `depthStatus:ambiguous-bone-depth` を残します。2Dだけのフレームから固定Zを生成しません。
補完した点は `source:boneConstrained`、`depthEvidence`、`depthUncertaintyM`、`boneLengthSource` を持ち、3D表示では紫です。近接した2解を狭い範囲に限定できた場合も `depthAmbiguous:true` を維持し、その点を使うトラッカーは送出しません。`diagnostics.lifted` はモデルが直接推論した3D点と別に数えます。履歴からの骨長は補完前の有効3Dだけで更新し、補完結果を教師として自己強化しません。
明示する骨長は校正JSONの `boneLengthsM` に `{"left_knee:left_ankle":0.42}` の形式で追加できます。履歴中央値と反対側の有効な骨長も利用します。本人身長を `heightCalibrated:true` で指定した場合は一部の身体骨に粗い人体比率を使いますが、`estimatedAnthropometry` として記録し、実測骨長とは扱いません。600ms以上、実際の深度根拠が更新されない時間外挿は続けません。
