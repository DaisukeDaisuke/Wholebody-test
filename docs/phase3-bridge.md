# フェーズ3：VRM表示とローカルVMT送出
## 実装範囲と検証の境界
実際の推定poseからVRMの腰・体幹・腕・脚・手・指・足の回転を計算する。Node.jsのローカルブリッジは、腰・左足・右足の位置と回転をOSC/UDPでVMTへ送る。VRM内の指アニメーションと、SteamVRの3個の仮想トラッカーは別の出力である。通常のトラッカー送出だけでVRChatの指ボーンが動くとは扱わない。
`server/bridge-ci.mjs` はGitHub Actions専用で、数値骨格からの回転、認証、実際のWebSocket受信からUDPパケット受信、座標変換、時間切れ・切断時の無効化を検査する。SteamVR・VMTドライバ・HMD・VRChatの実機はこの実験で稼働していない。UDPを受信できたことを実機FBT成功、カメラ精度、低遅延の証明としない。結果は `phase3-bridge-report.json` に保存する。
## 足だけの入力
足専用の入力は元画像の足点だけを持ち、3D全身関節や骨盤位置を生成しない。初回フレームの踵→足先の画像内角度を基準にし、以降の角度差をVRMのFootボーンへ適用する。これは画面内の回転表示であり、3Dのyaw・pitchを測定した値ではない。入力・対象・seed版が変われば基準角を作り直す。自動モデルの左右割当は未確認のまま保持し、利用者が左右を明示した入力だけを確認済みとする。
3D全身入力でも、足首・母趾・小趾・踵から足の方向を推定する。独立した5本の足指の曲がりは未観測で、VRMの左右Toesボーンは足に従った基準姿勢を保つ。5指の曲がりを捏造して表示しない。足だけの2D入力はSteamVR送出の有効条件を満たさない。
## 起動
Node.js 22以降で、ソース一式のルートから実行する。依存は `ws@8.22.1` をバージョン・integrity付きで固定した。
```sh
npm ci --prefix server --ignore-scripts --omit=optional
node server/bridge.mjs
```
既定の表示ページは `http://127.0.0.1:8787`、WebSocketは `ws://127.0.0.1:8787/v1/pose` である。Nodeプロセスが生成した接続tokenを表示ページのtoken欄へ渡す。tokenはブラウザの永続保存、URL、計測JSON、poseログに含めない。既定では同じローカルページのOriginだけを受け付ける。
公開Sitesページから接続する場合は、起動時の `--origin` にそのページの正確なOriginを指定する。許可にパスやワイルドカードは使わない。HTTP(S)のOrigin許可を増やしても、listenerとUDP宛先は127.0.0.1のままである。
```sh
node server/bridge.mjs --origin https://YOUR-SITE-ORIGIN
```
公開HTTPSページからローカルWebSocketへ接続できる条件はブラウザやLocal Network Accessの対応に依存する。ブロックされた場合は上記のローカル表示ページで同じアプリを使う。信頼済みのローカルTLS証明書がある場合だけ、`--tls-cert` と `--tls-key` を併用し `wss://127.0.0.1:8787/v1/pose` へ接続できる。証明書検証の無効化は行わない。
## 校正の適用
映像側のworld座標を、実際のHMD・SteamVRのルーム基準へ合わせた剛体変換をJSONで適用する。アバターの身長・骨格を本人の測定値に使わない。現在の推定poseを任意に原点へ移しただけではHMD一致とは扱わない。
```json
{
  "schemaVersion": 1,
  "confirmed": false,
  "worldCalibrationId": "camera-floor-v1",
  "roomAlignmentId": "steamvr-room-v1",
  "rotationXYZW": [0, 0, 0, 1],
  "translationM": [0, 0, 0],
  "allowEstimatedMonocular": false
}
```
上記の値は説明用で、`confirmed:false` のままでは適用できない。既知の共通基準から決めた回転・平行移動と撮影側の校正IDを記録し、校正を確認した設定だけを `confirmed:true` とする。UIの「SteamVR座標校正JSONを適用」または `node server/bridge.mjs --calibration PATH.json` で読み込む。撮影側の `worldCalibrationId` と異なる設定では送出を無効にする。
`rotationXYZW` と `translationM` は、右手系・Y上・メートルの撮影worldから、右手系のSteamVR roomへの変換である。人物の前後左右移動を含む実rootを変換する。送出直前にUnity系へZ軸を反転する。必要なら `trackerOffsets` に `waist`、`left-foot`、`right-foot` それぞれの `positionM` と `rotationXYZW` を与え、人体基準と仮想トラッカーの取付方向の差を補正する。既定はゼロ平行移動・恒等回転である。
既定では校正済み多視点由来のルーム位置を使う。カメラ内部・外部と人体尺度を校正した単眼推定を使う場合は、設定で `allowEstimatedMonocular:true` を明示する。この設定は推定を実測に変えるものではない。未校正の相対単眼、画像平面、校正ID不一致、欠測、低スコア、古い関節は有効トラッカーにならない。
## VMT側との契約
[VMT公式API](https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/api/)に従い、既定で `127.0.0.1:39570` へ次のOSCメッセージを送る。UDP 39571は占有せず、VMT Managerの応答ポートを妨げない。
```text
/VMT/Room/Unity
int index, int enable, float timeoffset,
float x, float y, float z, float qx, float qy, float qz, float qw
```
型タグは `,iiffffffff`、位置はメートル、回転は正規化した `xyzw`、`timeoffset=0`。index 0・1・2を腰・左足・右足に割り当て、`enable=1` でtracker、`enable=0` で無効にする。プロセス全体のVMT Resetやドライバ設定の書換えは送らない。
右手系の姿勢をUnity左手系へ変える式は `pUnity=[x,y,-z]`、`qUnity=[-qx,-qy,qz,qw]` である。これは単なる位置の符号変更に加えて回転行列も反射変換した結果である。VMT Managerのルーム設定、SteamVRの腰・左右足ロール割当、HMD原点合わせ、VRChat FBT校正は実機の条件として残る。SteamVRトラッカー移動は、VRChatのゲーム内ロコモーション操作とは異なる。
## 受信と失効
WebSocketは16 KiB上限のJSONだけを受け付け、最初の2.5秒以内にtoken認証する。OriginとHostを照合し、同時に1つのproducerだけを許可する。tokenをクエリ文字列へ置かない。受信内容から任意のUDP宛先やOSCアドレスを指定するAPIはない。
フレームの連番、source時刻、stream ID、数値型、クォータニオン、校正IDを検査する。反復した同一フレームを新しい観測として受け入れない。ブラウザは最新の3tracker分だけを送り、画像・動画・生の133点poseをシリアライズしない。ブリッジは姿勢トレースをファイルへ保存しない。
既定で受信が500 ms途切れると、最後に有効だった3trackerへ無効化パケットを送る。正常な切断、停止、拒否フレームでも無効化する。これは受信途絶に対する処理であり、Node自体の強制終了後のUDP送信やVMT側の自動失効を保証するものではない。`--stale-ms` は50〜2,000 msで指定できる。補完関節の年齢は別に180 msまでとし、通信を続けても古い補完が新観測へ戻らない。
`captureAgeMs` は撮影・入力から送信までの経過時間を呼出側が渡す。録画media時刻をPC時計として引き算しない。入力時刻の対応が不明な場合、通信更新率を撮影からHMDまでの遅延として報告しない。UDP送信成功の表示もSteamVRによる受理確認ではない。
## JavaScript API
`retargetPose(pose, rig, {previous, footHints})` はThree.js非依存で、`normalizedPose`、各boneのsource・score・valid、rootの単位状態、tracker候補、coverageを返す。rigは各normalized boneの基準位置、モデル座標でのworld quaternion、実nodeのlocal quaternion、最も近いhumanoid親を持つ。表示Groupの外側変換を含めず、VRM0補正などモデル内部の固定回転は含める。
[three-vrmのnormalized pose契約](https://pixiv.github.io/three-vrm/docs/classes/three-vrm.VRMHumanoid.html)は基準姿勢からのlocal変換である。実装の乗算順は `nodeQuaternion = poseDelta * restLocalQuaternion`。出力の `normalizedPose.hips.position` にrootの位置差分を含めるため、表示Groupでrootをもう一度加算しない。未校正ならhipsの絶対位置差分はゼロとし、相対回転だけを表示する。体幹の中間ボーン分配、長軸回りが直接観測できない四肢のtwist、足指の未観測状態も区別する。
```js
const client = new BridgeClient({ onStatus: status => showStatus(status) });
await client.connect({ url: 'ws://127.0.0.1:8787', token });
client.calibrate(roomCalibration);
client.sendPose(pose, { captureAgeMs, timeDomain: 'capture' });
client.disconnect();
```
公開された私的動画・抽出フレーム・生poseは、この実装にもActions成果物にも含めない。公開するものはコード、公開データの既存評価、合成数値fixtureと通信検証の集計だけである。
## 一次資料
- [VMT API：port・enable・Room/Unity・座標系](https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/api/)
- [VMT公式リポジトリ](https://github.com/gpsnmeajp/VirtualMotionTracker)
- [three-vrm humanoid API](https://pixiv.github.io/three-vrm/docs/classes/three-vrm.VRMHumanoid.html)
- [three-vrm VRMRigの基準姿勢と差分適用](https://github.com/pixiv/three-vrm/blob/dev/packages/three-vrm-core/src/humanoid/VRMRig.ts)
- [OSC 1.0仕様](https://opensoundcontrol.stanford.edu/spec-1_0.html)
- [Node.js UDP/datagram API](https://nodejs.org/api/dgram.html)
- [Chrome Local Network Access](https://developer.chrome.com/blog/local-network-access)
