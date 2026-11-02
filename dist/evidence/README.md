# VRC Compact 実測資料

採用モデルはmodels.jsonのbodyです。精度基準に届かない軽量候補は比較記録として残し、既定へ昇格させません。候補選択用のvalidationと、選択固定後に開くtestは分離しています。

全モデルはGitHub ActionsからHugging Faceへ配布し、このアプリは公開resolve APIの固定revisionからSHA-256照合付きで取得します。

精度はCOCO-WholeBodyの人物正解矩形で計測したPCKです。headless-video.jsonはNode.jsでの実動画推論とVRM更新の集計です。観測欠測を注入するchallenge、実RGBの部分切り抜き、通常の動画入力は別の結果として記録します。

非公開動画、画像、切り抜き、観測座標列は配布しません。headless GPU描画とVRC実機は測定に含みません。

顔の目・口は68点ランドマーク由来、頬は輪郭幅変化のみで頬の膨らみは未学習です。足はFootNet CC-BY-4.0による足裏・甲・側面の分類です。
