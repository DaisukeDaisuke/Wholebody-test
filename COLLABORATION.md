# このチャットの作業範囲
投稿者：ChatGPT 4fc7ff60
宛先：並行して全身トラッキングを実装する担当
種別：作業範囲と進捗
対象リポジトリ：DaisukeDaisuke/Wholebody-test
専用ブランチ：phase2-20261102-4fc7ff60
作業ディレクトリ：/workspace/scratch/4fc7ff60abb4/wholebody-phase2-20261102
新規Sites ID：appgprj_6ae7eb88f8188191b37ad05a226f16b5
親担当：モデル実行、前処理、モデル配信、GitHub Actions、Sites公開。
モデル研究担当：モデル選定、実画像注釈による評価、PythonとJSの数値比較。
幾何担当：校正、三角測量、時刻対応、単眼位置、遮蔽中の時系列推定。
画面担当：入力A/B、カメラ、LAN WebRTC、可視化、結果エクスポート。
実装範囲はフェーズ2まで。SteamVR/VRC送出は含まない。
既存のサイトや別担当のファイルは変更しない。
指定の共有掲示板 /workspace/scratch/agent-message-board はこの環境には存在せず、別Proの受信・同意は未確認。
実装コード7e3bf03021d7b2131ec7fc7b3b4583508b8ad6e9をActions run 41378750685で評価し、全step成功を確認した。調整4画像とは別の固定32画像を評価し、人物検出ON・手の選択採用ON・足専用補正OFFを既定にした。実録動作の欠測補完と数値幾何の評価を、実カメラ2台・指・足指の精度達成とは区別する。
公開URL：https://wholebody-phase2-4fc7ff60.daisukedaisuke389137.chatgpt.site
実測と限界：docs/holdout-results.md、docs/phase2-results.md、docs/results/evaluation-run6.tar.gz。
