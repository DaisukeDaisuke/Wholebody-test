# 軽量VRCモデル専用作業

専用ブランチは compact-vrc-20261102-7fe4fac1。作業ディレクトリは /workspace/scratch/7fe4fac156c0/compact-vrc-7fe4fac1。
今回の目的は約50〜60点の学習済み軽量モデル、ローカル動画推論、VRM表示、フェーズ3用の送出を実装し、公開データでActions評価を行うこと。
rootは独立ブランチ、Actions、Sites、統合を担当。compact_model_researchはtraining/以下の学習実装のみ。app担当の範囲は別途確定する。
既存の phase2-20261102-4fc7ff60、phase2/732e048e、phase3-20261102-4fc7ff60 および既存Sitesを更新しない。
添付動画・動画の切り抜き・フレーム・特徴キャッシュ・動画由来ランドマークをGitHub、Sites、Actions artifactsへ含めない。学習とCI評価は公開データのみ。本人動画は本人端末のPoC内処理に限定する。
本チャットから他の共有エージェントへの受信確認は未取得。各担当のファイルを削除・上書きしない。
