const $ = (id) => document.getElementById(id);

export async function loadEvaluationPanel() {
  const container = $('evaluation-content');
  const button = $('refresh-evaluation');
  button.disabled = true;
  const paths = ['report.json', 'codec-parity.json', 'geometry-report.json', 'panoptic-report.json', 'foot-report.json', 'detector-report.json', 'holdout-report.json',
    'phase3-real-model.json', 'phase3-tapir-wasm.json', 'phase3-vrm-integration.json', 'phase3-bridge-report.json', 'phase3-webgpu.json'];
  const element = (tag, text, className) => {
    const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node;
  };
  const percent = (value) => Number.isFinite(value) && value >= 0 && value <= 1 ? `${(100 * value).toFixed(1)}%` : '—';
  const signedPoints = (value) => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${(100 * value).toFixed(1)} pt` : '—';
  const check = (parent, label, value, style = '') => {
    const row = element('div', undefined, 'evaluation-check'); row.append(element('span', label), element('strong', value, style)); parent.append(row);
  };
  const table = (headers, rows) => {
    const wrapper = element('div', undefined, 'evaluation-table-wrap'), body = element('tbody'), grid = element('table', undefined, 'evaluation-table');
    const head = element('thead'), titles = element('tr'); headers.forEach((text) => titles.append(element('th', text))); head.append(titles); grid.append(head, body);
    rows.forEach((cells) => {
      const row = element('tr'); cells.forEach((value) => row.append(typeof value === 'object' && value.nodeType ? value : element('td', value))); body.append(row);
    });
    wrapper.append(grid); return wrapper;
  };
  const renderPhase3 = (real, wasm, vrm, bridge, gpu) => {
    const section = element('section', undefined, 'holdout-results');
    section.append(element('span', 'フェーズ3 · 公開データをActionsで実行', 'eyebrow'), element('h3', '実モデル・VRM・WebGPUの確認範囲'));
    const status = result => result.status === 'available' ? result.data.passed === true ? '通過' : '未通過' : '結果未配信';
    check(section, 'RTMW3D-X / 足検出モデルの実行', status(real));
    if (real.data?.passed === true) section.append(element('p', '公開静止画1枚で、出荷するWASMモデルの実出力を復号しています。全身133点の相対深度を扱う経路の確認で、3D精度の評価ではありません。', 'help-text'));
    check(section, '学習済み点追跡 · Node/WASM', status(wasm));
    if (wasm.data?.passed === true) {
      const comparisons = (wasm.data.comparisons || []).filter(row => Number.isFinite(row.coordinateMaxAbsPx));
      if (comparisons.length) check(section, '公式実装との最大座標差', `${Math.max(...comparisons.map(row => row.coordinateMaxAbsPx)).toPrecision(3)} px`);
      const times = comparisons.map(row => row.inferenceMs).filter(Number.isFinite);
      if (times.length) check(section, '同じ公開画像の再帰更新 · WASM実行時間', `${Math.min(...times).toFixed(0)}〜${Math.max(...times).toFixed(0)} ms / frame`);
      section.append(element('p', '同じ公開画像を異なる時刻で処理し、公式の追跡状態・座標と比較します。この差は変換と実装の一致度であり、実動画の足先誤差ではありません。', 'help-text'));
    }
    check(section, 'VRM読込 → 実ボーン → スキン頂点', status(vrm));
    if (vrm.data?.passed === true) {
      const checks = vrm.data.checks || [];
      check(section, '自作VRMでの反映・欠測復帰', `${checks.filter(row => row.passed === true).length} / ${checks.length} 条件`);
    }
    check(section, 'ローカルWebSocket → OSC UDP', status(bridge));
    if (bridge.data?.passed === true && Number.isFinite(bridge.data.udpPacketsReceived)) check(section, 'Actions内の実受信パケット', `${bridge.data.udpPacketsReceived} 件`);
    section.append(element('p', 'VRMの確認には自作モデルと既知の姿勢を使います。OSCの受信確認はSteamVRやVRChat実機の動作確認とは別です。', 'help-text'));
    check(section, 'WebGPUの演算互換性', status(gpu));
    if (gpu.data?.softwareOnly === true) {
      const difference = gpu.data.worker?.comparison?.coordinateMaxAbsPx;
      if (Number.isFinite(difference)) check(section, 'WebGPUと公式実装の最大座標差', `${difference.toPrecision(3)} px`);
      section.append(element('p', 'ActionsのSwiftShader（CPUによるWebGPU実装）で演算とGPUバッファの経路を確認します。実GPUの推論fpsは測っていません。端末の実測速度は上の速度欄で確認してください。', 'help-text'));
    } else section.append(element('p', '実GPUの推論fpsは未検証です。WebGPUを選択できることだけでは実時間動作を保証しません。', 'help-text'));
    const runId = wasm.data?.workflowRunId || bridge.data?.runId;
    if (/^\d+$/.test(String(runId || ''))) {
      const run = element('a', 'フェーズ3の実行記録 ↗');
      run.href = `https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/${runId}`; run.target = '_blank'; run.rel = 'noreferrer'; section.append(run);
    }
    return section;
  };
  const renderHoldout = (result) => {
    const section = element('section', undefined, 'holdout-results');
    section.append(element('span', '固定した標本・ルールで検証', 'eyebrow'), element('h3', '調整に使わなかった画像の実測'));
    const report = result.data || {}, dataset = report.dataset || {}, rule = dataset.selectionRule || report.selection?.rule || {};
    if (result.status !== 'available' || report.status !== 'complete' || !report.metrics?.clean?.detected_person?.wholebody) {
      const message = result.status === 'pending' ? '未評価 · 評価結果の配信待ち' : report.status === 'unavailable' ? '未評価 · 固定した画像をすべて取得できませんでした' : result.status === 'available' ? '未評価 · 完了した集計がありません' : '未評価 · 結果を読み込めません';
      section.append(element('p', message, 'holdout-state pending'));
      if (Number.isFinite(rule.count)) section.append(element('p', `選択済みの対象：${rule.count} 画像。未完了の部分集計は精度として表示しません。`, 'help-text'));
      return section;
    }
    if (Number.isFinite(dataset.images) && Number.isFinite(dataset.targets)) section.append(element('p', `${dataset.images} 画像 · ${dataset.targets} 人物`, 'holdout-state'));
    if (dataset.name) section.append(element('p', dataset.name, 'help-text'));
    section.append(element('p', '主表は人物の自動検出から開始。手足の補正条件を固定して比較します。', 'holdout-intro'));
    const footMethod = 'hand_selective_foot_one2many_blend';
    const conditions = [['clean', '通常画像'], ['hand', '手を遮蔽'], ['foot', '足を遮蔽']];
    const appendRoute = (parent, route) => {
      const rows = [];
      for (const [condition, label] of conditions) {
        const methods = report.metrics?.[condition]?.[route]; if (!methods) continue;
        const base = methods.wholebody, hand = methods.hand_selective, foot = methods[footMethod];
        const before = base?.foot?.pck020, after = foot?.foot?.pck020;
        const delta = Number.isFinite(before) && Number.isFinite(after) ? after - before : null;
        rows.push([label, percent(base?.body?.pck020), `${percent(base?.hand?.pck020)} → ${percent(hand?.hand?.pck020)}`,
          `${percent(before)} → ${percent(after)}`, element('td', signedPoints(delta), Number.isFinite(delta) && delta < 0 ? 'evaluation-delta-negative' : Number.isFinite(delta) && delta > 0 ? 'evaluation-delta-positive' : '')]);
      }
      parent.append(table(['PCK@0.2', '身体：本体', '手：本体 → 選択採用', '足：本体 → 比較補正', '足の差'], rows));
      const clean = report.metrics?.clean?.[route]?.wholebody;
      if (Number.isFinite(clean?.hand?.points) && Number.isFinite(clean?.foot?.points)) parent.append(element('p', `通常画像の採点分母：手 ${clean.hand.points} 点 / 足 ${clean.foot.points} 点。未検出・低スコアの点も含めます。`, 'help-text'));
      if (Number.isFinite(clean?.hand?.invalidPredictionCount) && Number.isFinite(clean?.foot?.invalidPredictionCount)) check(parent, '本体の無効予測（通常画像の手 / 足）', `${clean.hand.invalidPredictionCount} / ${clean.foot.invalidPredictionCount} 点`);
    };
    appendRoute(section, 'detected_person');
    section.append(element('p', '手の選択採用が既定ON、足の比較補正は既定OFFです。足の列は比較設定をONにした経路の測定で、採用済みという意味ではありません。', 'help-text'));
    section.append(element('p', '手は注釈BBoxの長辺、足は注釈点の最大広がりで正規化。遮蔽は静止画へ加えた矩形マスクです。', 'help-text'));
    const masked = element('details'); masked.append(element('summary', '人物の自動検出：隠した点だけの成績'));
    const maskedRows = [];
    for (const [condition, label, part, method] of [['hand', '隠した手の点', 'hand', 'hand_selective'], ['foot', '隠した足の点', 'foot', footMethod]]) {
      const methods = report.maskedPointMetrics?.[condition]?.detected_person;
      const before = methods?.wholebody?.[part], after = methods?.[method]?.[part]; if (!before) continue;
      maskedRows.push([label, Number.isFinite(before.points) ? String(before.points) : '—', percent(before.pck020), percent(after?.pck020)]);
    }
    if (maskedRows.length) masked.append(table(['PCK@0.2', '点数', '本体', '補正後'], maskedRows));
    else masked.append(element('p', '隠した点だけの集計は記載されていません。', 'help-text'));
    section.append(masked);
    const reference = element('details'); reference.append(element('summary', '参考：正解の人物枠を入力した場合'));
    if (report.metrics?.clean?.gt_person_reference) {
      reference.append(element('p', '正解BBoxをモデルに与える参考経路です。主表の自動人物選択とは条件が異なります。', 'help-text'));
      appendRoute(reference, 'gt_person_reference');
    } else reference.append(element('p', '参考経路の集計は記載されていません。', 'help-text'));
    section.append(reference);
    const methods = Object.keys(report.metrics.clean.detected_person).filter((method) => method.startsWith('hand_selective_foot_'));
    if (methods.length) {
      const alternatives = element('details'); alternatives.append(element('summary', '足補正の候補を比較'));
      const name = (method) => method.replace('hand_selective_foot_', '').replace('_associated', ' · 直接採用').replace('_blend', ' · 50%補正');
      alternatives.append(table(['足 PCK@0.2', ...conditions.map(([, label]) => label)], methods.map((method) => [name(method), ...conditions.map(([condition]) => percent(report.metrics?.[condition]?.detected_person?.[method]?.foot?.pck020))])));
      alternatives.append(element('p', '画面の比較設定は one2many・50%補正です。5本の足指の独立した関節精度は評価していません。', 'help-text'));
      section.append(alternatives);
    }
    const selection = element('details'); selection.append(element('summary', '固定した標本と評価条件'));
    if (dataset.developmentImagesExcluded === true) check(selection, '本PoCの調整画像', '除外済み');
    if (rule.selectionMadeBeforeAnyHoldoutModelInference === true && rule.selectionUsesModelOutputs === false) check(selection, '画像・人物の選択', '推論前に固定 · モデル出力は不使用');
    if (report.policies?.frozenBeforeHoldoutInference === true) check(selection, '手足の補正ルール', 'この評価の推論前に固定');
    if (rule.replaceUnavailableImages === false) check(selection, '取得できない画像の差替え', 'なし');
    if (Array.isArray(rule.excludedImageIds)) check(selection, '除外した調整画像数', `${rule.excludedImageIds.length} 枚`);
    if (typeof rule.selectionSha256 === 'string') selection.append(element('p', `標本 SHA-256：${rule.selectionSha256}`, 'holdout-selection'));
    selection.append(element('p', '最大の対象人物を先に固定し、採点できる手と足の注釈がある画像を選んでいます。一般の動画全体を代表する抽出ではありません。', 'help-text'));
    if (dataset.trainingOverlapExcluded === false || rule.upstreamTrainingOverlapExcluded === false) selection.append(element('p', '元モデルの学習・開発データとの重複は除外できていません。', 'evaluation-basis'));
    section.append(selection, element('p', 'この結果は静止画の限定評価です。実動画の追従、2視点の精度、足指の独立した曲がりを実証した値ではありません。', 'evaluation-basis'));
    const date = report.createdAt ? new Date(report.createdAt) : null;
    if (date && Number.isFinite(date.getTime())) section.append(element('p', `計測：${new Intl.DateTimeFormat('ja-JP', { timeZone: 'Asia/Tokyo', dateStyle: 'short', timeStyle: 'short' }).format(date)} JST${typeof report.gitCommit === 'string' ? ` · ${report.gitCommit.slice(0, 8)}` : ''}`, 'help-text'));
    if (/^\d+$/.test(String(report.workflowRunId || ''))) {
      const run = element('a', 'この固定標本のActions run ↗'); run.href = `https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/${report.workflowRunId}`; run.target = '_blank'; run.rel = 'noreferrer'; section.append(run);
    }
    return section;
  };
  try {
    const results = await Promise.all(paths.map(async (path) => {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
      try {
        const response = await fetch(`./evaluation/${path}`, { signal: controller.signal, cache: 'no-cache', credentials: 'same-origin' });
        if (response.status === 404) return { path, status: 'pending' };
        if (!response.ok) return { path, status: 'unavailable', message: `HTTP ${response.status}` };
        const data = await response.json();
        if (!data || typeof data !== 'object') throw new Error('Invalid report');
        return { path, status: 'available', data };
      } catch (error) { return { path, status: 'unavailable', message: error.name === 'AbortError' ? '読込タイムアウト' : '読込失敗' }; }
      finally { clearTimeout(timer); }
    }));
    const [main, codec, geometry, panoptic, foot, detector, holdout, real, wasm, vrm, bridge, gpu] = results;
    const content = document.createDocumentFragment();
    content.append(renderPhase3(real, wasm, vrm, bridge, gpu));
    const historical = element('details'); historical.append(element('summary', '以前のRTMW-l · 固定32画像での精度'));
    historical.append(element('p', 'この静止画の成績はRTMW-l経路の結果です。現在のRTMW3D-Xや足の点追跡の動画精度へ読み替えません。', 'help-text'), renderHoldout(holdout)); content.append(historical);
    const development = element('details');
    development.append(element('summary', `調整に使った画像の診断${Number.isFinite(main.data?.dataset?.images) ? ` · ${main.data.dataset.images} 画像` : ''}`));
    if (main.status !== 'available') development.append(element('p', main.status === 'pending' ? '評価結果の配信待ち' : `実画像の評価結果を読めません（${main.message}）。`, 'evaluation-summary'));
    else {
      const report = main.data, dataset = report.dataset || {};
      const size = Number.isFinite(dataset.images) && Number.isFinite(dataset.people) ? `実画像 ${dataset.images} 枚 · ${dataset.people} 人物` : '実画像の評価結果';
      development.append(element('p', size, 'evaluation-summary'));
      if (dataset.name) development.append(element('p', dataset.name, 'help-text'));
      const rows = [];
      const hasSelective = Object.values(report.metrics || {}).some((metrics) => metrics?.hand_selective);
      const method = hasSelective ? 'hand_selective' : 'hand_refined';
      const differences = hasSelective ? report.handSelectiveDifference : report.handRefinementDifference;
      for (const [condition, label] of [['clean', '通常画像'], ['hand', '手を遮蔽'], ['foot', '足を遮蔽'], ['torso', '体幹を遮蔽']]) {
        const metrics = report.metrics?.[condition]; if (!metrics) continue;
        const before = metrics.wholebody?.hand?.pck020, after = metrics[method]?.hand?.pck020;
        const reported = differences?.[condition]?.pck020RefinedMinusBaseline;
        const delta = Number.isFinite(reported) ? reported : Number.isFinite(before) && Number.isFinite(after) ? after - before : null;
        const deltaCell = element('td', signedPoints(delta), Number.isFinite(delta) && delta > 0 ? 'evaluation-delta-positive' : Number.isFinite(delta) && delta < 0 ? 'evaluation-delta-negative' : '');
        rows.push([label, `${percent(before)} → ${percent(after)}`, deltaCell, percent(metrics.wholebody?.foot?.pck020)]);
      }
      if (rows.length) development.append(table(['PCK@0.2', hasSelective ? '手：本体 → 選択採用' : '手：本体 → 再推論', '差', '足：本体'], rows));
      else development.append(element('p', '対応する手・足のPCK集計がありません。', 'help-text'));
      const clean = report.metrics?.clean?.wholebody;
      if (Number.isFinite(clean?.hand?.points) && Number.isFinite(clean?.foot?.points)) development.append(element('p', `通常画像の分母：手 ${clean.hand.points} 点 / 足 ${clean.foot.points} 点。小規模な診断用標本です。`, 'help-text'));
      if (hasSelective) {
        development.append(element('p', '選択採用：手の再推論が不安定な場合、全身モデルの点を保持します。', 'help-text'));
        if (String(report.handSelectivePolicy?.evaluationStatus || '').includes('not independent')) development.append(element('p', 'この小標本の失敗例を参照して採用条件を設計しています。別データでの改善は未確認です。', 'evaluation-basis'));
        const detail = element('details'), comparisonRows = [];
        detail.append(element('summary', '従来の手の再推論と、選択採用を比較'));
        for (const [condition, label] of [['clean', '通常画像'], ['hand', '手を遮蔽'], ['foot', '足を遮蔽'], ['torso', '体幹を遮蔽']]) {
          const metrics = report.metrics?.[condition]; if (!metrics) continue;
          const legacy = metrics.hand_refined?.hand?.pck020, selected = metrics.hand_selective?.hand?.pck020;
          comparisonRows.push([label, percent(metrics.wholebody?.hand?.pck020), percent(legacy), percent(selected), signedPoints(Number.isFinite(legacy) && Number.isFinite(selected) ? selected - legacy : null)]);
        }
        detail.append(table(['手 PCK@0.2', '本体', '従来再推論', '選択採用', '従来との差'], comparisonRows)); development.append(detail);
      }
      development.append(element('p', '手は注釈BBoxの長辺、足は注釈点の最大広がりで誤差を正規化。低信頼点も分母に含みます。遮蔽行は矩形マスクを追加した静止画です。', 'help-text'));
      development.append(element('p', 'この表の人物領域は正解BBox。自動検出を含む精度や、実動画での追跡精度とは別です。', 'evaluation-basis'));
      const initialization = report.initializationExperiment;
      if (initialization?.metrics && Array.isArray(initialization.methodOrder)) {
        const detail = element('details'), rows = [];
        detail.append(element('summary', '正解BBoxを使わない全画面からの初期化'));
        const names = { full_frame: '全画面', full_frame_hand_refined: '全画面＋従来の手', full_frame_hand_selective: '全画面＋手の選択採用', predicted_person_crop: '予測した人物範囲', predicted_person_crop_hand_refined: '予測した人物範囲＋従来の手', predicted_person_crop_hand_selective: '予測した人物範囲＋手の選択採用' };
        for (const method of initialization.methodOrder) {
          const metrics = initialization.metrics[method]; if (!metrics) continue;
          rows.push([names[method] || method, percent(metrics.body?.pck020), percent(metrics.hand?.pck020), percent(metrics.foot?.pck020)]);
        }
        if (Number.isFinite(initialization.images)) detail.append(element('p', `${initialization.images} 画像。全画面 → 予測人物範囲 → 手の切り出し。`, 'help-text'));
        detail.append(table(['PCK@0.2', '身体', '手', '足'], rows));
        detail.append(element('p', '同じ静止画を2回処理した初期化の診断です。別人を選んだ場合も固定した対象との誤差に含みます。実動画の追従は未検証です。', 'help-text'));
        development.append(detail);
      }
      if (report.maskedPointMetrics) {
        const detail = element('details'), rows = [];
        detail.append(element('summary', '隠した点だけの成績'));
        for (const [condition, label, part] of [['hand', '手の遮蔽点', 'hand'], ['foot', '足の遮蔽点', 'foot'], ['torso', '体幹マスク内の手', 'hand']]) {
          const metrics = report.maskedPointMetrics[condition], before = metrics?.wholebody?.[part], after = metrics?.[method]?.[part];
          if (!before) continue;
          rows.push([label, Number.isFinite(before.points) ? String(before.points) : '—', percent(before.pck020), percent(after?.pck020)]);
        }
        detail.append(table(['対象', '点数', '本体', hasSelective ? '選択採用後' : '手の再推論後'], rows)); development.append(detail);
      }
      const date = report.createdAt ? new Date(report.createdAt) : null;
      if (date && Number.isFinite(date.getTime())) development.append(element('p', `計測：${new Intl.DateTimeFormat('ja-JP', { timeZone: 'Asia/Tokyo', dateStyle: 'short', timeStyle: 'short' }).format(date)} JST${typeof report.gitCommit === 'string' ? ` · ${report.gitCommit.slice(0, 8)}` : ''}`, 'help-text'));
    }
    const detectorDetail = element('details');
    detectorDetail.append(element('summary', '人物の自動検出から開始：実画像の精度'));
    if (detector.status === 'available' && detector.data.metrics?.detected_person) {
      const report = detector.data, rows = [];
      const appendMethod = (label, metric) => {
        if (!metric) return;
        rows.push([label, percent(metric.body?.pck020), percent(metric.hand?.pck020), percent(metric.foot?.pck020)]);
      };
      if (report.fullFrameComparison?.sameTargets === true) {
        const fullFrame = report.fullFrameComparison.metrics || {};
        appendMethod('全画面から開始', fullFrame.full_frame);
        if (fullFrame.full_frame_hand_selective) appendMethod('全画面＋手の選択採用', fullFrame.full_frame_hand_selective);
        else appendMethod('全画面＋従来の手', fullFrame.full_frame_hand_refined);
      }
      appendMethod('人物検出＋RTMW', report.metrics.detected_person);
      appendMethod('人物検出＋従来の手', report.metrics.detected_person_hand_refined);
      appendMethod('人物検出＋手の選択採用', report.metrics.detected_person_hand_selective);
      if (Number.isFinite(report.dataset?.images) && Number.isFinite(report.dataset?.people)) detectorDetail.append(element('p', `${report.dataset.images} 画像 · ${report.dataset.people} 人物。自動検出の最大枠を入力し、画像ごとの最大注釈人物を固定して採点。`, 'help-text'));
      detectorDetail.append(table(['開始方法 · PCK@0.2', '身体', '手', '足'], rows));
      const baseline = report.metrics.detected_person;
      if (Number.isFinite(baseline.hand?.points) && Number.isFinite(baseline.foot?.points)) check(detectorDetail, '採点した手 / 足の点数', `${baseline.hand.points} / ${baseline.foot.points}`);
      if (Number.isFinite(baseline.hand?.invalidPredictionCount) && Number.isFinite(baseline.foot?.invalidPredictionCount)) check(detectorDetail, '未検出等の無効予測（手 / 足）', `${baseline.hand.invalidPredictionCount} / ${baseline.foot.invalidPredictionCount}`);
      if (Array.isArray(report.cases) && report.cases.length) check(detectorDetail, '人物枠を検出した画像', `${report.cases.filter((item) => item.chosen != null).length} / ${report.cases.length}`);
      detectorDetail.append(element('p', '推論の人物選択に正解BBoxは使いません。未検出や別人を選んだ誤差も分母に残します。正解BBoxを入力する評価とは別の経路です。', 'help-text'));
      detectorDetail.append(element('p', '小標本での診断です。手の採用条件は同じ標本の失敗例から設計しており、独立データの改善・実動画の追従・WebGPU速度は未検証です。', 'evaluation-basis'));
      if (report.fullFrameComparison && report.fullFrameComparison.sameTargets !== true) detectorDetail.append(element('p', '全画面開始の評価とは採点対象が一致しないため、この表では比較していません。', 'help-text'));
      if (typeof report.gitCommit === 'string') detectorDetail.append(element('p', `人物検出評価のソース：${report.gitCommit.slice(0, 8)}`, 'help-text'));
    } else check(detectorDetail, '人物検出からの手・足精度', detector.status === 'pending' ? '未評価 · 評価結果の配信待ち' : detector.status === 'available' ? '未評価 · 測定済みの集計がありません' : '未評価 · 結果を読み込めません', 'pending');
    development.append(detectorDetail); content.append(development);
    const checks = element('details'); checks.append(element('summary', '実装の参照比較と遮蔽補完の検証'));
    if (codec.status === 'available') {
      check(checks, '前処理・復号の参照比較', codec.data.passed === true ? '一致条件を通過' : codec.data.passed === false ? '不一致あり' : '判定記載なし', codec.data.passed === false ? 'failed' : '');
      const max = codec.data.comparison?.coordinates?.maxAbs;
      if (Number.isFinite(max)) check(checks, '復号座標の最大差', `${max.toExponential(2)} px`);
    } else check(checks, '前処理・復号の参照比較', codec.status === 'pending' ? '評価結果の配信待ち' : '結果を読み込めません', 'pending');
    if (geometry.status === 'available') {
      const experiments = Array.isArray(geometry.data.experiments) ? geometry.data.experiments : [];
      check(checks, '幾何演算の数値fixture', experiments.length ? `${experiments.filter((item) => item.passed === true).length} / ${experiments.length} 条件通過` : geometry.data.passed === true ? '条件を通過' : '結果JSONを参照', geometry.data.passed === false ? 'failed' : '');
      checks.append(element('p', '既知座標を使う数式の確認です。カメラの認識精度を測った値ではありません。', 'help-text'));
    } else check(checks, '幾何演算の数値fixture', geometry.status === 'pending' ? '評価結果の配信待ち' : '結果を読み込めません', 'pending');
    if (panoptic.status === 'available' && panoptic.data.status === 'measured') {
      const data = panoptic.data, count = data.dataset?.evaluatedFrameCount;
      check(checks, '公開運動ラベルでの遮蔽補完', Number.isFinite(count) ? `${count} フレーム` : '評価結果あり');
      if (data.dataset?.name) checks.append(element('p', `${data.dataset.name}${data.dataset.sequence ? ` · ${data.dataset.sequence}` : ''}`, 'help-text'));
      const rows = (data.experiments || []).map((experiment) => [experiment.label || `${experiment.nominalGapMs} ms`,
        Number.isFinite(experiment.tracker?.mpjpeM) ? `${(experiment.tracker.mpjpeM * 1000).toFixed(1)} mm` : '—',
        Number.isFinite(experiment.holdLastVisible?.mpjpeM) ? `${(experiment.holdLastVisible.mpjpeM * 1000).toFixed(1)} mm` : '—']);
      if (rows.length) checks.append(table(['人工的な欠測', '補完のMPJPE', '最後の位置を保持'], rows));
      checks.append(element('p', '記録された3D運動ラベルを隠して補完を測定。映像から隠れた部位を認識する精度の検証ではありません。', 'help-text'));
    } else {
      const status = panoptic.data?.status;
      check(checks, '公開運動ラベルでの遮蔽補完', panoptic.status === 'pending' ? '評価結果の配信待ち' : status === 'unavailable' ? 'データ取得不可・未完了' : status === 'insufficient-label-coverage' ? '有効ラベルが不足' : '評価結果を確認できません', 'pending');
    }
    if (foot.status === 'available' && foot.data.status === 'measured' && foot.data.metrics) {
      const report = foot.data, detail = element('details');
      detail.append(element('summary', '足の専門モデル：実画像での比較結果'));
      if (Number.isFinite(report.dataset?.images) && Number.isFinite(report.dataset?.people)) detail.append(element('p', `${report.dataset.images} 画像 · ${report.dataset.people} 人物。母趾・小趾・踵・足首の8点モデル。`, 'help-text'));
      const label = (method) => method === 'rtmw_l' ? 'RTMW-l 本体' : method.split('/').map((part) => ({
        'focus-foot8-one2many': 'FOCUS one2many', 'focus-foot8-one2one': 'FOCUS one2one',
        person_crop: '人物範囲', predicted_foot_crop: '予測足範囲', selective: '選択採用', direct: '直接採用'
      })[part] || part).join(' / ');
      const rows = [], clean = report.metrics.clean || {}, base = clean.rtmw_l?.foot;
      const methods = Object.keys(clean).sort((a, b) => a === 'rtmw_l' ? -1 : b === 'rtmw_l' ? 1 : a.localeCompare(b));
      for (const method of methods) {
        const metric = clean[method]?.foot; if (!metric) continue;
        const reported = report.footRefinementDifference?.clean?.[method]?.pck020RefinedMinusBaseline;
        const delta = Number.isFinite(reported) ? reported : Number.isFinite(metric.pck020) && Number.isFinite(base?.pck020) ? metric.pck020 - base.pck020 : null;
        const deltaCell = element('td', method === 'rtmw_l' ? '基準' : signedPoints(delta), Number.isFinite(delta) && delta > 0 ? 'evaluation-delta-positive' : Number.isFinite(delta) && delta < 0 ? 'evaluation-delta-negative' : '');
        rows.push([label(method), percent(metric.pck020), deltaCell, percent(report.maskedPointMetrics?.foot?.[method]?.foot?.pck020)]);
      }
      detail.append(table(['足 PCK@0.2', '通常画像', '本体との差', '隠した足点'], rows));
      if (Number.isFinite(base?.points)) detail.append(element('p', `通常画像の足の分母：${base.points} 点。低信頼・未検出も分母に残し、元の全身モデルと同じ尺度で比較します。`, 'help-text'));
      const recommendation = report.recommendation;
      if (recommendation && typeof recommendation.adoptForDiagnosticUse === 'boolean') {
        check(detail, '評価スクリプトの提案', recommendation.adoptForDiagnosticUse ? '診断用候補として採用' : '診断用の採用を推奨せず', recommendation.adoptForDiagnosticUse ? '' : 'pending');
        if (recommendation.method) detail.append(element('p', label(recommendation.method), 'help-text'));
        if (Number.isFinite(recommendation.additionalCorrectCleanPoints)) check(detail, '候補の追加正解点（通常画像）', `${recommendation.additionalCorrectCleanPoints >= 0 ? '+' : ''}${recommendation.additionalCorrectCleanPoints} 点`);
      }
      detail.append(element('p', 'この比較は専門モデルの診断です。推論画面での採用、5本の足指の関節精度、実動画の精度を示しません。', 'help-text'));
      if (report.dataset?.trainingOverlapExcluded === false) detail.append(element('p', '学習データとの重複は未除外です。独立した検証データの結果ではありません。', 'evaluation-basis'));
      if (typeof report.gitCommit === 'string') detail.append(element('p', `足モデル評価のソース：${report.gitCommit.slice(0, 8)}`, 'help-text'));
      checks.append(detail);
    } else if (foot.status === 'available') {
      check(checks, '足の専門モデル', '未評価 · 測定済みの集計がありません', 'pending');
    } else check(checks, '足の専門モデル', foot.status === 'pending' ? '未評価 · 評価結果の配信待ち' : '未評価 · 結果を読み込めません', 'pending');
    checks.append(element('p', 'コードの検証通過だけで、実用精度を達成したとは判定しません。実カメラの動画精度と利用者のWebGPU速度は別の測定が必要です。', 'help-text'));
    content.append(checks);
    const links = element('div', undefined, 'evaluation-links');
    const labels = ['調整画像JSON', '参照比較JSON', '幾何演算JSON', '運動補完JSON', '足モデルJSON', '人物検出JSON', '固定標本JSON', '実モデルJSON', '点追跡WASM JSON', 'VRM反映JSON', 'OSC実受信JSON', 'WebGPU互換JSON'];
    results.forEach((result, index) => { if (result.status !== 'available') return; const link = element('a', labels[index]); link.href = `./evaluation/${result.path}`; link.target = '_blank'; link.rel = 'noreferrer'; links.append(link); });
    if (/^\d+$/.test(String(main.data?.workflowRunId || ''))) {
      const run = element('a', '調整画像のActions run ↗'); run.href = `https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/${main.data.workflowRunId}`; run.target = '_blank'; run.rel = 'noreferrer'; links.append(run);
    }
    if (/^\d+$/.test(String(foot.data?.workflowRunId || '')) && foot.data.workflowRunId !== main.data?.workflowRunId) {
      const run = element('a', '足モデルのActions run ↗'); run.href = `https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/${foot.data.workflowRunId}`; run.target = '_blank'; run.rel = 'noreferrer'; links.append(run);
    }
    if (/^\d+$/.test(String(detector.data?.workflowRunId || '')) && detector.data.workflowRunId !== main.data?.workflowRunId) {
      const run = element('a', '人物検出のActions run ↗'); run.href = `https://github.com/DaisukeDaisuke/Wholebody-test/actions/runs/${detector.data.workflowRunId}`; run.target = '_blank'; run.rel = 'noreferrer'; links.append(run);
    }
    content.append(links); container.replaceChildren(content);
  } finally { button.disabled = false; }
}
