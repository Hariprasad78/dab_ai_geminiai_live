(() => {
  const API_STORAGE_KEY = 'dab_api_base';
  const THEME_STORAGE_KEY = 'dab_theme';
  const TARGET_DEVICE_STORAGE_KEY = 'dab_target_device_profile';
  const YTS_COMMAND_STORAGE_KEY = 'dab_yts_live_command_id';

  const state = {
    apiBase: '',
    workspace: 'yts',
    streamRunning: false,
    audioRunning: false,
    currentCommandId: null,
    commandPollTimer: null,
    history: [],
    catalog: [],
    selectedTests: new Set(),
    advancedCommand: 'launch',
  };

  const commandDefs = {
    launch: { title: 'Launch App', description: 'Launch YouTube app on the specified device.', args: [{ name: 'device' }, { name: 'payload' }] },
    stop: { title: 'Stop App', description: 'Stop YouTube app on the specified device.', args: [{ name: 'device' }] },
    update: { title: 'Update YTS', description: 'Update YTS to the latest or target version.', args: [{ name: 'target-version' }] },
    cert: { title: 'Certification Tests', description: 'Run certification tests applicable to the device.', args: [{ name: 'device' }, { name: 'filter', multiple: true }] },
    check: { title: 'Check Device', description: 'Check device connectivity and properties.', args: [{ name: 'device' }] },
    reset: { title: 'Reset Device', description: 'Clear cookies and local storage on the device.', args: [{ name: 'device' }] },
    wakeup: { title: 'Wake Up Device', description: 'Send a wakeup packet to the target device.', args: [{ name: 'device' }] },
    'evergreen-channel': { title: 'Evergreen Channel', description: 'Get or set the Evergreen channel on a device.', args: [{ name: 'device' }, { name: 'channel' }] },
    'evergreen-update': { title: 'Evergreen Update', description: 'Set Evergreen update channel and wait for completion.', args: [{ name: 'device' }, { name: 'channel' }] },
    login: { title: 'Login', description: 'Login to the YouTube Certification Portal.', args: [] },
    logout: { title: 'Logout', description: 'Logout from the YouTube Certification Portal.', args: [] },
    user: { title: 'Check User', description: 'Check membership in Saltmine using local credentials.', args: [] },
    credits: { title: 'Credits', description: 'Print open source credits used by YTS.', args: [] },
  };

  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const AV_MIME = 'video/mp4; codecs="avc1.42E01F, mp4a.40.2"';
  let avSocket = null;
  let avMediaSource = null;
  let avSourceBuffer = null;
  let avObjectUrl = null;
  let avQueue = [];
  let avAppending = false;
  const parseTokens = (value) => {
    const input = String(value || '').trim();
    if (!input) return [];
    return [...input.matchAll(/"([^"]+)"|'([^']+)'|(\S+)/g)]
      .map((match) => match[1] || match[2] || match[3] || '')
      .map((token) => token.trim())
      .filter(Boolean);
  };

  function apiOrigin() {
    return state.apiBase || window.location.origin;
  }

  function requestUrl(path) {
    return `${state.apiBase || ''}${path}`;
  }

  async function api(path, method = 'GET', body = null) {
    const response = await fetch(requestUrl(path), {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : null,
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_) {}
      throw new Error(detail || response.statusText);
    }
    return response.json();
  }

  function showBanner(kind, message) {
    const error = $('app-error-banner');
    const info = $('app-info-banner');
    if (kind === 'error') {
      info.classList.add('hidden');
      error.textContent = message;
      error.classList.remove('hidden');
      return;
    }
    error.classList.add('hidden');
    info.textContent = message;
    info.classList.remove('hidden');
  }

  function clearBanners() {
    $('app-error-banner').classList.add('hidden');
    $('app-info-banner').classList.add('hidden');
  }

  function setManualResult(message, isError = false) {
    const element = $('manual-result');
    if (!element) return;
    element.textContent = message;
    element.style.color = isError ? 'var(--danger)' : '';
  }

  function setTheme(theme) {
    const normalized = theme === 'light' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', normalized);
    localStorage.setItem(THEME_STORAGE_KEY, normalized);
    $('theme-toggle').textContent = normalized === 'light' ? 'Dark Theme' : 'Light Theme';
  }

  function initTheme() {
    setTheme(localStorage.getItem(THEME_STORAGE_KEY) || 'dark');
    $('theme-toggle').addEventListener('click', () => {
      setTheme(document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    });
  }

  function initApiBase() {
    const configured = String(window.__HARNESS_API_BASE__ || '').trim();
    const stored = String(localStorage.getItem(API_STORAGE_KEY) || '').trim();
    const sameOrigin = ['127.0.0.1', 'localhost'].includes(window.location.hostname);
    state.apiBase = sameOrigin && !configured ? '' : (stored || configured || '');
    $('api-summary').textContent = `API: ${state.apiBase || `same-origin (${window.location.origin})`}`;
  }

  function switchWorkspace(workspace) {
    state.workspace = workspace;
    document.querySelectorAll('.workspace').forEach((element) => {
      element.classList.toggle('active', element.id === `workspace-${workspace}`);
    });
    document.querySelectorAll('.workspace-btn').forEach((button) => {
      button.classList.toggle('active', button.dataset.workspace === workspace);
    });
  }

  function switchTab(tab) {
    document.querySelectorAll('#yts-tab-bar .tab-btn').forEach((button) => {
      button.classList.toggle('active', button.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-panel').forEach((panel) => {
      panel.classList.toggle('active', panel.id === `tab-${tab}`);
    });
  }

  function inferDeviceType(deviceId) {
    const id = String(deviceId || '').toLowerCase();
    return id.includes('tv') || id.includes('living-room') || id.includes('camera') ? 'tv' : 'setup-box';
  }

  function syncTestDeviceFromTarget(force = false) {
    const top = $('target-device-id').value.trim();
    const test = $('test-device-input');
    if (!top || !test) return;
    if (force || !test.value.trim()) test.value = top;
  }

  function saveTargetDevice() {
    const payload = {
      deviceType: $('target-device-type').value,
      deviceId: $('target-device-id').value.trim(),
    };
    localStorage.setItem(TARGET_DEVICE_STORAGE_KEY, JSON.stringify(payload));
  }

  function loadSavedTargetDevice() {
    try {
      const saved = JSON.parse(localStorage.getItem(TARGET_DEVICE_STORAGE_KEY) || 'null');
      if (saved?.deviceType) $('target-device-type').value = saved.deviceType;
      if (saved?.deviceId) $('target-device-id').value = saved.deviceId;
      syncTestDeviceFromTarget(true);
      renderTargetStatus(saved?.deviceId ? `Selected ${saved.deviceType === 'tv' ? 'TV' : 'Setup Box'}: ${saved.deviceId}` : 'No target device selected yet.');
    } catch (_) {
      renderTargetStatus('No target device selected yet.');
    }
  }

  function renderTargetStatus(message, isError = false) {
    const target = $('target-device-status');
    target.textContent = message;
    target.style.color = isError ? 'var(--danger)' : '';
  }

  async function applyTargetDeviceSelection() {
    const deviceType = $('target-device-type').value;
    const deviceId = $('target-device-id').value.trim();
    saveTargetDevice();
    syncTestDeviceFromTarget(true);
    try {
      await api('/capture/select', 'POST', {
        source: deviceType === 'tv' ? 'camera-capture' : 'hdmi-capture',
        preferred_kind: deviceType === 'tv' ? 'camera' : 'hdmi',
        device: '',
        persist: true,
      });
      renderTargetStatus(`${deviceType === 'tv' ? 'TV' : 'Setup Box'} routing applied${deviceId ? ` for ${deviceId}` : ''}.`);
      await Promise.allSettled([loadCaptureSource(), loadCaptureDevices(), refreshStreamStatus()]);
    } catch (error) {
      renderTargetStatus(`Failed to apply routing: ${error.message}`, true);
    }
  }

  async function loadHealth() {
    try {
      const data = await api('/health');
      const pill = $('health-pill');
      pill.textContent = data.mock_mode ? 'Mock Mode' : 'Live';
      pill.className = `pill ${data.status === 'ok' ? 'pill-online' : 'pill-offline'}`;
    } catch (error) {
      $('health-pill').textContent = 'Offline';
      $('health-pill').className = 'pill pill-offline';
      showBanner('error', `Backend health check failed: ${error.message}`);
    }
  }

  async function loadCaptureSource() {
    try {
      const data = await api('/capture/source');
      $('capture-mode-select').value = data.configured_source || 'auto';
      $('capture-kind-select').value = data.preferred_video_kind || 'auto';
      const audio = await api('/audio/source').catch(() => null);
      const parts = [
        `source=${data.configured_source || 'auto'}`,
        `kind=${data.preferred_video_kind || 'auto'}`,
        data.hdmi_available ? `video=${data.hdmi_device || 'ready'}` : 'video=not-ready',
      ];
      if (audio) parts.push(audio.enabled ? `audio=${audio.device || 'enabled'}` : 'audio=disabled');
      $('capture-source-info').textContent = parts.join(' · ');
    } catch (error) {
      $('capture-source-info').textContent = `Failed to load capture source: ${error.message}`;
    }
  }

  async function loadCaptureDevices() {
    try {
      const data = await api('/capture/devices');
      const select = $('capture-device-select');
      select.innerHTML = '<option value="">device: auto</option>';
      (data.devices || []).forEach((device) => {
        if (!device.device) return;
        const option = document.createElement('option');
        option.value = device.device;
        option.textContent = `${device.device} ${device.kind ? `[${device.kind}]` : ''} ${device.readable ? 'ok' : 'no-access'}${device.name ? ` - ${device.name}` : ''}`.trim();
        select.appendChild(option);
      });
      if (data.selected_video_device) select.value = data.selected_video_device;
    } catch (_) {}
  }

  async function applyCaptureSelection() {
    try {
      await api('/capture/select', 'POST', {
        source: $('capture-mode-select').value,
        preferred_kind: $('capture-kind-select').value,
        device: $('capture-device-select').value,
        persist: true,
      });
      setManualResult('Capture selection updated.');
      await Promise.allSettled([loadCaptureSource(), loadCaptureDevices(), refreshStreamStatus()]);
    } catch (error) {
      setManualResult(`Failed to apply capture selection: ${error.message}`, true);
    }
  }

  async function refreshStreamStatus() {
    try {
      const data = await api('/stream/status');
      const video = data.video || {};
      const audio = data.audio || {};
      const av = data.av || {};
      $('stream-diagnostics').textContent = [
        `video.source: ${video.configured_source || 'unknown'}`,
        `video.device: ${video.hdmi_device || video.selected_video_device || 'auto'}`,
        `video.kind: ${video.preferred_video_kind || 'auto'}`,
        `video.ready: ${video.hdmi_available ? 'yes' : 'no'}`,
        `audio.enabled: ${audio.enabled ? 'yes' : 'no'}`,
        `audio.ffmpeg: ${audio.ffmpeg_available ? 'yes' : 'no'}`,
        `audio.device: ${audio.device || 'auto'}`,
        `av.transport: ${av.transport || 'fallback'}`,
        `av.clients: ${av.subscriber_count || 0}`,
      ].join('\n');
    } catch (error) {
      $('stream-diagnostics').textContent = `Failed to load stream diagnostics: ${error.message}`;
    }
  }

  function stopCombinedStreamPlayer() {
    if (avSocket) {
      try { avSocket.onopen = null; } catch (_) {}
      try { avSocket.onmessage = null; } catch (_) {}
      try { avSocket.onerror = null; } catch (_) {}
      try { avSocket.onclose = null; } catch (_) {}
      try { avSocket.close(); } catch (_) {}
      avSocket = null;
    }
    avQueue = [];
    avAppending = false;
    avSourceBuffer = null;
    avMediaSource = null;
    if (avObjectUrl) {
      try { URL.revokeObjectURL(avObjectUrl); } catch (_) {}
      avObjectUrl = null;
    }
  }

  function pumpCombinedStreamQueue() {
    if (!avSourceBuffer || avAppending || avSourceBuffer.updating || !avQueue.length) return;
    avAppending = true;
    avSourceBuffer.appendBuffer(avQueue.shift());
  }

  function enqueueCombinedChunk(chunk) {
    if (!(chunk instanceof ArrayBuffer)) return;
    avQueue.push(new Uint8Array(chunk));
    if (avQueue.length > 120) avQueue.shift();
    pumpCombinedStreamQueue();
  }

  function avWebSocketUrl() {
    const url = new URL('/ws/stream/av', apiOrigin());
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return url.toString();
  }

  function toggleStream() {
    const button = $('toggle-stream-btn');
    const frame = $('stream-frame');
    if (state.streamRunning) {
      stopCombinedStreamPlayer();
      state.streamRunning = false;
      state.audioRunning = false;
      button.textContent = 'Start Stream';
      button.classList.add('secondary');
      frame.innerHTML = '<div class="empty-state">HDMI stream is stopped.</div>';
      $('audio-frame').innerHTML = '<div class="empty-state slim">Audio stream is stopped.</div>';
      $('toggle-audio-btn').textContent = 'Start Audio';
      $('toggle-audio-btn').classList.add('secondary');
      return;
    }
    state.streamRunning = true;
    button.textContent = 'Stop Stream';
    button.classList.remove('secondary');
    frame.innerHTML = `<img src="${apiOrigin()}/stream/hdmi?ts=${Date.now()}" alt="Live stream" />`;
  }

  async function toggleAudio() {
    const button = $('toggle-audio-btn');
    const frame = $('audio-frame');
    if (state.audioRunning) {
      state.audioRunning = false;
      button.textContent = 'Start Audio';
      button.classList.add('secondary');
      frame.innerHTML = '<div class="empty-state slim">Audio stream is stopped.</div>';
      return;
    }
    try {
      const audio = await api('/audio/source');
      if (!audio.enabled) throw new Error('Audio streaming is disabled on the backend.');
      if (!audio.ffmpeg_available) throw new Error('ffmpeg is not available on the host.');
      state.audioRunning = true;
      button.textContent = 'Stop Audio';
      button.classList.remove('secondary');
      frame.innerHTML = `<audio controls autoplay src="${apiOrigin()}/stream/audio?ts=${Date.now()}" style="width:100%"></audio>`;
    } catch (error) {
      setManualResult(error.message, true);
    }
  }

  async function captureScreenshot() {
    try {
      await api('/screenshot', 'POST');
      setManualResult('Screenshot captured.');
    } catch (error) {
      setManualResult(`Screenshot failed: ${error.message}`, true);
    }
  }

  async function sendKey(action) {
    try {
      const data = await api('/action', 'POST', { action });
      setManualResult(`${action}: ${JSON.stringify(data.result)}`);
    } catch (error) {
      setManualResult(`Manual action failed: ${error.message}`, true);
    }
  }

  function globalOptions() {
    const options = {};
    if ($('global-verbose-checkbox').checked) options['--verbose'] = true;
    if ($('global-colors-checkbox').checked) options['--colors'] = true;
    if ($('global-adb-path-input').value.trim()) options['--adb_path'] = $('global-adb-path-input').value.trim();
    if ($('global-logging-level-input').value.trim()) options['--logging-level'] = $('global-logging-level-input').value.trim();
    if ($('global-verbose-output-input').value.trim()) options['--verbose-output'] = $('global-verbose-output-input').value.trim();
    return options;
  }

  function renderCatalog() {
    const suiteFilter = $('catalog-suite-filter').value;
    const categoryFilter = $('catalog-category-filter').value;
    const query = $('catalog-search-input').value.trim().toLowerCase();
    const rows = state.catalog.filter((test) => {
      if (suiteFilter && test.test_suite !== suiteFilter) return false;
      if (categoryFilter && test.test_category !== categoryFilter) return false;
      if (!query) return true;
      return [test.test_id, test.test_title, test.test_suite, test.test_category].some((part) => String(part || '').toLowerCase().includes(query));
    });

    $('catalog-count').textContent = `${rows.length} tests shown · ${state.catalog.length} loaded`;
    const body = $('catalog-table-body');
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="4" class="empty-cell">No tests match the current filters.</td></tr>';
    } else {
      body.innerHTML = rows.slice(0, 250).map((test) => `
        <tr data-testid="${esc(test.test_id)}">
          <td>${esc(test.test_id)}</td>
          <td>${esc(test.test_title || '')}</td>
          <td>${esc(test.test_suite || '')}</td>
          <td>${esc(test.test_category || '')}</td>
        </tr>`).join('');
    }

    const suites = [...new Set(state.catalog.map((test) => test.test_suite).filter(Boolean))].sort();
    const categories = [...new Set(state.catalog.filter((test) => !suiteFilter || test.test_suite === suiteFilter).map((test) => test.test_category).filter(Boolean))].sort();
    fillSelect($('catalog-suite-filter'), suites, suiteFilter, 'All Suites');
    fillSelect($('catalog-category-filter'), categories, categoryFilter, 'All Categories');
    syncRunnerFilters();
  }

  function fillSelect(select, values, currentValue, placeholder) {
    const previous = currentValue || select.value;
    select.innerHTML = `<option value="">${placeholder}</option>`;
    values.forEach((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      if (value === previous) option.selected = true;
      select.appendChild(option);
    });
  }

  async function loadCatalog(refresh = false) {
    try {
      const guided = $('guided-mode-checkbox').checked || $('catalog-guided-checkbox').checked;
      const suffix = guided ? '?guided=1' : '';
      const data = refresh ? await api(`/yts/tests/refresh${suffix}`, 'POST') : await api(`/yts/tests${suffix}`);
      state.catalog = Array.isArray(data) ? data : [];
      renderCatalog();
      showBanner('info', `Loaded ${state.catalog.length} ${guided ? 'guided ' : ''}tests.`);
    } catch (error) {
      showBanner('error', `Failed to load test catalog: ${error.message}`);
    }
  }

  function syncRunnerFilters() {
    const suite = $('test-suite-select').value;
    const category = $('test-category-select').value;
    const suites = [...new Set(state.catalog.map((test) => test.test_suite).filter(Boolean))].sort();
    const categories = [...new Set(state.catalog.filter((test) => !suite || test.test_suite === suite).map((test) => test.test_category).filter(Boolean))].sort();
    fillSelect($('test-suite-select'), suites, suite, 'All Suites');
    fillSelect($('test-category-select'), categories, category, 'All Categories');
    const filtered = currentRunnerMatches();
    const select = $('test-name-select');
    const current = new Set([...select.selectedOptions].map((option) => option.value));
    select.innerHTML = '';
    filtered.forEach((test) => {
      const option = document.createElement('option');
      option.value = test.test_id;
      option.textContent = `${test.test_title || test.test_id} · ${test.test_id}`;
      option.selected = current.has(test.test_id);
      select.appendChild(option);
    });
    $('test-filter-summary').textContent = filtered.length
      ? `${filtered.length} tests match the current suite/category filters.`
      : 'No tests match the current suite/category filters.';
  }

  function currentRunnerMatches() {
    const suite = $('test-suite-select').value;
    const category = $('test-category-select').value;
    return state.catalog.filter((test) => {
      if (suite && test.test_suite !== suite) return false;
      if (category && test.test_category !== category) return false;
      return true;
    });
  }

  function renderSelectedTests() {
    const container = $('selected-tests');
    if (!state.selectedTests.size) {
      container.innerHTML = '<div class="empty-state slim">No tests selected yet.</div>';
      return;
    }
    container.innerHTML = [...state.selectedTests].map((testId) => {
      const match = state.catalog.find((item) => item.test_id === testId);
      return `<span class="chip">${esc(testId)}${match?.test_title ? ` · ${esc(match.test_title)}` : ''}<button type="button" data-remove-test="${esc(testId)}">×</button></span>`;
    }).join('');
  }

  function addSelectedTests(ids) {
    ids.forEach((id) => {
      const value = String(id || '').trim();
      if (value) state.selectedTests.add(value);
    });
    renderSelectedTests();
  }

  function addManualIds() {
    addSelectedTests(parseTokens($('test-manual-ids-input').value));
    $('test-manual-ids-input').value = '';
  }

  function renderSearchResults() {
    const query = $('test-search-input').value.trim().toLowerCase();
    const container = $('test-search-results');
    if (!query) {
      container.classList.add('hidden');
      container.innerHTML = '';
      return;
    }
    const results = state.catalog.filter((test) => {
      return [test.test_id, test.test_title, test.test_suite, test.test_category].some((part) => String(part || '').toLowerCase().includes(query));
    }).slice(0, 12);
    if (!results.length) {
      container.classList.add('hidden');
      return;
    }
    container.classList.remove('hidden');
    container.innerHTML = results.map((test) => `
      <div class="search-item" data-add-test="${esc(test.test_id)}">
        <strong>${esc(test.test_id)}</strong><br />
        <span>${esc(test.test_suite || '')} · ${esc(test.test_category || '')} · ${esc(test.test_title || '')}</span>
      </div>`).join('');
  }

  function testRequestBody() {
    const device = $('test-device-input').value.trim() || $('target-device-id').value.trim();
    if (!device) throw new Error('Device ID is required to run YTS tests.');
    syncTestDeviceFromTarget();

    const dropdownSelections = [...$('test-name-select').selectedOptions].map((option) => option.value);
    addSelectedTests(dropdownSelections);
    const selectedIds = [...state.selectedTests];
    const fallbackIds = selectedIds.length ? [] : currentRunnerMatches().map((test) => test.test_id).filter(Boolean);
    const resolvedIds = selectedIds.length ? selectedIds : fallbackIds;
    const filterTokens = parseTokens($('test-filter-tokens-input').value);
    const extraArgs = parseTokens($('test-extra-args-input').value);
    const guided = $('guided-mode-checkbox').checked;
    const interactive_ai = $('interactive-ai-checkbox').checked;
    const record_video = $('record-video-checkbox').checked;
    const jsonOutput = $('test-json-output-input').value.trim();

    if (!resolvedIds.length && !filterTokens.length && !guided && !extraArgs.length) {
      throw new Error('Select tests, add manual IDs, or choose filters before running.');
    }

    const params = [device, ...resolvedIds, ...filterTokens];
    if (guided) params.push('--guided');
    params.push(...extraArgs);
    if (jsonOutput) params.push('--json-output', jsonOutput);

    return {
      command: 'test',
      params,
      global_options: globalOptions(),
      output_file: jsonOutput || null,
      interactive_ai,
      record_video,
    };
  }

  function renderAdvancedCommandArgs() {
    const command = $('advanced-command-select').value;
    state.advancedCommand = command;
    const def = commandDefs[command];
    $('advanced-command-description').textContent = def?.description || '';
    const container = $('advanced-command-args');
    container.innerHTML = '';
    (def?.args || []).forEach((arg) => {
      const wrapper = document.createElement('label');
      wrapper.innerHTML = `<span>${esc(arg.name)}</span>${arg.multiple ? `<textarea data-advanced-arg="${esc(arg.name)}" placeholder="${esc(arg.name)}"></textarea>` : `<input data-advanced-arg="${esc(arg.name)}" placeholder="${esc(arg.name)}" />`}`;
      container.appendChild(wrapper);
    });
    container.querySelectorAll('[data-advanced-arg="device"]').forEach((input) => {
      if (!input.value) input.value = $('target-device-id').value.trim();
    });
  }

  function advancedRequestBody() {
    const command = $('advanced-command-select').value;
    const def = commandDefs[command];
    if (!def) throw new Error('Choose an advanced command first.');
    const params = [];
    (def.args || []).forEach((arg) => {
      const input = document.querySelector(`[data-advanced-arg="${CSS.escape(arg.name)}"]`);
      let value = input ? String(input.value || '').trim() : '';
      if (!value && arg.name === 'device') value = $('target-device-id').value.trim();
      if (!value) return;
      if (arg.multiple) params.push(...parseTokens(value));
      else params.push(value);
    });
    return { command, params, global_options: globalOptions(), interactive_ai: false, record_video: false };
  }

  function renderHistory() {
    const container = $('command-history-list');
    if (!state.history.length) {
      container.innerHTML = '<div class="empty-state slim">No YTS jobs yet.</div>';
      return;
    }
    container.innerHTML = state.history.map((item) => `
      <div class="history-item ${item.command_id === state.currentCommandId ? 'active' : ''}" data-command-id="${esc(item.command_id)}">
        <div><strong>${esc(item.command || '(starting...)')}</strong></div>
        <div class="meta">${esc(item.status)} · updated ${esc(item.updated_at || '')}</div>
        <div class="meta">${esc(item.command_id)}</div>
      </div>`).join('');
  }

  function renderDownloads(data) {
    const container = $('downloads-row');
    const origin = apiOrigin();
    const links = [];
    if (data.command_id) {
      links.push(`<a class="ghost-btn" href="${origin}/yts/command/live/${encodeURIComponent(data.command_id)}/terminal-log" target="_blank" rel="noopener">Terminal Log</a>`);
    }
    if (data.result_file_name) {
      links.push(`<a class="ghost-btn" href="${origin}/yts/command/live/${encodeURIComponent(data.command_id)}/result" target="_blank" rel="noopener">Saved Result</a>`);
    }
    if (data.video_file_name && data.video_recording_status !== 'unavailable') {
      links.push(`<a class="ghost-btn" href="${origin}/yts/command/live/${encodeURIComponent(data.command_id)}/video" target="_blank" rel="noopener">Video</a>`);
    }
    container.innerHTML = links.join(' ');
  }

  function renderPrompt(data) {
    const container = $('prompt-box');
    const prompt = data.pending_prompt;
    if (!prompt || !data.awaiting_input) {
      container.classList.add('hidden');
      container.innerHTML = '';
      return;
    }
    const options = Array.isArray(prompt.options) && prompt.options.length ? prompt.options : ['yes', 'no', '1', '2', '3', '4'];
    container.classList.remove('hidden');
    container.innerHTML = `
      <div class="prompt-title">Interactive Prompt</div>
      <div>${esc(prompt.text || '')}</div>
      ${prompt.ai_suggestion ? `<div class="status-line muted">Gemini suggestion: ${esc(prompt.ai_suggestion)}</div>` : ''}
      ${prompt.ai_visual_summary ? `<div class="status-line muted">TV visual context: ${esc(prompt.ai_visual_summary)}</div>` : ''}
      <div class="prompt-actions">${options.map((option) => `<button type="button" data-prompt-response="${esc(option)}" class="secondary">${esc(option)}</button>`).join('')}</div>
      <div class="prompt-actions">
        <input id="prompt-custom-input" placeholder="Custom response" />
        <button type="button" id="prompt-send-custom-btn" class="secondary">Send</button>
        <button type="button" id="prompt-suggest-btn" class="secondary">Suggest with Gemini</button>
        <button type="button" id="prompt-send-suggestion-btn">Send Gemini Suggestion</button>
      </div>`;
  }

  function renderInteractionLog(data) {
    const prompts = Array.isArray(data.prompts) ? data.prompts : [];
    const responses = Array.isArray(data.responses) ? data.responses : [];
    if (!prompts.length && !responses.length) {
      $('interaction-log').textContent = 'No interactive prompts or Gemini responses recorded for this job yet.';
      return;
    }
    const lines = [];
    prompts.forEach((prompt, index) => {
      lines.push(`Prompt ${index + 1}: ${prompt.text || '(empty prompt)'}`);
      if (prompt.options?.length) lines.push(`  Options: ${prompt.options.join(', ')}`);
      if (prompt.ai_suggestion) lines.push(`  Gemini: ${prompt.ai_suggestion}`);
      if (prompt.response) lines.push(`  Sent: ${prompt.response}`);
    });
    if (responses.length) {
      lines.push('', 'Responses:');
      responses.forEach((response, index) => {
        lines.push(`  ${index + 1}. [${response.source || 'manual'}] ${response.message || ''}`);
      });
    }
    $('interaction-log').textContent = lines.join('\n');
  }

  function renderCommandDetail(data) {
    state.currentCommandId = data.command_id || null;
    if (state.currentCommandId) localStorage.setItem(YTS_COMMAND_STORAGE_KEY, state.currentCommandId);
    $('live-command-status').textContent = `${data.status || 'unknown'} · ${data.command || '(starting...)'} · ${data.updated_at || ''}`;
    const lines = Array.isArray(data.logs) && data.logs.length
      ? data.logs.map((entry) => `[${entry.stream}] ${entry.message}`).join('\n')
      : [
          'Command: ' + (data.command || '(starting...)'),
          '',
          data.stdout ? `STDOUT:\n${data.stdout}` : '',
          data.stderr ? `STDERR:\n${data.stderr}` : '(waiting for output)',
        ].filter(Boolean).join('\n');
    $('live-command-log').textContent = lines;
    renderDownloads(data);
    renderPrompt(data);
    renderInteractionLog(data);
    renderHistory();
  }

  async function openCommand(commandId, startPolling = true) {
    if (!commandId) return;
    const data = await api(`/yts/command/live/${commandId}`);
    renderCommandDetail(data);
    if (data.status === 'running' && startPolling) startCommandPolling(commandId);
    else stopCommandPolling();
  }

  function stopCommandPolling() {
    if (state.commandPollTimer) {
      clearInterval(state.commandPollTimer);
      state.commandPollTimer = null;
    }
  }

  function startCommandPolling(commandId) {
    stopCommandPolling();
    state.commandPollTimer = window.setInterval(async () => {
      try {
        const data = await api(`/yts/command/live/${commandId}`);
        renderCommandDetail(data);
        if (data.status !== 'running') stopCommandPolling();
      } catch (_) {
        stopCommandPolling();
      }
    }, 1000);
  }

  async function loadHistory(preserve = true) {
    try {
      state.history = await api('/yts/command/live?limit=100');
      renderHistory();
      if ((!preserve || !state.currentCommandId) && state.history.length) {
        await openCommand(state.history[0].command_id, false);
      }
    } catch (error) {
      showBanner('error', `Failed to load YTS history: ${error.message}`);
    }
  }

  async function runLiveCommand(body, successMessage = '') {
    clearBanners();
    try {
      const started = await api('/yts/command/live', 'POST', body);
      if (successMessage) showBanner('info', successMessage);
      await loadHistory(false);
      await openCommand(started.command_id, true);
    } catch (error) {
      showBanner('error', `Failed to start YTS command: ${error.message}`);
    }
  }

  async function runDiscover() {
    switchTab('discover');
    $('discover-output').textContent = 'Discovery started…';
    await runLiveCommand({ command: 'discover', params: [], global_options: globalOptions(), interactive_ai: false, record_video: false }, 'YTS discover started.');
  }

  async function runTests() {
    switchTab('test');
    try {
      await runLiveCommand(testRequestBody(), 'YTS test run started.');
    } catch (error) {
      showBanner('error', error.message);
    }
  }

  async function runQuickCommand(command) {
    const def = commandDefs[command];
    const params = [];
    (def?.args || []).forEach((arg) => {
      if (arg.name === 'device') {
        const value = $('target-device-id').value.trim();
        if (value) params.push(value);
      }
    });
    await runLiveCommand({ command, params, global_options: globalOptions(), interactive_ai: false, record_video: false }, `${def?.title || command} started.`);
  }

  async function stopCurrentCommand() {
    if (!state.currentCommandId) return;
    try {
      await api(`/yts/command/live/${state.currentCommandId}/stop`, 'POST');
      await openCommand(state.currentCommandId, false);
      await loadHistory(true);
    } catch (error) {
      showBanner('error', `Failed to stop command: ${error.message}`);
    }
  }

  async function respondToPrompt(response) {
    if (!state.currentCommandId) return;
    try {
      await api(`/yts/command/live/${state.currentCommandId}/respond`, 'POST', { response });
      await openCommand(state.currentCommandId, true);
      await loadHistory(true);
    } catch (error) {
      showBanner('error', `Failed to answer prompt: ${error.message}`);
    }
  }

  async function suggestPrompt(sendResponse) {
    if (!state.currentCommandId) return;
    try {
      await api(`/yts/command/live/${state.currentCommandId}/suggest`, 'POST', { send_response: !!sendResponse });
      await openCommand(state.currentCommandId, true);
      await loadHistory(true);
    } catch (error) {
      showBanner('error', `Failed to request Gemini suggestion: ${error.message}`);
    }
  }

  function exportCatalog() {
    if (!state.catalog.length) {
      showBanner('error', 'Load the catalog before exporting.');
      return;
    }
    const blob = new Blob([JSON.stringify({ test_count: state.catalog.length, tests: state.catalog }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `yts-tests-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function loadRuns() {
    try {
      const runs = await api('/runs');
      $('ai-runs-list').innerHTML = runs.length ? runs.map((run) => `
        <div class="history-item">
          <div><strong>${esc(run.goal)}</strong></div>
          <div class="meta">${esc(run.status)} · ${esc(run.run_id)}</div>
        </div>`).join('') : '<div class="empty-state slim">No runs yet.</div>';
    } catch (error) {
      $('ai-runs-list').innerHTML = `<div class="empty-state slim">Failed to load runs: ${esc(error.message)}</div>`;
    }
  }

  async function startRun() {
    const goal = $('ai-goal-input').value.trim();
    if (!goal) {
      $('ai-start-run-status').textContent = 'Goal is required.';
      return;
    }
    try {
      const body = { goal };
      const appId = $('ai-app-id-input').value.trim();
      if (appId) body.app_id = appId;
      const data = await api('/run/start', 'POST', body);
      $('ai-start-run-status').textContent = `Run started: ${data.run_id}`;
      await loadRuns();
    } catch (error) {
      $('ai-start-run-status').textContent = `Failed to start run: ${error.message}`;
    }
  }

  function bindEvents() {
    document.querySelectorAll('.workspace-btn').forEach((button) => {
      button.addEventListener('click', () => switchWorkspace(button.dataset.workspace));
    });
    document.querySelectorAll('#yts-tab-bar .tab-btn').forEach((button) => {
      button.addEventListener('click', () => switchTab(button.dataset.tab));
    });

    $('target-device-id').addEventListener('input', () => {
      const id = $('target-device-id').value.trim();
      $('target-device-type').value = inferDeviceType(id);
      syncTestDeviceFromTarget();
    });
    $('apply-target-device-btn').addEventListener('click', applyTargetDeviceSelection);

    $('toggle-stream-btn').addEventListener('click', toggleStream);
    $('toggle-audio-btn').addEventListener('click', toggleAudio);
    $('capture-screenshot-btn').addEventListener('click', captureScreenshot);
    $('refresh-capture-devices-btn').addEventListener('click', loadCaptureDevices);
    $('apply-capture-selection-btn').addEventListener('click', applyCaptureSelection);
    document.querySelectorAll('[data-key]').forEach((button) => {
      button.addEventListener('click', () => sendKey(button.dataset.key));
    });

    $('action-discover-btn').addEventListener('click', runDiscover);
    $('action-load-catalog-btn').addEventListener('click', async () => {
      switchTab('catalog');
      await loadCatalog(true);
    });
    $('action-open-catalog-btn').addEventListener('click', async () => {
      switchTab('catalog');
      if (!state.catalog.length) await loadCatalog(false);
    });
    $('action-open-test-btn').addEventListener('click', () => switchTab('test'));
    $('action-run-tests-btn').addEventListener('click', runTests);
    $('action-update-btn').addEventListener('click', () => runQuickCommand('update'));
    $('action-stop-app-btn').addEventListener('click', () => runQuickCommand('stop'));
    $('action-stop-live-btn').addEventListener('click', stopCurrentCommand);

    $('discover-run-btn').addEventListener('click', runDiscover);
    $('catalog-refresh-btn').addEventListener('click', () => loadCatalog(true));
    $('catalog-export-btn').addEventListener('click', exportCatalog);
    $('catalog-guided-checkbox').addEventListener('change', async () => {
      $('guided-mode-checkbox').checked = $('catalog-guided-checkbox').checked;
      await loadCatalog(true);
    });
    $('catalog-suite-filter').addEventListener('change', renderCatalog);
    $('catalog-category-filter').addEventListener('change', renderCatalog);
    $('catalog-search-input').addEventListener('input', renderCatalog);
    $('catalog-table-body').addEventListener('click', (event) => {
      const row = event.target.closest('[data-testid]');
      if (!row) return;
      addSelectedTests([row.dataset.testid]);
      switchTab('test');
    });

    $('test-suite-select').addEventListener('change', syncRunnerFilters);
    $('test-category-select').addEventListener('change', syncRunnerFilters);
    $('refresh-catalog-from-test-btn').addEventListener('click', () => loadCatalog(true));
    $('add-selected-names-btn').addEventListener('click', () => addSelectedTests([...$('test-name-select').selectedOptions].map((option) => option.value)));
    $('add-filtered-tests-btn').addEventListener('click', () => addSelectedTests(currentRunnerMatches().map((test) => test.test_id).filter(Boolean)));
    $('add-manual-ids-btn').addEventListener('click', addManualIds);
    $('clear-selected-tests-btn').addEventListener('click', () => {
      state.selectedTests = new Set();
      renderSelectedTests();
    });
    $('selected-tests').addEventListener('click', (event) => {
      const button = event.target.closest('[data-remove-test]');
      if (!button) return;
      state.selectedTests.delete(button.dataset.removeTest);
      renderSelectedTests();
    });
    $('test-search-input').addEventListener('input', renderSearchResults);
    $('test-search-results').addEventListener('click', (event) => {
      const item = event.target.closest('[data-add-test]');
      if (!item) return;
      addSelectedTests([item.dataset.addTest]);
      $('test-search-input').value = '';
      renderSearchResults();
    });
    $('guided-mode-checkbox').addEventListener('change', async () => {
      $('catalog-guided-checkbox').checked = $('guided-mode-checkbox').checked;
      if (state.catalog.length) await loadCatalog(true);
    });
    $('run-test-btn').addEventListener('click', runTests);

    Object.keys(commandDefs).forEach((command) => {
      const option = document.createElement('option');
      option.value = command;
      option.textContent = commandDefs[command].title;
      $('advanced-command-select').appendChild(option);
    });
    $('advanced-command-select').value = state.advancedCommand;
    $('advanced-command-select').addEventListener('change', renderAdvancedCommandArgs);
    $('advanced-run-btn').addEventListener('click', async () => {
      try {
        await runLiveCommand(advancedRequestBody(), `${commandDefs[$('advanced-command-select').value].title} started.`);
      } catch (error) {
        showBanner('error', error.message);
      }
    });

    $('refresh-history-btn').addEventListener('click', () => loadHistory(true));
    $('stop-current-job-btn').addEventListener('click', stopCurrentCommand);
    $('command-history-list').addEventListener('click', (event) => {
      const item = event.target.closest('[data-command-id]');
      if (!item) return;
      openCommand(item.dataset.commandId, true);
    });
    $('prompt-box').addEventListener('click', (event) => {
      const response = event.target.closest('[data-prompt-response]');
      if (response) return respondToPrompt(response.dataset.promptResponse);
      if (event.target.id === 'prompt-send-custom-btn') {
        const value = $('prompt-custom-input')?.value?.trim();
        if (value) respondToPrompt(value);
      }
      if (event.target.id === 'prompt-suggest-btn') suggestPrompt(false);
      if (event.target.id === 'prompt-send-suggestion-btn') suggestPrompt(true);
    });

    $('ai-start-run-btn').addEventListener('click', startRun);
  }

  async function resumeLastCommand() {
    const saved = localStorage.getItem(YTS_COMMAND_STORAGE_KEY);
    if (!saved) return;
    try {
      await openCommand(saved, true);
    } catch (_) {
      localStorage.removeItem(YTS_COMMAND_STORAGE_KEY);
    }
  }

  async function initialize() {
    clearBanners();
    initTheme();
    initApiBase();
    bindEvents();
    loadSavedTargetDevice();
    renderAdvancedCommandArgs();
    renderSelectedTests();
    switchWorkspace('yts');
    switchTab('test');

    await Promise.allSettled([
      loadHealth(),
      loadCaptureSource(),
      loadCaptureDevices(),
      refreshStreamStatus(),
      loadCatalog(false),
      loadHistory(false),
      loadRuns(),
    ]);
    await resumeLastCommand();

    window.setInterval(loadHealth, 30000);
    window.setInterval(refreshStreamStatus, 30000);
    window.setInterval(loadCaptureSource, 30000);
    window.setInterval(loadCaptureDevices, 30000);
    window.setInterval(() => loadHistory(true), 30000);
  }

  window.addEventListener('error', (event) => showBanner('error', `Frontend error: ${event.message}`));
  window.addEventListener('unhandledrejection', (event) => showBanner('error', `Frontend promise error: ${event.reason?.message || event.reason || 'Unhandled promise rejection'}`));
  initialize().catch((error) => showBanner('error', `Startup failed: ${error.message}`));
})();
