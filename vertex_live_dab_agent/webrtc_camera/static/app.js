const summaryEl = document.getElementById('summary');
const devicesEl = document.getElementById('devices');

const deviceStates = new Map();
const peerStates = new Map();

function statusClass(value) {
  const normalized = String(value || 'unknown').toLowerCase();
  if (normalized === 'available') return 'available';
  if (normalized === 'failed') return 'failed';
  if (normalized === 'stopped') return 'stopped';
  if (normalized === 'starting') return 'starting';
  return 'degraded';
}

function transportLabel(value) {
  if (!value) return 'idle';
  return String(value).toLowerCase();
}

function formatAge(value) {
  if (value === null || value === undefined) return 'n/a';
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return 'n/a';
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

function formatNumber(value, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : '0.0';
}

function renderSummary(payload) {
  summaryEl.innerHTML = `
    <div class="panel"><strong>Overall</strong><div>${payload.status}</div></div>
    <div class="panel"><strong>Devices</strong><div>${payload.device_count}</div></div>
    <div class="panel"><strong>Available</strong><div>${payload.available_device_count}</div></div>
    <div class="panel"><strong>Failed Required</strong><div>${payload.failed_required_count}</div></div>
    <div class="panel"><strong>Peers</strong><div>${payload.peer_count ?? 0}</div></div>
  `;
}

function ensureCard(device) {
  let card = document.getElementById(`device-${device.device_id}`);
  if (card) return card;

  card = document.createElement('article');
  card.id = `device-${device.device_id}`;
  card.className = 'panel';
  card.innerHTML = `
    <div class="device-header">
      <div>
        <div class="device-title" data-role="title"></div>
        <div class="device-meta" data-role="meta"></div>
      </div>
      <span class="badge" data-role="device-state"></span>
    </div>
    <div class="video-shell">
      <video autoplay playsinline muted data-role="video"></video>
      <div class="transport" data-role="transport">idle</div>
    </div>
    <div class="stats">
      <div><strong>Capture FPS:</strong> <span data-role="fps"></span></div>
      <div><strong>Startup:</strong> <span data-role="startup"></span></div>
      <div><strong>Frames:</strong> <span data-role="frames"></span></div>
      <div><strong>Dropped:</strong> <span data-role="dropped"></span></div>
      <div><strong>Reconnects:</strong> <span data-role="reconnects"></span></div>
      <div><strong>Frame Age:</strong> <span data-role="frame-age"></span></div>
      <div><strong>Open:</strong> <span data-role="open"></span></div>
      <div><strong>Cached Frame:</strong> <span data-role="cached"></span></div>
    </div>
    <div class="error" data-role="error"></div>
  `;
  devicesEl.appendChild(card);
  return card;
}

function updateCard(device) {
  const card = ensureCard(device);
  card.querySelector('[data-role="title"]').textContent = `${device.device_id} · ${device.kind}`;
  card.querySelector('[data-role="meta"]').textContent = `${device.device_path || 'unresolved'} · locator=${device.locator}`;
  const stateBadge = card.querySelector('[data-role="device-state"]');
  stateBadge.className = `badge ${statusClass(device.state)}`;
  stateBadge.textContent = device.state;
  card.querySelector('[data-role="fps"]').textContent = formatNumber(device.fps);
  card.querySelector('[data-role="startup"]').textContent = device.startup_duration_ms == null ? 'n/a' : `${Math.round(device.startup_duration_ms)} ms`;
  card.querySelector('[data-role="frames"]').textContent = String(device.frames_captured ?? 0);
  card.querySelector('[data-role="dropped"]').textContent = String(device.dropped_frames ?? 0);
  card.querySelector('[data-role="reconnects"]').textContent = String(device.reconnect_attempts ?? 0);
  card.querySelector('[data-role="frame-age"]').textContent = formatAge(device.last_frame_age_seconds);
  card.querySelector('[data-role="open"]').textContent = String(Boolean(device.is_open));
  card.querySelector('[data-role="cached"]').textContent = String(Boolean(device.frame_available));
  card.querySelector('[data-role="error"]').textContent = device.last_error || device.last_warning || '';
  updateTransportState(device.device_id);
}

function removeMissingCards(deviceIds) {
  for (const element of Array.from(devicesEl.children)) {
    if (!deviceIds.has(element.id.replace('device-', ''))) {
      element.remove();
    }
  }
}

function applyStatus(payload) {
  renderSummary(payload);
  const devices = payload.devices || [];
  const knownIds = new Set();
  for (const device of devices) {
    knownIds.add(device.device_id);
    deviceStates.set(device.device_id, device);
    updateCard(device);
    maybeStartPeer(device);
  }
  removeMissingCards(knownIds);
}

function updateTransportState(deviceId, explicitState) {
  const card = document.getElementById(`device-${deviceId}`);
  if (!card) return;
  const transportEl = card.querySelector('[data-role="transport"]');
  const peer = peerStates.get(deviceId);
  const state = explicitState || peer?.transportState || 'idle';
  transportEl.textContent = `webrtc: ${transportLabel(state)}`;
}

async function negotiate(deviceId) {
  const existing = peerStates.get(deviceId);
  if (existing?.connecting) return;

  const pc = new RTCPeerConnection();
  const state = {
    pc,
    peerId: null,
    stream: null,
    reconnectTimer: null,
    connecting: true,
    transportState: 'connecting'
  };
  peerStates.set(deviceId, state);
  updateTransportState(deviceId, 'connecting');

  pc.addTransceiver('video', { direction: 'recvonly' });

  pc.ontrack = event => {
    const card = document.getElementById(`device-${deviceId}`);
    if (!card) return;
    const video = card.querySelector('[data-role="video"]');
    const [remoteStream] = event.streams;
    video.srcObject = remoteStream || new MediaStream([event.track]);
    state.stream = video.srcObject;
  };

  pc.onconnectionstatechange = () => {
    state.transportState = pc.connectionState;
    updateTransportState(deviceId, pc.connectionState);
    if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
      schedulePeerReconnect(deviceId);
    }
  };

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const response = await fetch('/api/signaling/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        device_id: deviceId,
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });
    if (!response.ok) {
      throw new Error(`offer failed: ${response.status}`);
    }
    const answer = await response.json();
    state.peerId = answer.peer_id;
    await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    state.transportState = 'connected';
    state.connecting = false;
    updateTransportState(deviceId, 'connected');
  } catch (error) {
    state.connecting = false;
    state.transportState = 'failed';
    updateTransportState(deviceId, 'failed');
    console.error('webrtc negotiation failed', deviceId, error);
    schedulePeerReconnect(deviceId);
  }
}

function maybeStartPeer(device) {
  const current = peerStates.get(device.device_id);
  if (current?.connecting) return;
  if (current?.pc && !['closed', 'failed', 'disconnected'].includes(current.transportState)) return;
  if (['failed', 'stopped'].includes(String(device.state || '').toLowerCase())) return;
  negotiate(device.device_id);
}

function schedulePeerReconnect(deviceId) {
  const device = deviceStates.get(deviceId);
  const current = peerStates.get(deviceId);
  if (!device) return;
  if (current?.reconnectTimer) return;
  if (current?.pc) {
    closePeer(deviceId, false);
  }
  const timeoutMs = 2000;
  const timer = setTimeout(() => {
    const latest = peerStates.get(deviceId);
    if (latest) {
      latest.reconnectTimer = null;
    }
    maybeStartPeer(device);
  }, timeoutMs);
  peerStates.set(deviceId, { ...(current || {}), reconnectTimer: timer, transportState: 'reconnecting' });
  updateTransportState(deviceId, 'reconnecting');
}

async function closePeer(deviceId, notifyBackend = true) {
  const state = peerStates.get(deviceId);
  if (!state) return;
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
  }
  if (notifyBackend && state.peerId) {
    try {
      await fetch(`/api/signaling/close/${encodeURIComponent(state.peerId)}`, { method: 'POST' });
    } catch (error) {
      console.warn('peer close request failed', deviceId, error);
    }
  }
  if (state.pc) {
    try {
      state.pc.ontrack = null;
      state.pc.onconnectionstatechange = null;
      state.pc.close();
    } catch (error) {
      console.warn('peer close failed', deviceId, error);
    }
  }
  peerStates.delete(deviceId);
  updateTransportState(deviceId, 'idle');
}

function connectStatusSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws/status`);
  socket.onmessage = event => {
    const payload = JSON.parse(event.data);
    applyStatus(payload);
  };
  socket.onclose = () => {
    setTimeout(connectStatusSocket, 1500);
  };
}

async function loadInitialStatus() {
  const response = await fetch('/api/status', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`status failed: ${response.status}`);
  }
  const payload = await response.json();
  applyStatus(payload);
}

window.addEventListener('beforeunload', () => {
  for (const deviceId of Array.from(peerStates.keys())) {
    closePeer(deviceId, true);
  }
});

loadInitialStatus().catch(error => {
  summaryEl.innerHTML = `<div class="panel">Failed to load status: ${error.message}</div>`;
});
connectStatusSocket();
