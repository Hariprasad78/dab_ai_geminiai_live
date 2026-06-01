(() => {
  const API_STORAGE_KEY = 'dab_api_base';

  function normalizeApiBase(value) {
    return String(value || '').trim().replace(/\/+$/, '');
  }

  function resolveApiBase() {
    const configured = normalizeApiBase(window.__HARNESS_API_BASE__);
    const queryApi = new URLSearchParams(window.location.search).get('api');
    if (queryApi !== null) {
      const normalized = normalizeApiBase(queryApi);
      localStorage.setItem(API_STORAGE_KEY, normalized);
      return normalized;
    }
    const stored = normalizeApiBase(localStorage.getItem(API_STORAGE_KEY));
    const hostname = window.location.hostname || '';
    const shouldPreferSameOrigin = (hostname === '127.0.0.1' || hostname === 'localhost') && !configured;
    return shouldPreferSameOrigin ? '' : (configured || stored || '');
  }

  const apiBase = resolveApiBase();
  const url = (path) => `${apiBase}${path}`;
  let resolveReady;
  let currentUser = null;
  let googleScriptRequested = false;
  let apiCredentialsMode = 'same-origin';
  const ready = new Promise((resolve) => { resolveReady = resolve; });

  function esc(value) {
    return String(value || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function initials(user) {
    const words = String(user?.name || user?.email || 'Operator').trim().split(/\s+/).filter(Boolean);
    return words.slice(0, 2).map((word) => word[0]).join('').toUpperCase() || 'OP';
  }

  function injectStyle() {
    const style = document.createElement('style');
    style.textContent = `
      #dab-auth-overlay { position:fixed; inset:0; z-index:10000; display:grid; place-items:center; padding:20px; background:#111113; color:#fafafa; font:14px Inter,Arial,sans-serif; }
      #dab-auth-overlay.hidden, #dab-auth-profile.hidden, #dab-auth-menu.hidden { display:none; }
      #dab-auth-card { width:min(420px,100%); padding:28px; border:1px solid #3f3f46; border-radius:8px; background:#18181b; box-shadow:0 18px 48px rgba(0,0,0,.38); }
      #dab-auth-mark { width:42px; height:42px; display:grid; place-items:center; margin-bottom:18px; border:1px solid #52525b; border-radius:8px; background:#27272a; color:#fafafa; font-size:14px; font-weight:800; }
      #dab-auth-card h1 { margin:0 0 8px; color:#fafafa; font-size:22px; letter-spacing:0; }
      #dab-auth-card p { margin:0 0 20px; color:#b0b0b5; line-height:1.55; }
      #dab-auth-card small { display:block; margin-top:20px; color:#808087; line-height:1.5; }
      #dab-google-signin { min-height:44px; }
      #dab-auth-message { margin-top:14px; color:#fca5a5; font-size:13px; line-height:1.45; }
      #dab-auth-profile { position:fixed; right:12px; top:10px; z-index:9999; color:#fafafa; font:13px Inter,Arial,sans-serif; }
      #dab-auth-trigger { display:flex; align-items:center; gap:8px; min-height:36px; padding:4px 8px 4px 4px; border:1px solid #52525b; border-radius:8px; background:#18181b; color:#fafafa; cursor:pointer; box-shadow:0 8px 24px rgba(0,0,0,.24); }
      #dab-auth-trigger:hover { background:#27272a; }
      .dab-auth-avatar { width:28px; height:28px; display:grid; place-items:center; overflow:hidden; flex:0 0 auto; border-radius:50%; background:#3f3f46; color:#fafafa; font-size:10px; font-weight:800; }
      .dab-auth-avatar.large { width:42px; height:42px; font-size:13px; }
      .dab-auth-avatar img { width:100%; height:100%; object-fit:cover; }
      #dab-auth-trigger-email { max-width:190px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      #dab-auth-menu { position:absolute; right:0; top:44px; width:min(310px,calc(100vw - 24px)); padding:14px; border:1px solid #52525b; border-radius:8px; background:#18181b; box-shadow:0 16px 36px rgba(0,0,0,.36); }
      .dab-auth-menu-head { display:flex; gap:11px; align-items:center; padding-bottom:13px; border-bottom:1px solid #3f3f46; }
      .dab-auth-menu-copy { min-width:0; }
      .dab-auth-menu-name { overflow:hidden; color:#fafafa; font-weight:700; text-overflow:ellipsis; white-space:nowrap; }
      .dab-auth-menu-email { margin-top:3px; overflow:hidden; color:#b0b0b5; font-size:12px; text-overflow:ellipsis; white-space:nowrap; }
      .dab-auth-menu-meta { display:grid; gap:7px; padding:13px 0; color:#b0b0b5; font-size:12px; }
      .dab-auth-menu-row { display:flex; justify-content:space-between; gap:10px; }
      .dab-auth-menu-row strong { color:#d4d4d8; font-weight:600; }
      #dab-auth-logout, #dab-auth-login-status { width:100%; padding:8px 10px; border:1px solid #52525b; border-radius:6px; background:#27272a; color:#fafafa; cursor:pointer; font:inherit; font-weight:600; }
      #dab-auth-logout:hover { background:#3f3f46; }
      #dab-auth-login-status { color:#b0b0b5; cursor:default; font-size:12px; line-height:1.4; }
      #dab-auth-login-status.hidden, #dab-auth-logout.hidden { display:none; }
      @media (max-width:760px) { #dab-auth-trigger-email { display:none; } }
    `;
    document.head.appendChild(style);
  }

  function avatarMarkup(user, large = false) {
    const label = esc(initials(user));
    const picture = String(user?.picture || '').trim();
    const image = picture.startsWith('https://') ? `<img src="${esc(picture)}" alt="" referrerpolicy="no-referrer" />` : label;
    return `<span class="dab-auth-avatar${large ? ' large' : ''}">${image}</span>`;
  }

  function ensureUi() {
    if (document.getElementById('dab-auth-overlay')) return;
    injectStyle();
    const overlay = document.createElement('div');
    overlay.id = 'dab-auth-overlay';
    overlay.className = 'hidden';
    overlay.innerHTML = `
      <section id="dab-auth-card" aria-labelledby="dab-auth-title">
        <div id="dab-auth-mark">DAB</div>
        <h1 id="dab-auth-title">Sign in to DAB Console</h1>
        <p>Use your authorized Google account to control devices, view live streams, and run guided tests.</p>
        <div id="dab-google-signin"></div>
        <div id="dab-auth-message" role="alert"></div>
        <small>Your browser session is protected with a signed, HttpOnly cookie. Device actions remain linked to your operator account.</small>
      </section>`;
    document.body.appendChild(overlay);

    const profile = document.createElement('div');
    profile.id = 'dab-auth-profile';
    profile.className = 'hidden';
    profile.innerHTML = `
      <button type="button" id="dab-auth-trigger" aria-haspopup="true" aria-expanded="false">
        <span id="dab-auth-trigger-avatar"></span><span id="dab-auth-trigger-email"></span><span aria-hidden="true">▾</span>
      </button>
      <section id="dab-auth-menu" class="hidden" aria-label="User profile">
        <div class="dab-auth-menu-head">
          <span id="dab-auth-menu-avatar"></span>
          <div class="dab-auth-menu-copy"><div class="dab-auth-menu-name" id="dab-auth-name"></div><div class="dab-auth-menu-email" id="dab-auth-email"></div></div>
        </div>
        <div class="dab-auth-menu-meta">
          <div class="dab-auth-menu-row"><span>Account</span><strong id="dab-auth-account">Google</strong></div>
          <div class="dab-auth-menu-row"><span>Access</span><strong id="dab-auth-domain">Authorized operator</strong></div>
          <div class="dab-auth-menu-row"><span>Session</span><strong id="dab-auth-session">Active</strong></div>
          <div class="dab-auth-menu-row"><span>Jobs</span><strong id="dab-auth-job-count">0</strong></div>
          <div class="dab-auth-menu-row"><span>Artifacts</span><strong id="dab-auth-artifact-count">0</strong></div>
        </div>
        <button type="button" id="dab-auth-login-status" class="hidden" disabled>Google sign-in is not configured on this controller.</button>
        <button type="button" id="dab-auth-logout">Sign out</button>
      </section>`;
    document.body.appendChild(profile);
    document.getElementById('dab-auth-trigger').addEventListener('click', () => toggleProfileMenu());
    document.getElementById('dab-auth-logout').addEventListener('click', logout);
    document.addEventListener('click', (event) => { if (!profile.contains(event.target)) toggleProfileMenu(false); });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') toggleProfileMenu(false); });
  }

  function toggleProfileMenu(force) {
    const menu = document.getElementById('dab-auth-menu');
    const trigger = document.getElementById('dab-auth-trigger');
    if (!menu || !trigger) return;
    const open = typeof force === 'boolean' ? force : menu.classList.contains('hidden');
    menu.classList.toggle('hidden', !open);
    trigger.setAttribute('aria-expanded', String(open));
  }

  function showUser(user) {
    currentUser = user || null;
    ensureUi();
    document.getElementById('dab-auth-overlay').classList.add('hidden');
    const profile = document.getElementById('dab-auth-profile');
    if (!user) { profile.classList.add('hidden'); return; }
    const localMode = Boolean(user.auth_disabled);
    const email = String(user.email || '');
    const domain = localMode ? 'Google auth disabled' : (email.includes('@') ? email.split('@').pop() : 'Authorized operator');
    document.getElementById('dab-auth-trigger-avatar').innerHTML = avatarMarkup(user);
    document.getElementById('dab-auth-menu-avatar').innerHTML = avatarMarkup(user, true);
    document.getElementById('dab-auth-trigger-email').textContent = email || user.name || 'Local operator';
    document.getElementById('dab-auth-name').textContent = user.name || (localMode ? 'Local operator' : 'Google account');
    document.getElementById('dab-auth-email').textContent = email || 'Controller-local session';
    document.getElementById('dab-auth-account').textContent = localMode ? 'Local mode' : 'Google';
    document.getElementById('dab-auth-domain').textContent = domain;
    document.getElementById('dab-auth-session').textContent = 'Active';
    document.getElementById('dab-auth-logout').classList.toggle('hidden', localMode);
    document.getElementById('dab-auth-login-status').classList.toggle('hidden', !localMode);
    profile.classList.remove('hidden');
    void refreshProfile();
  }

  async function refreshProfile() {
    try {
      const response = await fetch(url('/auth/profile'), { credentials: apiCredentialsMode, cache: 'no-store' });
      if (!response.ok) return;
      const profile = await response.json();
      const jobs = profile.jobs || {};
      document.getElementById('dab-auth-job-count').textContent = String(jobs.total || 0);
      document.getElementById('dab-auth-artifact-count').textContent = String(jobs.artifacts || 0);
    } catch (_) {
      // The account menu remains useful even if profile totals are temporarily unavailable.
    }
  }

  function showLogin(message = '') {
    ensureUi();
    document.getElementById('dab-auth-message').textContent = message;
    document.getElementById('dab-auth-overlay').classList.remove('hidden');
    document.getElementById('dab-auth-profile').classList.add('hidden');
  }

  async function post(path, body) {
    const response = await fetch(url(path), { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : null });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    return payload;
  }

  function loadGoogleButton(clientId) {
    if (googleScriptRequested) return;
    googleScriptRequested = true;
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.onload = () => {
      google.accounts.id.initialize({ client_id: clientId, callback: async ({ credential }) => {
        try { const result = await post('/auth/google', { credential }); showUser(result.user); resolveReady(result.user); }
        catch (error) { showLogin(error.message || String(error)); }
      }});
      google.accounts.id.renderButton(document.getElementById('dab-google-signin'), { theme: 'outline', size: 'large', width: Math.min(340, window.innerWidth - 96) });
    };
    script.onerror = () => showLogin('Unable to load Google sign-in. Check network access to accounts.google.com.');
    document.head.appendChild(script);
  }

  async function logout() { try { await post('/auth/logout'); } finally { window.location.reload(); } }

  async function bootstrap() {
    ensureUi();
    const authConfigUrl = url('/auth/config');
    if (window.location.protocol === 'https:' && authConfigUrl.startsWith('http://')) {
      showLogin(`This HTTPS page cannot call the HTTP controller at ${authConfigUrl}. Open http://10.99.57.66:5173/index.html on the lab network, or expose the controller through HTTPS.`);
      return;
    }
    try {
      let response = await fetch(url('/auth/config'), { credentials: 'same-origin', cache: 'no-store' });
      let config = await response.json();
      if (!config.enabled) { showUser(config.user); resolveReady(config.user); return; }
      apiCredentialsMode = 'include';
      if (config.configured && apiBase) {
        response = await fetch(url('/auth/config'), { credentials: 'include', cache: 'no-store' });
        config = await response.json();
      }
      if (config.user) { showUser(config.user); resolveReady(config.user); return; }
      showLogin(config.configured ? '' : 'Google authentication is enabled but the backend configuration is incomplete.');
      if (config.client_id) loadGoogleButton(config.client_id);
    } catch (error) {
      showLogin(`Authentication service unavailable at ${url('/auth/config')}: ${error.message || error}`);
    }
  }

  window.dabAuth = { ready, currentUser: () => currentUser, credentialsMode: () => apiCredentialsMode, logout, toggleProfileMenu };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootstrap, { once: true }); else bootstrap();
})();
