const dom = {
  badge: document.getElementById('live-badge'),
  latestNode: document.getElementById('latest-result'),
  warningNode: document.getElementById('latest-warning'),
  body: document.getElementById('sessions-body'),
  limit: document.getElementById('limit'),
  pollRate: document.getElementById('poll-rate'),
  refreshBtn: document.getElementById('refresh-btn'),
  lastUpdated: document.getElementById('last-updated'),
  sessionCount: document.getElementById('session-count'),
  statusCard: document.querySelector('.status-card')
};

const state = {
  latestFingerprint: null,
  isFetching: false,
  timer: null
};

function setBadge(message, level) {
  if (!dom.badge) {
    return;
  }
  dom.badge.classList.remove('live', 'alert', 'warn');
  dom.badge.classList.add(level);
  dom.badge.innerHTML = `<span class="dot"></span>${message}`;
}

function setWarning(message, level) {
  if (!dom.warningNode) {
    return;
  }
  dom.warningNode.textContent = message;
  dom.warningNode.className = `status-text status-${level}`;
}

function toBinary(value) {
  if (value === null || value === undefined || value === '') {
    return 'n/a';
  }
  const raw = String(value).trim();
  if (raw === '0' || raw === '1') {
    return raw;
  }
  const numeric = Number(raw);
  if (Number.isNaN(numeric)) {
    return raw;
  }
  return numeric ? '1' : '0';
}

function binaryPair(left, right) {
  return `${toBinary(left)}/${toBinary(right)}`;
}

function formatDate(value) {
  if (!value) {
    return 'n/a';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) {
    return String(value);
  }
  return parsed.toLocaleString([], {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
}

function createCell(value, cssClass = '') {
  const cell = document.createElement('td');
  if (cssClass.includes('status-chip')) {
    const chip = document.createElement('span');
    chip.className = cssClass;
    chip.textContent = value;
    cell.appendChild(chip);
    return cell;
  }
  cell.textContent = value;
  if (cssClass) {
    cell.className = cssClass;
  }
  return cell;
}

function renderRows(sessions) {
  if (!dom.body) {
    return;
  }
  dom.body.innerHTML = '';
  sessions.forEach((s) => {
    const headWarningCount = Number(s.head_warning_count || 0);
    const warnText = headWarningCount >= 2 ? 'void/retest' : 'ok';
    const warnClass = headWarningCount >= 2 ? 'status-chip status-chip--warn' : 'status-chip status-chip--ok';

    const row = document.createElement('tr');
    if (headWarningCount >= 2) {
      row.classList.add('row--warn');
    }

    row.appendChild(createCell(formatDate(s.created_at)));
    row.appendChild(createCell(s.subject_name || 'Unknown'));
    row.appendChild(createCell(binaryPair(s.lack_of_smooth_pursuit_left_binary, s.lack_of_smooth_pursuit_right_binary)));
    row.appendChild(createCell(binaryPair(s.nystagmus_prior_to_45_left_binary, s.nystagmus_prior_to_45_right_binary)));
    row.appendChild(createCell(binaryPair(s.distinct_nystagmus_max_deviation_left_binary, s.distinct_nystagmus_max_deviation_right_binary)));
    const vert = Number(s.vertical_nystagmus || 0);
    row.appendChild(createCell(Number.isFinite(vert) ? vert.toFixed(1) : 'n/a'));
    row.appendChild(createCell(warnText, warnClass));

    dom.body.appendChild(row);
  });
}

function renderLatest(sessions) {
  if (dom.sessionCount) {
    dom.sessionCount.textContent = `${sessions.length} sessions loaded`;
  }
  if (dom.lastUpdated) {
    dom.lastUpdated.textContent = `Last update: ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
  }

  if (!sessions.length) {
    dom.latestNode.textContent = 'No test yet.';
    setWarning('Awaiting first sync.', 'neutral');
    setBadge('Waiting', 'warn');
    dom.body.innerHTML = '';
    return;
  }

  const latest = sessions[0];
  const fp = `${latest.session_id || ''}|${latest.created_at || ''}`;
  const isNew = state.latestFingerprint && state.latestFingerprint !== fp;
  state.latestFingerprint = fp;

  const headWarningCount = Number(latest.head_warning_count || 0);
  const hasHeadWarning = headWarningCount >= 2;

  dom.latestNode.textContent =
    `${latest.subject_name || 'Unknown'} · ${formatDate(latest.created_at)} · ` +
    `SP ${binaryPair(latest.lack_of_smooth_pursuit_left_binary, latest.lack_of_smooth_pursuit_right_binary)} · ` +
    `Prior45 ${binaryPair(latest.nystagmus_prior_to_45_left_binary, latest.nystagmus_prior_to_45_right_binary)} · ` +
    `MaxDev ${binaryPair(latest.distinct_nystagmus_max_deviation_left_binary, latest.distinct_nystagmus_max_deviation_right_binary)} · ` +
    `Vert ${Number(latest.vertical_nystagmus || 0).toFixed(1)}`;

  if (hasHeadWarning) {
    setWarning('Head movement warning: HIGH. Result may be void.', 'warn');
    setBadge('Head warning', 'alert');
  } else {
    setWarning('Head movement warning: none.', 'ok');
    setBadge('Monitoring', 'live');
  }

  if (isNew) {
    if (dom.statusCard) {
      dom.statusCard.classList.remove('pulse');
      void dom.statusCard.offsetWidth;
      dom.statusCard.classList.add('pulse');
    }
  }

  renderRows(sessions);
}

async function fetchSessions() {
  if (state.isFetching) {
    return;
  }
  state.isFetching = true;
  if (dom.refreshBtn) {
    dom.refreshBtn.disabled = true;
  }
  setBadge('Syncing', 'warn');

  try {
    const limit = Number(dom.limit.value || 25);
    const res = await fetch(`/api/sessions?limit=${encodeURIComponent(limit)}`, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    const sessions = Array.isArray(data.items) ? data.items : [];
    renderLatest(sessions);
  } catch (error) {
    setBadge('Offline', 'warn');
    setWarning(`Unable to contact /api/sessions (${error.message})`, 'neutral');
  } finally {
    state.isFetching = false;
    if (dom.refreshBtn) {
      dom.refreshBtn.disabled = false;
    }
  }
}

function pollInterval() {
  const base = Number(dom.pollRate ? dom.pollRate.value : 2000);
  return document.hidden ? Math.max(base * 3, 6000) : base;
}

function scheduleNextPoll() {
  clearTimeout(state.timer);
  state.timer = window.setTimeout(async () => {
    await fetchSessions();
    scheduleNextPoll();
  }, pollInterval());
}

function restartPolling() {
  scheduleNextPoll();
  void fetchSessions();
}

function onRefreshClick() {
  restartPolling();
}

function onConfigChange() {
  restartPolling();
}

dom.pollRate.addEventListener('change', onConfigChange);
dom.limit.addEventListener('change', onConfigChange);
dom.refreshBtn.addEventListener('click', onRefreshClick);
function onVisibilityChange() {
  if (document.hidden) {
    setBadge('Background', 'warn');
  } else {
    setBadge('Monitoring', 'live');
    void fetchSessions();
  }
  restartPolling();
}

if (dom.pollRate) {
  dom.pollRate.addEventListener('change', onConfigChange);
}
if (dom.limit) {
  dom.limit.addEventListener('change', onConfigChange);
}
if (dom.refreshBtn) {
  dom.refreshBtn.addEventListener('click', onRefreshClick);
}
document.addEventListener('visibilitychange', onVisibilityChange);

void fetchSessions();
scheduleNextPoll();
