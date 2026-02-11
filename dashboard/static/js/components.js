/**
 * LLMQ Dashboard — UI Components
 * Reusable DOM renderer helpers.
 */
const Components = (() => {

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    function pct(v, d = 1) { return `${((v || 0) * 100).toFixed(d)}%`; }

    function relativeTime(dateStr) {
        if (!dateStr) return '--';
        const d = new Date(dateStr);
        const now = new Date();
        const diff = (now - d) / 1000;
        if (diff < 60) return 'just now';
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    }

    // ── Toast Notification ──
    function toast(message, level = 'info') {
        const container = document.getElementById('toast-container');
        const el = document.createElement('div');
        el.className = `toast toast-${level}`;
        el.innerHTML = `
      <svg class="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
        ${level === 'pass' ? '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>'
                : level === 'fail' ? '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>'
                    : '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>'}
      </svg>
      <span>${escapeHtml(message)}</span>
    `;
        container.appendChild(el);
        setTimeout(() => el.remove(), 3200);
    }

    // ── Metric Card Update ──
    function updateMetricCard(id, value, barId, rawValue) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
        if (barId) {
            const bar = document.getElementById(barId);
            if (bar) bar.style.width = `${Math.min((rawValue || 0) * 100, 100)}%`;
        }
    }

    // ── Gate Status Badge ──
    function updateGateStatus(passed) {
        const el = document.getElementById('gate-status');
        const text = document.getElementById('gate-status-text');
        el.className = `gate-indicator ${passed === true ? 'gate-pass' : passed === false ? 'gate-fail' : 'gate-unknown'}`;
        text.textContent = passed === true ? 'PASS' : passed === false ? 'FAIL' : 'No Data';
    }

    // ── Provider/Model Badges ──
    function updateBadges(provider, model) {
        const pb = document.getElementById('provider-badge');
        const mb = document.getElementById('model-badge');
        if (provider) {
            pb.classList.remove('hidden');
            document.getElementById('provider-badge-text').textContent = provider;
        }
        if (model) {
            mb.classList.remove('hidden');
            document.getElementById('model-badge-text').textContent = model;
        }
    }

    // ── Run Table Row ──
    function renderRunRow(r, delta, regression) {
        return `<tr class="clickable" onclick="LLMQ.openRunDetail('${r.id}')">
      <td>
        <div class="flex items-center gap-2">
          <span class="text-brand-400 font-medium">${escapeHtml(r.provider_name)}</span>
          <span class="text-gray-500">/</span>
          <span>${escapeHtml(r.model_name)}</span>
        </div>
      </td>
      <td class="text-gray-500 font-mono text-xs">${escapeHtml(r.dataset_version || '-')}</td>
      <td class="font-mono text-xs text-gray-500">${(r.commit_hash || '-').slice(0, 8)}</td>
      <td class="text-center font-mono font-semibold">${pct(r.overall_score)}</td>
      <td class="text-center"><span class="gate-pill ${r.quality_gate_passed ? 'gate-pill-pass' : 'gate-pill-fail'}">${r.quality_gate_passed ? 'PASS' : 'FAIL'}</span></td>
      <td class="text-center">${delta === null ? '<span class="text-gray-600">—</span>' : `<span class="${regression ? 'text-red-400' : 'text-emerald-400'} font-mono text-xs font-semibold">${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}%</span>${regression ? ' <span class="text-amber-400">⚠</span>' : ''}`}</td>
      <td class="text-xs text-gray-500">${relativeTime(r.created_at)}</td>
    </tr>`;
    }

    // ── Run Detail Modal Content ──
    function renderRunDetail(run, testCases) {
        const r = run || {};
        const metrics = [
            { label: 'Task Success', value: r.task_success_score, color: 'emerald' },
            { label: 'Relevance', value: r.relevance_score, color: 'blue' },
            { label: 'Hallucination', value: r.hallucination_score, color: 'rose' },
            { label: 'Consistency', value: r.consistency_score, color: 'violet' },
        ];

        let html = `
      <div class="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div class="flex items-center gap-3">
          <span class="gate-pill ${r.quality_gate_passed ? 'gate-pill-pass' : 'gate-pill-fail'}">${r.quality_gate_passed ? 'PASS' : 'FAIL'}</span>
          <span class="font-mono text-sm text-gray-400">${(r.id || '').slice(0, 12)}</span>
          <span class="text-gray-600">·</span>
          <span class="text-sm text-gray-400">${escapeHtml(r.provider_name || '')} / ${escapeHtml(r.model_name || '')}</span>
        </div>
        <button onclick="LLMQ.markBaseline('${r.id}')" class="btn btn-primary text-xs">Mark as Baseline</button>
      </div>

      <!-- Metric Breakdown (mini radar) -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        ${metrics.map(m => `
          <div class="bg-surface-3/30 rounded-lg p-3 text-center">
            <div class="text-[10px] text-gray-500 uppercase tracking-wider mb-1">${m.label}</div>
            <div class="text-lg font-bold font-mono text-${m.color}-400">${pct(m.value)}</div>
          </div>
        `).join('')}
      </div>

      <div class="flex items-center gap-2 mb-3">
        <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Overall Score</div>
        <div class="font-mono font-bold text-brand-400">${pct(r.overall_score)}</div>
        ${r.regression_detected ? '<span class="ml-2 gate-pill gate-pill-fail text-[10px]">Regression Detected</span>' : ''}
      </div>
    `;

        // Test cases
        if (testCases && testCases.length) {
            html += `<div class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 mt-6">Test Cases (${testCases.length})</div>`;
            html += '<div class="space-y-2">';
            testCases.forEach((tc, i) => {
                const failed = !tc.success;
                html += `
          <div class="test-case-card ${failed ? 'failed' : ''} ${i === 0 ? 'open' : ''}" onclick="this.classList.toggle('open')">
            <div class="test-case-header">
              <span class="text-gray-500 text-xs">${i === 0 ? '▼' : '▶'}</span>
              <span class="font-mono text-xs text-gray-400">${escapeHtml(tc.test_case_id)}</span>
              <span>${tc.success ? '✅' : '❌'}</span>
              <span class="text-xs text-gray-500">${escapeHtml(tc.task_type || '')}</span>
              <span class="ml-auto text-xs font-mono ${failed ? 'text-red-400' : 'text-emerald-400'}">${pct(tc.task_success_score)}</span>
            </div>
            <div class="test-case-body">
              <div class="grid grid-cols-3 gap-3 mb-3">
                <div class="text-center">
                  <div class="text-[10px] text-gray-500">Task</div>
                  <div class="font-mono text-sm text-emerald-400">${pct(tc.task_success_score)}</div>
                </div>
                <div class="text-center">
                  <div class="text-[10px] text-gray-500">Relevance</div>
                  <div class="font-mono text-sm text-blue-400">${pct(tc.relevance_score)}</div>
                </div>
                <div class="text-center">
                  <div class="text-[10px] text-gray-500">Hallucination</div>
                  <div class="font-mono text-sm text-rose-400">${pct(tc.hallucination_score)}</div>
                </div>
              </div>
              <div class="side-by-side">
                <div class="side-by-side-panel">
                  <div class="side-by-side-label text-gray-500">Model Output</div>
                  <div class="text-xs text-gray-300 leading-relaxed">${escapeHtml(tc.generated_output || 'N/A')}</div>
                </div>
                <div class="side-by-side-panel">
                  <div class="side-by-side-label text-cyan-500">Expected Output</div>
                  <div class="text-xs text-gray-300 leading-relaxed">${escapeHtml(tc.expected_output || 'N/A')}</div>
                </div>
              </div>
              ${tc.input_prompt ? `<div class="mt-3 text-xs"><span class="text-gray-500 font-medium">Prompt:</span> <span class="text-gray-400">${escapeHtml(tc.input_prompt)}</span></div>` : ''}
            </div>
          </div>
        `;
            });
            html += '</div>';
        } else {
            html += '<p class="text-gray-500 text-sm">No test case details available.</p>';
        }

        return html;
    }

    // ── Compare Table Row ──
    function renderCompareRow(c, idx) {
        const isBest = idx === 0;
        return `<tr class="${isBest ? 'bg-brand/5' : ''}">
      <td>
        <div class="flex items-center gap-2">
          ${isBest ? '<span class="text-amber-400 text-xs">👑</span>' : ''}
          <span class="font-medium">${escapeHtml(c.provider_model)}</span>
        </div>
      </td>
      <td class="text-center text-gray-400">${c.runs}</td>
      <td class="text-center font-mono font-semibold">${pct(c.overall_score)}</td>
      <td class="text-center font-mono text-emerald-400">${pct(c.task_success_score)}</td>
      <td class="text-center font-mono text-blue-400">${pct(c.relevance_score)}</td>
      <td class="text-center font-mono text-rose-400">${pct(c.hallucination_score)}</td>
      <td class="text-center font-mono text-brand-400">${pct(c.quality_gate_pass_rate, 0)}</td>
      <td class="text-center">
        <span class="${(c.overall_score || 0) >= 0.8 ? 'text-emerald-400' : (c.overall_score || 0) >= 0.6 ? 'text-amber-400' : 'text-red-400'} text-xs font-semibold">
          ${(c.overall_score || 0) >= 0.8 ? '↑ Strong' : (c.overall_score || 0) >= 0.6 ? '→ Stable' : '↓ Weak'}
        </span>
      </td>
    </tr>`;
    }

    // ── CI Gate Event Row ──
    function renderCIRow(g) {
        const failedMetrics = Array.isArray(g.failed_metrics)
            ? g.failed_metrics
            : (typeof g.failed_metrics === 'string' ? (() => { try { return JSON.parse(g.failed_metrics); } catch { return []; } })() : []);
        return `<tr>
      <td class="font-mono text-xs text-gray-400">${(g.run_id || g.id || '').slice(0, 10)}</td>
      <td class="text-center"><span class="gate-pill ${g.passed ? 'gate-pill-pass' : 'gate-pill-fail'}">${g.passed ? 'PASS' : 'FAIL'}</span></td>
      <td class="text-center font-mono">${pct(g.overall_score)}</td>
      <td>${failedMetrics.length ? failedMetrics.map(m => `<span class="inline-block bg-red-900/30 text-red-300 text-[10px] px-1.5 py-0.5 rounded mr-1">${escapeHtml(m)}</span>`).join('') : '<span class="text-gray-600">—</span>'}</td>
      <td class="font-mono text-xs text-gray-500">${(g.commit_hash || '-').slice(0, 8)}</td>
      <td class="text-xs text-gray-500">${relativeTime(g.created_at)}</td>
    </tr>`;
    }

    // ── Last Run Summary ──
    function renderLastRunSummary(run) {
        if (!run) return '<div class="text-sm text-gray-500">No runs available yet.</div>';
        return `
      <div class="flex flex-wrap items-center gap-3 mb-3">
        <span class="gate-pill ${run.quality_gate_passed ? 'gate-pill-pass' : 'gate-pill-fail'}">${run.quality_gate_passed ? 'PASS' : 'FAIL'}</span>
        <div class="badge badge-provider"><span class="badge-dot bg-brand-400"></span>${escapeHtml(run.provider_name)}</div>
        <div class="badge badge-model">${escapeHtml(run.model_name)}</div>
        ${run.commit_hash ? `<span class="font-mono text-xs text-gray-500">@ ${run.commit_hash.slice(0, 8)}</span>` : ''}
        ${run.regression_detected ? '<span class="gate-pill gate-pill-fail text-[10px]">⚠ Regression</span>' : ''}
      </div>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div class="text-center">
          <div class="text-[10px] text-gray-500 uppercase">Overall</div>
          <div class="font-mono font-bold text-brand-400 text-lg">${pct(run.overall_score)}</div>
        </div>
        <div class="text-center">
          <div class="text-[10px] text-gray-500 uppercase">Task</div>
          <div class="font-mono font-semibold text-emerald-400">${pct(run.task_success_score)}</div>
        </div>
        <div class="text-center">
          <div class="text-[10px] text-gray-500 uppercase">Relevance</div>
          <div class="font-mono font-semibold text-blue-400">${pct(run.relevance_score)}</div>
        </div>
        <div class="text-center">
          <div class="text-[10px] text-gray-500 uppercase">Halluc.</div>
          <div class="font-mono font-semibold text-rose-400">${pct(run.hallucination_score)}</div>
        </div>
        <div class="text-center">
          <div class="text-[10px] text-gray-500 uppercase">Tests</div>
          <div class="font-mono font-semibold text-gray-300">${run.successful_executions || 0}/${run.total_test_cases || 0}</div>
        </div>
      </div>
      <div class="mt-3 text-xs text-gray-500">
        Dataset ${escapeHtml(run.dataset_version || '-')} · ${relativeTime(run.created_at)}
      </div>
    `;
    }

    // ── Live Jobs ──
    function renderLiveJobs(jobs) {
        if (!jobs || !jobs.length) return '<div class="text-sm text-gray-500">No active evaluations.</div>';
        return jobs.map(j => {
            const pctVal = j.total ? Math.round((j.progress / j.total) * 100) : 0;
            return `
        <div class="live-job">
          <div class="flex items-center justify-between">
            <div>
              <span class="text-sm font-medium">${escapeHtml(j.provider)}/${escapeHtml(j.model || '-')}</span>
              <span class="font-mono text-xs text-gray-500 ml-2">${(j.job_id || '').slice(0, 8)}</span>
            </div>
            <span class="text-xs font-mono text-brand-400">${j.progress}/${j.total} (${pctVal}%)</span>
          </div>
          <div class="live-job-progress">
            <div class="live-job-progress-bar" style="width: ${pctVal}%"></div>
          </div>
        </div>
      `;
        }).join('');
    }

    // ── Hydrate Select ──
    function hydrateSelect(id, values) {
        const el = document.getElementById(id);
        if (!el || !el.querySelector('option')) return;
        const first = el.querySelector('option').outerHTML;
        el.innerHTML = first + values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
    }

    return {
        escapeHtml, pct, relativeTime, toast,
        updateMetricCard, updateGateStatus, updateBadges,
        renderRunRow, renderRunDetail, renderCompareRow, renderCIRow,
        renderLastRunSummary, renderLiveJobs, hydrateSelect,
    };
})();
