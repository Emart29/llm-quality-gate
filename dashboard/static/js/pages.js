/**
 * LLMQ Dashboard — Page Controllers
 * Each page's data-loading / rendering logic.
 */
const Pages = (() => {

    // ── Overview ──
    async function loadOverview() {
        const data = await API.getOverview();
        if (!data) return;
        State.set('overview', data);

        // KPI cards
        Components.updateMetricCard('kpi-task', Components.pct(data.avg_task_success), 'bar-task', data.avg_task_success);
        Components.updateMetricCard('kpi-rel', Components.pct(data.avg_relevance), 'bar-rel', data.avg_relevance);
        Components.updateMetricCard('kpi-hal', Components.pct(data.avg_hallucination), 'bar-hal', data.avg_hallucination);
        Components.updateMetricCard('kpi-cons', Components.pct(data.avg_consistency || 0), 'bar-cons', data.avg_consistency || 0);
        Components.updateMetricCard('kpi-gate', Components.pct(data.quality_gate_pass_rate, 0), 'bar-gate', data.quality_gate_pass_rate);

        // Charts
        const recent = data.recent_runs || [];
        Charts.renderTrends(recent);
        Charts.renderCompareOverview(recent);

        // Gate status
        const lastRun = recent[0];
        if (lastRun) {
            State.set('lastRun', lastRun);
            Components.updateGateStatus(lastRun.quality_gate_passed);
            Components.updateBadges(lastRun.provider_name, lastRun.model_name);

            // Gauge
            Charts.renderGauge(data.avg_success_rate || lastRun.overall_score || 0);

            // Last run summary
            document.getElementById('last-run-summary').innerHTML = Components.renderLastRunSummary(lastRun);
        }
    }

    // ── Runs ──
    async function loadFilters() {
        const data = await API.getRunFilters();
        if (!data) return;
        State.set('filters', data);
        Components.hydrateSelect('filter-provider', data.providers || []);
        Components.hydrateSelect('filter-model', data.models || []);
        Components.hydrateSelect('filter-dataset', data.dataset_versions || []);
        Components.hydrateSelect('filter-commit', data.commit_hashes || []);
    }

    async function loadRuns() {
        const params = {};
        const val = id => document.getElementById(id)?.value;
        if (val('filter-provider')) params.provider = val('filter-provider');
        if (val('filter-model')) params.model = val('filter-model');
        if (val('filter-dataset')) params.dataset_version = val('filter-dataset');
        if (val('filter-status')) params.quality_gate_passed = val('filter-status');
        if (val('filter-commit')) params.commit_hash = val('filter-commit');

        const data = await API.getRuns(params);
        if (!data) return;
        const runs = data.runs || [];
        State.set('runs', runs);

        renderRunsTable(runs);

        // Toast for latest
        const latest = runs[0];
        if (latest?.quality_gate_passed === true) Components.toast(`Run ${latest.id.slice(0, 8)} passed quality gate`, 'pass');
        else if (latest?.quality_gate_passed === false) Components.toast(`Run ${latest.id.slice(0, 8)} failed quality gate`, 'fail');
    }

    function renderRunsTable(runs) {
        const tbody = document.getElementById('runs-body');
        if (!tbody) return;
        const count = document.getElementById('runs-count');
        if (count) count.textContent = `${runs.length} runs`;

        tbody.innerHTML = runs.map((r, idx) => {
            const prev = runs.slice(idx + 1).find(x => x.provider_name === r.provider_name && x.model_name === r.model_name);
            const delta = prev ? ((r.overall_score || 0) - (prev.overall_score || 0)) : null;
            const regression = r.regression_detected || (delta !== null && delta < -0.03);
            return Components.renderRunRow(r, delta, regression);
        }).join('');
    }

    // Client-side sorting
    function sortRuns(field) {
        const current = State.get('runsSortField');
        const asc = current === field ? !State.get('runsSortAsc') : false;
        State.set('runsSortField', field);
        State.set('runsSortAsc', asc);

        const runs = [...State.get('runs')];
        runs.sort((a, b) => {
            let va, vb;
            switch (field) {
                case 'provider': va = `${a.provider_name}/${a.model_name}`; vb = `${b.provider_name}/${b.model_name}`; break;
                case 'dataset': va = a.dataset_version || ''; vb = b.dataset_version || ''; break;
                case 'commit': va = a.commit_hash || ''; vb = b.commit_hash || ''; break;
                case 'score': va = a.overall_score || 0; vb = b.overall_score || 0; break;
                case 'gate': va = a.quality_gate_passed ? 1 : 0; vb = b.quality_gate_passed ? 1 : 0; break;
                case 'date': va = a.created_at || ''; vb = b.created_at || ''; break;
                default: va = ''; vb = '';
            }
            if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va);
            return asc ? va - vb : vb - va;
        });

        renderRunsTable(runs);

        // Update sort indicators
        document.querySelectorAll('.sortable').forEach(th => {
            th.classList.toggle('sorted', th.dataset.sort === field);
            const icon = th.querySelector('.sort-icon');
            if (icon) icon.textContent = th.dataset.sort === field ? (asc ? '↑' : '↓') : '↕';
        });
    }

    // ── Run Detail ──
    async function openRunDetail(runId) {
        const modal = document.getElementById('run-detail-modal');
        const body = document.getElementById('run-detail-body');
        modal.classList.remove('hidden');
        body.innerHTML = '<div class="text-gray-500 text-sm">Loading...</div>';

        const data = await API.getRunDetail(runId);
        if (!data) {
            body.innerHTML = '<div class="text-red-400">Failed to load run details.</div>';
            return;
        }
        body.innerHTML = Components.renderRunDetail(data.run, data.test_cases);
    }

    function closeModal() {
        document.getElementById('run-detail-modal').classList.add('hidden');
    }

    // ── Compare ──
    async function loadCompare() {
        const data = await API.getCompare();
        if (!data) return;
        const comparisons = data.comparisons || [];
        State.set('compare', comparisons);

        Charts.renderCompareDetail(comparisons);
        Charts.renderRadar(comparisons);

        const tbody = document.getElementById('compare-body');
        if (tbody) {
            tbody.innerHTML = comparisons.map((c, i) => Components.renderCompareRow(c, i)).join('');
        }
    }

    // ── CI History ──
    async function loadCIHistory() {
        const data = await API.getQualityGates();
        if (!data) return;
        const gates = data.gates || [];
        State.set('qualityGates', gates);

        Charts.renderCITimeline(gates);

        const tbody = document.getElementById('ci-body');
        if (tbody) {
            tbody.innerHTML = gates.map(g => Components.renderCIRow(g)).join('');
        }
    }

    // ── Settings ──
    async function loadSettings() {
        const [providersRes, settingsRes] = await Promise.all([API.getProviders(), API.getSettings()]);
        const providers = (providersRes?.providers || []).map(p => p.name);
        State.set('providers', providers);

        const options = providers.map(p => `<option value="${p}">${p}</option>`).join('');
        const gp = document.getElementById('generator-provider');
        const jp = document.getElementById('judge-provider');
        if (gp) gp.innerHTML = options;
        if (jp) jp.innerHTML = options;

        if (settingsRes) {
            State.set('settings', settingsRes);
            const val = (id, v) => { const el = document.getElementById(id); if (el) el.value = v || ''; };
            val('generator-provider', settingsRes.roles?.generator?.provider);
            val('judge-provider', settingsRes.roles?.judge?.provider);
            val('generator-model', settingsRes.roles?.generator?.model);
            val('judge-model', settingsRes.roles?.judge?.model);
            val('th-task', settingsRes.quality_gates?.task_success_threshold ?? 0.8);
            val('th-rel', settingsRes.quality_gates?.relevance_threshold ?? 0.7);
            val('th-hal', settingsRes.quality_gates?.hallucination_threshold ?? 0.1);
            val('th-cons', settingsRes.quality_gates?.consistency_threshold ?? 0.8);

            // CI mode detection
            if (settingsRes.evaluation?.ci_lightweight_mode) {
                document.getElementById('ci-banner')?.classList.remove('hidden');
            }
        }
    }

    async function saveSettings() {
        const val = id => document.getElementById(id)?.value || '';
        const num = id => Number(document.getElementById(id)?.value || 0);
        const payload = {
            roles: {
                generator: { provider: val('generator-provider'), model: val('generator-model') },
                judge: { provider: val('judge-provider'), model: val('judge-model') },
            },
            quality_gates: {
                task_success_threshold: num('th-task'),
                relevance_threshold: num('th-rel'),
                hallucination_threshold: num('th-hal'),
                consistency_threshold: num('th-cons'),
            },
        };
        await API.saveSettings(payload);
        Components.toast('Settings saved successfully.', 'pass');
    }

    async function startEvaluation() {
        const provider = document.getElementById('generator-provider')?.value;
        const model = document.getElementById('generator-model')?.value;
        if (!provider) { Components.toast('Select a provider first.', 'fail'); return; }
        const data = await API.startEvaluation({ provider, model });
        if (data?.job_id) {
            Components.toast(`Evaluation started: ${data.job_id.slice(0, 8)}`, 'info');
            loadLiveJobs();
        }
    }

    async function markBaseline(runId) {
        const data = await API.markBaseline(runId);
        if (data?.error) {
            Components.toast(`Failed: ${data.error}`, 'fail');
        } else {
            Components.toast(`Baseline set for ${runId.slice(0, 8)}`, 'pass');
        }
    }

    // ── Live Jobs ──
    async function loadLiveJobs() {
        const data = await API.getActiveJobs();
        const jobs = data?.jobs || [];
        State.set('activeJobs', jobs);
        const container = document.getElementById('live-jobs');
        if (container) container.innerHTML = Components.renderLiveJobs(jobs);
    }

    return {
        loadOverview, loadFilters, loadRuns, sortRuns,
        openRunDetail, closeModal,
        loadCompare, loadCIHistory,
        loadSettings, saveSettings, startEvaluation, markBaseline,
        loadLiveJobs,
    };
})();
