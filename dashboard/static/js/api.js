/**
 * LLMQ Dashboard — API Layer
 * Centralized fetch wrapper for all /api endpoints.
 */
const API = (() => {
    const BASE_V1 = '/api/v1';
    const BASE = '/api';

    async function _fetch(url, opts = {}) {
        try {
            const res = await fetch(url, opts);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`[API] ${url}`, err);
            return null;
        }
    }

    function _post(url, body) {
        return _fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    }

    return {
        // Overview
        getOverview: () => _fetch(`${BASE}/overview`),

        // Runs (v1)
        getRuns: (params = {}) => {
            const qs = new URLSearchParams({ limit: '80', ...params });
            return _fetch(`${BASE_V1}/runs?${qs}`);
        },
        getRunDetail: (id) => _fetch(`${BASE}/runs/${id}`),
        getRunFilters: () => _fetch(`${BASE_V1}/runs/filters`),
        markBaseline: (id) => _post(`${BASE_V1}/runs/${id}/baseline`, {}),

        // Compare
        getCompare: (limit = 100) => _fetch(`${BASE_V1}/compare?limit=${limit}`),

        // CI / Quality Gates
        getQualityGates: (limit = 50) => _fetch(`${BASE}/quality-gates?limit=${limit}`),

        // Providers & Settings
        getProviders: () => _fetch(`${BASE_V1}/providers`),
        getSettings: () => _fetch(`${BASE_V1}/settings`),
        saveSettings: (payload) => _post(`${BASE_V1}/settings`, payload),

        // Evaluations
        startEvaluation: (p) => _post(`${BASE_V1}/evaluate/start`, p),
        getActiveJobs: () => _fetch(`${BASE_V1}/evaluate/active`),
        getJobStatus: (id) => _fetch(`${BASE_V1}/evaluate/status/${id}`),

        // History
        getHistory: (provider, model, limit = 30) =>
            _fetch(`${BASE}/history?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}&limit=${limit}`),
    };
})();
