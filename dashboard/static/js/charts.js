/**
 * LLMQ Dashboard — Charts
 * Chart.js configuration and rendering helpers.
 */
const Charts = (() => {
    // Chart instances (for destroy/recreate)
    let trendsChart = null;
    let compareChart = null;
    let compareDetailChart = null;
    let radarChart = null;
    let ciTimelineChart = null;

    // Shared chart defaults
    const gridColor = 'rgba(99,102,241,0.06)';
    const tickColor = '#64748b';
    const fontFamily = "'Inter', 'JetBrains Mono', sans-serif";

    Chart.defaults.font.family = fontFamily;
    Chart.defaults.font.size = 11;
    Chart.defaults.color = tickColor;

    const baseOpts = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'bottom',
                labels: {
                    color: '#94a3b8',
                    padding: 16,
                    usePointStyle: true,
                    pointStyleWidth: 8,
                    font: { size: 11 },
                },
            },
            tooltip: {
                backgroundColor: 'rgba(15,17,32,0.95)',
                borderColor: 'rgba(99,102,241,0.2)',
                borderWidth: 1,
                titleFont: { weight: '600' },
                padding: 10,
                cornerRadius: 8,
                displayColors: true,
            },
        },
        scales: {
            x: { ticks: { color: tickColor }, grid: { color: gridColor, drawBorder: false } },
            y: { ticks: { color: tickColor }, grid: { color: gridColor, drawBorder: false }, min: 0, max: 100 },
        },
    };

    function _mergeOpts(overrides) {
        return JSON.parse(JSON.stringify({ ...baseOpts, ...overrides }));
    }

    // ── Overview Trends Chart ──
    function renderTrends(runs) {
        const ctx = document.getElementById('chart-trends');
        if (!ctx) return;
        const labels = runs.map((_, i) => `Run ${i + 1}`).reverse();
        const overall = runs.map(r => ((r.overall_score || 0) * 100).toFixed(1)).reverse();
        const task = runs.map(r => ((r.task_success_score || 0) * 100).toFixed(1)).reverse();
        const rel = runs.map(r => ((r.relevance_score || 0) * 100).toFixed(1)).reverse();

        if (trendsChart) trendsChart.destroy();
        trendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Overall', data: overall, borderColor: '#818cf8', backgroundColor: 'rgba(129,140,248,0.08)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 3, pointHoverRadius: 6 },
                    { label: 'Task Success', data: task, borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.05)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 2, pointHoverRadius: 5 },
                    { label: 'Relevance', data: rel, borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.05)', fill: false, tension: 0.4, borderWidth: 1.5, pointRadius: 2, borderDash: [5, 3] },
                ],
            },
            options: _mergeOpts({}),
        });
    }

    // ── Overview Compare (bar) ──
    function renderCompareOverview(runs) {
        const ctx = document.getElementById('chart-compare');
        if (!ctx) return;
        const grouped = {};
        runs.forEach(r => {
            const key = `${r.provider_name}/${r.model_name}`;
            grouped[key] = grouped[key] || [];
            grouped[key].push(r.overall_score || 0);
        });
        const labels = Object.keys(grouped);
        const data = labels.map(k => ((grouped[k].reduce((a, b) => a + b, 0) / grouped[k].length) * 100).toFixed(1));

        if (compareChart) compareChart.destroy();
        compareChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Avg Overall %',
                    data,
                    backgroundColor: labels.map((_, i) => {
                        const colors = ['rgba(99,102,241,0.7)', 'rgba(34,211,238,0.7)', 'rgba(168,85,247,0.7)', 'rgba(52,211,153,0.7)', 'rgba(251,146,60,0.7)'];
                        return colors[i % colors.length];
                    }),
                    borderColor: labels.map((_, i) => {
                        const colors = ['#6366f1', '#22d3ee', '#a855f7', '#34d399', '#fb923c'];
                        return colors[i % colors.length];
                    }),
                    borderWidth: 1,
                    borderRadius: 6,
                    borderSkipped: false,
                }],
            },
            options: _mergeOpts({ plugins: { ...baseOpts.plugins, legend: { display: false } } }),
        });
    }

    // ── Compare Page Charts ──
    function renderCompareDetail(comparisons) {
        const ctx = document.getElementById('chart-compare-detail');
        if (!ctx) return;
        const labels = comparisons.map(c => c.provider_model);
        const scores = comparisons.map(c => ((c.overall_score || 0) * 100).toFixed(1));
        const colors = ['#6366f1', '#22d3ee', '#a855f7', '#34d399', '#fb923c', '#f472b6', '#facc15'];

        if (compareDetailChart) compareDetailChart.destroy();
        compareDetailChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Overall Score %',
                    data: scores,
                    backgroundColor: labels.map((_, i) => colors[i % colors.length] + 'cc'),
                    borderColor: labels.map((_, i) => colors[i % colors.length]),
                    borderWidth: 1.5,
                    borderRadius: 8,
                    borderSkipped: false,
                }],
            },
            options: _mergeOpts({ plugins: { ...baseOpts.plugins, legend: { display: false } } }),
        });
    }

    function renderRadar(comparisons) {
        const ctx = document.getElementById('chart-radar');
        if (!ctx) return;
        const labels = ['Task Success', 'Relevance', 'Consistency', 'Factuality', 'Gate Rate'];
        const colors = ['#6366f1', '#22d3ee', '#a855f7', '#34d399', '#fb923c', '#f472b6'];

        const datasets = comparisons.slice(0, 5).map((c, i) => ({
            label: c.provider_model,
            data: [
                ((c.task_success_score || 0) * 100).toFixed(1),
                ((c.relevance_score || 0) * 100).toFixed(1),
                ((c.consistency_score || 0) * 100).toFixed(1),
                (100 - (c.hallucination_score || 0) * 100).toFixed(1),
                ((c.quality_gate_pass_rate || 0) * 100).toFixed(1),
            ],
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '18',
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 6,
        }));

        if (radarChart) radarChart.destroy();
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#94a3b8', usePointStyle: true, font: { size: 10 } } },
                    tooltip: baseOpts.plugins.tooltip,
                },
                scales: {
                    r: {
                        grid: { color: gridColor },
                        angleLines: { color: gridColor },
                        ticks: { color: tickColor, backdropColor: 'transparent', font: { size: 9 } },
                        pointLabels: { color: '#94a3b8', font: { size: 11 } },
                        min: 0,
                        max: 100,
                    },
                },
            },
        });
    }

    // ── CI Timeline ──
    function renderCITimeline(gates) {
        const ctx = document.getElementById('chart-ci-timeline');
        if (!ctx) return;
        const labels = gates.map((g, i) => g.commit_hash ? g.commit_hash.slice(0, 7) : `#${gates.length - i}`).reverse();
        const scores = gates.map(g => ((g.overall_score || 0) * 100).toFixed(1)).reverse();
        const bgColors = gates.map(g => g.passed ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)').reverse();
        const borderColors = gates.map(g => g.passed ? '#22c55e' : '#ef4444').reverse();

        if (ciTimelineChart) ciTimelineChart.destroy();
        ciTimelineChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Gate Score %',
                    data: scores,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: _mergeOpts({ plugins: { ...baseOpts.plugins, legend: { display: false } } }),
        });
    }

    // ── Score Gauge ──
    function renderGauge(score) {
        const canvas = document.getElementById('gauge-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const size = canvas.width;
        const cx = size / 2;
        const cy = size / 2;
        const radius = size / 2 - 20;
        const lineWidth = 14;
        const startAngle = 0.75 * Math.PI;
        const endAngle = 2.25 * Math.PI;
        const pct = Math.min(Math.max(score, 0), 1);

        ctx.clearRect(0, 0, size, size);

        // Background arc
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, endAngle);
        ctx.strokeStyle = 'rgba(99,102,241,0.08)';
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Value arc with gradient
        const valAngle = startAngle + (endAngle - startAngle) * pct;
        const grad = ctx.createLinearGradient(0, size, size, 0);
        if (pct >= 0.7) {
            grad.addColorStop(0, '#6366f1');
            grad.addColorStop(1, '#22d3ee');
        } else if (pct >= 0.4) {
            grad.addColorStop(0, '#f59e0b');
            grad.addColorStop(1, '#fb923c');
        } else {
            grad.addColorStop(0, '#ef4444');
            grad.addColorStop(1, '#f87171');
        }

        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, valAngle);
        ctx.strokeStyle = grad;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Glow effect
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, valAngle);
        ctx.strokeStyle = grad;
        ctx.lineWidth = lineWidth + 8;
        ctx.lineCap = 'round';
        ctx.globalAlpha = 0.15;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Label
        document.getElementById('gauge-label').textContent = `${(pct * 100).toFixed(1)}%`;
    }

    return {
        renderTrends,
        renderCompareOverview,
        renderCompareDetail,
        renderRadar,
        renderCITimeline,
        renderGauge,
    };
})();
