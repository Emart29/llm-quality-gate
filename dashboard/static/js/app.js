/**
 * LLMQ Dashboard — Main Application Controller
 * Navigation, routing, initialization, and polling.
 */
const LLMQ = (() => {
    let poller = null;

    // ── Navigation ──
    function navigate(page) {
        // Update nav items
        document.querySelectorAll('.nav-item').forEach(el => {
            el.classList.toggle('active', el.dataset.page === page);
        });

        // Update pages
        document.querySelectorAll('.page').forEach(el => {
            el.classList.toggle('active', el.id === `page-${page}`);
        });

        // Update page title
        const titles = {
            'overview': 'Overview',
            'runs': 'Evaluation Runs',
            'compare': 'Provider Comparison',
            'ci-history': 'CI History',
            'settings': 'Settings',
        };
        document.getElementById('page-title').textContent = titles[page] || page;
        State.set('currentPage', page);

        // Load page data on first visit
        switch (page) {
            case 'overview': Pages.loadOverview(); break;
            case 'runs': Pages.loadRuns(); break;
            case 'compare': Pages.loadCompare(); break;
            case 'ci-history': Pages.loadCIHistory(); break;
            case 'settings': Pages.loadSettings(); break;
        }

        // Close mobile sidebar
        document.getElementById('sidebar')?.classList.remove('open');
        document.getElementById('sidebar-overlay')?.remove();
    }

    // ── Sidebar Toggle (mobile) ──
    function toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('open');
        if (sidebar.classList.contains('open')) {
            const overlay = document.createElement('div');
            overlay.id = 'sidebar-overlay';
            overlay.className = 'sidebar-overlay';
            overlay.onclick = () => {
                sidebar.classList.remove('open');
                overlay.remove();
            };
            document.body.appendChild(overlay);
        } else {
            document.getElementById('sidebar-overlay')?.remove();
        }
    }

    // ── Initialize ──
    async function init() {
        // Nav click handlers
        document.querySelectorAll('.nav-item').forEach(el => {
            el.addEventListener('click', e => {
                e.preventDefault();
                navigate(el.dataset.page);
            });
        });

        // Sidebar toggle
        document.getElementById('sidebar-toggle')?.addEventListener('click', toggleSidebar);

        // Sort handlers for runs table
        document.querySelectorAll('.sortable').forEach(th => {
            th.addEventListener('click', () => Pages.sortRuns(th.dataset.sort));
        });

        // Modal close on overlay click
        document.getElementById('run-detail-modal')?.addEventListener('click', e => {
            if (e.target.id === 'run-detail-modal') Pages.closeModal();
        });

        // Escape key to close modal
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') Pages.closeModal();
        });

        // Load initial data
        await Promise.all([
            Pages.loadOverview(),
            Pages.loadFilters(),
            Pages.loadRuns(),
            Pages.loadLiveJobs(),
            Pages.loadSettings(),
        ]);

        // Start polling (every 8s)
        poller = setInterval(async () => {
            await Pages.loadLiveJobs();
            if (State.get('currentPage') === 'overview') {
                await Pages.loadOverview();
            }
        }, 8000);
    }

    // Expose public API
    return {
        navigate,
        init,
        loadRuns: Pages.loadRuns,
        openRunDetail: Pages.openRunDetail,
        closeModal: Pages.closeModal,
        saveSettings: Pages.saveSettings,
        startEvaluation: Pages.startEvaluation,
        markBaseline: Pages.markBaseline,
    };
})();

// Boot
document.addEventListener('DOMContentLoaded', LLMQ.init);
// Fallback if DOMContentLoaded already fired
if (document.readyState !== 'loading') LLMQ.init();
