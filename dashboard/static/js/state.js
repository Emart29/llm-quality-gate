/**
 * LLMQ Dashboard — Application State
 * Reactive-ish state management for the dashboard.
 */
const State = (() => {
    const _state = {
        currentPage: 'overview',
        overview: null,
        runs: [],
        runsSortField: 'date',
        runsSortAsc: false,
        filters: {},
        compare: [],
        qualityGates: [],
        settings: {},
        providers: [],
        activeJobs: [],
        lastRun: null,
    };

    const _listeners = {};

    function get(key) { return _state[key]; }

    function set(key, value) {
        _state[key] = value;
        (_listeners[key] || []).forEach(fn => fn(value));
    }

    function on(key, fn) {
        if (!_listeners[key]) _listeners[key] = [];
        _listeners[key].push(fn);
    }

    return { get, set, on };
})();
