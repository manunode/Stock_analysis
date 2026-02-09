/**
 * StockLens - Client-side JavaScript
 */

// ── Search widget (Alpine.js component) ─────────────────────────────────────
function searchWidget() {
    return {
        query: '',
        results: [],
        async search() {
            if (this.query.length < 1) {
                this.results = [];
                return;
            }
            try {
                const resp = await fetch(`/api/search?q=${encodeURIComponent(this.query)}`);
                this.results = await resp.json();
            } catch (e) {
                this.results = [];
            }
        }
    };
}

// ── Radar Chart builder ─────────────────────────────────────────────────────
function createRadarChart(canvasId, data, labels) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels || ['Quality', 'Valuation', 'Strength', 'Growth', 'Cash Flow', 'Piotroski'],
            datasets: [{
                label: 'Stock Profile',
                data: data,
                fill: true,
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                borderColor: 'rgb(16, 185, 129)',
                pointBackgroundColor: 'rgb(16, 185, 129)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(16, 185, 129)',
                borderWidth: 2,
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { stepSize: 20, font: { size: 10 }, backdropColor: 'transparent' },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                    pointLabels: { font: { size: 11, weight: '600' }, color: '#374151' },
                    angleLines: { color: 'rgba(0,0,0,0.06)' },
                }
            },
            plugins: {
                legend: { display: false },
            }
        }
    });
}

// ── Donut Chart builder ─────────────────────────────────────────────────────
function createDonutChart(canvasId, labels, data, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors || [
                    '#3b82f6', '#8b5cf6', '#10b981', '#9ca3af', '#f59e0b'
                ],
                borderWidth: 2,
                borderColor: '#ffffff',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 12, font: { size: 11 } }
                }
            }
        }
    });
}

// ── Compare radar chart (multi-dataset) ─────────────────────────────────────
function createCompareRadar(canvasId, datasets, labels) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const colors = [
        { bg: 'rgba(16, 185, 129, 0.12)', border: 'rgb(16, 185, 129)' },
        { bg: 'rgba(59, 130, 246, 0.12)', border: 'rgb(59, 130, 246)' },
        { bg: 'rgba(245, 158, 11, 0.12)', border: 'rgb(245, 158, 11)' },
        { bg: 'rgba(139, 92, 246, 0.12)', border: 'rgb(139, 92, 246)' },
        { bg: 'rgba(239, 68, 68, 0.12)', border: 'rgb(239, 68, 68)' },
    ];
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels || ['Quality', 'Valuation', 'Strength', 'Growth', 'Cash Flow', 'Piotroski'],
            datasets: datasets.map((ds, i) => ({
                label: ds.label,
                data: ds.data,
                fill: true,
                backgroundColor: colors[i % colors.length].bg,
                borderColor: colors[i % colors.length].border,
                pointBackgroundColor: colors[i % colors.length].border,
                borderWidth: 2,
                pointRadius: 3,
            }))
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { stepSize: 20, font: { size: 10 }, backdropColor: 'transparent' },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                    pointLabels: { font: { size: 11, weight: '600' } },
                }
            },
            plugins: {
                legend: { position: 'top', labels: { padding: 12, font: { size: 11 } } }
            }
        }
    });
}

// ── Utility: format number ──────────────────────────────────────────────────
function fmt(val, decimals = 1) {
    if (val === null || val === undefined) return '--';
    return Number(val).toFixed(decimals);
}
