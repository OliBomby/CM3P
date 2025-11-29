/* ==========================================================================
   MAIN THREAD
   ========================================================================== */

const state = {
    worker: null,
    schema: [],
    indices: [], // subset index -> real index map
    clusters: [], // subset index -> cluster id
    k: 10,
    n_neighbors: 10,
    colorMode: 'cluster',
    viewMode: 'scatter',
    dataFrame: {}, // To store raw columns for coloring if needed (lightweight)
    meta: [],
    fields: [],
    searchMatches: [],
    selectedIndex: null
};

const dom = {
    overlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    fileInput: document.getElementById('parquet-upload'),
    viz: document.getElementById('plotly-div'),
    tooltip: document.getElementById('custom-tooltip'),
    details: document.getElementById('details-panel'),
    neighbors: document.getElementById('neighbors-list'),

    // Controls
    filterInput: document.getElementById('filter-input'),
    filterSuggest: document.getElementById('filter-suggestions'),

    projSelect: document.getElementById('projection-select'),
    colorSelect: document.getElementById('color-select'),

    kSlider: document.getElementById('k-slider'),
    kDisplay: document.getElementById('k-value-display'),

    nSlider: document.getElementById('n-slider'),
    nDisplay: document.getElementById('n-value-display'),

    searchInput: document.getElementById('search-input'),
    btnExport: document.getElementById('btn-export'),
    radios: document.getElementsByName('viewmode')
};

function initWorker() {
    state.worker = new Worker(new URL('./worker_wasm.js', import.meta.url), {type: 'module'});
    state.worker.onmessage = handleMsg;
}

function handleMsg(e) {
    const {type, ...data} = e.data;
    if (type === 'STATUS') {
        dom.loadingText.innerText = data.msg;
    } else if (type === 'ERROR') {
        alert(data.msg);
        dom.overlay.classList.remove('visible');
    } else if (type === 'DATA_READY' || type === 'UPDATE_VIEW') {
        state.indices = data.indices;
        state.clusters = data.clusters;
        if (data.schema) state.schema = data.schema;
        if (data.meta) state.meta = data.meta;
        if (data.fields) state.fields = data.fields;
        state.selectedIndex = null; // reset selection after re-projection/filter
        render(data.coords);
        dom.overlay.classList.remove('visible');
    } else if (type === 'CLUSTERS_UPDATED') {
        state.clusters = data.clusters;
        if (state.colorMode === 'cluster') updateColor();
        dom.overlay.classList.remove('visible');
    } else if (type === 'METADATA') {
        renderMeta(data.data);
    } else if (type === 'NEIGHBORS_FOUND') {
        renderNeighbors(data.results);
        highlightNeighbors(data.results);
    } else if (type === 'CSV_READY') {
        const blob = new Blob([data.blob], {type: 'text/csv'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'filtered_maps.csv';
        a.click();
    } else if (type === 'SEARCH_RESULTS') {
        state.searchMatches = data.indices || [];
        highlightSearch(state.searchMatches);
    }
}

// --- Visuals ---

function render(coords) {
    // Coords is Float32Array [x, y, x, y...]
    const x = [], y = [];
    for (let i = 0; i < coords.length; i += 2) {
        x.push(coords[i]);
        y.push(coords[i + 1]);
    }
    state.x = x;
    state.y = y;

    // Initial Trace with customdata storing subset indices
    const baseCustom = Array.from({length: x.length}, (_, i) => i);
    const trace = {
        x: x, y: y,
        type: state.viewMode === 'scatter' ? 'scattergl' : 'histogram2dcontour',
        mode: 'markers',
        marker: {size: 3, opacity: 0.6},
        hoverinfo: 'none',
        customdata: state.viewMode === 'scatter' ? baseCustom : undefined
    };

    if (state.viewMode === 'density') {
        trace.colorscale = 'Viridis';
        trace.ncontours = 20;
    }

    const layout = {
        margin: {t: 0, l: 0, r: 0, b: 0},
        paper_bgcolor: '#000',
        plot_bgcolor: '#000',
        showlegend: false,
        dragmode: 'pan',
        xaxis: {visible: false},
        yaxis: {visible: false}
    };
    const config = {responsive: true, scrollZoom: true, displayModeBar: false};

    Plotly.newPlot(dom.viz, [trace], layout, config).then(updateColor);

    dom.viz.on('plotly_click', d => {
        if (state.viewMode !== 'scatter') return;
        const pt = d.points[0];
        // subset index from customdata if available (works for overlay traces)
        const subsetIdx = (pt.customdata !== undefined) ? pt.customdata : pt.pointIndex;
        state.selectedIndex = subsetIdx; // track selection for neighbor refresh
        state.worker.postMessage({type: 'GET_METADATA', payload: {index: subsetIdx}});
        state.worker.postMessage({type: 'FIND_NEIGHBORS', payload: {index: subsetIdx, n: state.n_neighbors}});
        addHighlight(subsetIdx, pt.x, pt.y);
    });

    dom.viz.on('plotly_hover', d => {
        if (state.viewMode !== 'scatter') return;
        const pt = d.points[0];
        const subsetIdx = (pt.customdata !== undefined) ? pt.customdata : pt.pointIndex;
        const ev = d.event || (pt && pt.event) || null;
        const rect = dom.viz.getBoundingClientRect();
        let left, top;
        if (!ev) {
            left = rect.width / 2;
            top = rect.height / 2 + 20;
        } else {
            left = ev.clientX - rect.left;
            top = ev.clientY - rect.top + 20;
        }
        dom.tooltip.style.left = left + 'px';
        dom.tooltip.style.top = top + 'px';

        const m = state.meta[subsetIdx] || {};
        const colorLine = formatColorLabel(subsetIdx);
        dom.tooltip.innerHTML = `${m.Artist || 'Unknown'} - ${m.Title || 'Unknown'}<br>` +
            `Diff: ${m.Version || 'Unknown'} by ${m.Creator || 'Unknown'}<br>` +
            (colorLine ? colorLine + '<br>' : '') +
            `ID: ${m.Id ?? state.indices[subsetIdx]}`;
        dom.tooltip.style.display = 'block';
    });
    dom.viz.on('plotly_unhover', () => dom.tooltip.style.display = 'none');
}

function updateColor() {
    if (state.viewMode === 'density') return;

    const raw = dom.colorSelect.value || '';
    const modeNorm = String(raw).toLowerCase().replace(/\s+/g, '_');
    let colorscale = 'Viridis';

    if (modeNorm === 'cluster') {
        Plotly.restyle(dom.viz, {'marker.color': [state.clusters], 'marker.colorscale': colorscale}, [0]);
        return;
    }
    if (modeNorm === 'none') {
        Plotly.restyle(dom.viz, {'marker.color': ['#3498db']}, [0]);
        return;
    }

    // Synonyms map (extended with difficulty)
    const fieldMap = {
        artist: 'Artist',
        creator: 'Creator',
        gamemode: 'GameMode',
        mode: 'GameMode',
        ranked_status: 'RankedStatus',
        rankedstatus: 'RankedStatus',
        approved: 'RankedStatus',
        difficulty: 'DifficultyRating'
    };
    const fieldName = fieldMap[modeNorm] || raw; // fall back to raw select value

    // Special numeric handling for difficulty rating
    if (modeNorm === 'difficulty') {
        if (!state.fields.length || !("DifficultyRating" in (state.fields[0] || {}))) {
            console.warn('DifficultyRating not available in fields');
            Plotly.restyle(dom.viz, {'marker.color': [state.clusters], 'marker.colorscale': colorscale}, [0]);
            return;
        }
        const nums = state.fields.map(f => (typeof f.DifficultyRating === 'number' ? f.DifficultyRating : NaN));

        // Absolute thresholds mapping
        function colorFor(v) {
            if (Number.isNaN(v)) return '#555';
            if (v < 2.0) return '#4fc0ff';
            if (v < 2.7) return '#7cff4f'; // 2.0 - 2.69
            if (v < 4.0) return '#f6f05c'; // 2.7 - 3.99
            if (v < 5.3) return '#ff4e6f'; // 4.0 - 5.29
            if (v < 6.5) return '#c645b8'; // 5.3 - 6.49
            if (v <= 8.0) return '#6563de'; // 6.5 - 8.0
            return '#ccccff'; // > 8
        }

        const colors = nums.map(colorFor);
        Plotly.restyle(dom.viz, {
            'marker.color': [colors]
        }, [0]);
        return;
    }

    // Build categorical values from fields or meta
    let values = [];
    if (Array.isArray(state.fields) && state.fields.length && fieldName in (state.fields[0] || {})) {
        values = state.fields.map(f => String(f[fieldName] ?? 'Unknown'));
    } else if (Array.isArray(state.meta) && state.meta.length && fieldName in (state.meta[0] || {})) {
        values = state.meta.map(m => String(m[fieldName] ?? 'Unknown'));
    } else {
        console.warn('Color mode unavailable or fields missing:', raw);
        Plotly.restyle(dom.viz, {'marker.color': [state.clusters], 'marker.colorscale': colorscale}, [0]);
        return;
    }

    const categories = [];
    const catIndex = new Map();
    values.forEach(v => {
        if (!catIndex.has(v)) {
            catIndex.set(v, categories.length);
            categories.push(v);
        }
    });

    const palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363'
    ];
    const colors = values.map(v => palette[catIndex.get(v) % palette.length]);

    Plotly.restyle(dom.viz, {'marker.color': [colors]}, [0]);
}

function addHighlight(subsetIdx, x, y) {
    // Trace 1 is selection
    const trace = {
        x: [x], y: [y],
        type: 'scattergl', mode: 'markers',
        marker: {color: 'white', size: 12, symbol: 'star'},
        hoverinfo: 'skip',
        customdata: [subsetIdx]
    };
    while (dom.viz.data.length > 1) Plotly.deleteTraces(dom.viz, 1);
    Plotly.addTraces(dom.viz, trace);
}

function highlightNeighbors(results) {
    // Remove previous neighbor trace(s) while preserving selection and search traces
    if (dom.viz && Array.isArray(dom.viz.data)) {
        const removeIndices = [];
        for (let i = 1; i < dom.viz.data.length; i++) { // skip base scatter at 0
            const tr = dom.viz.data[i];
            if (tr && tr.meta === 'neighbors') removeIndices.push(i);
        }
        // Delete from highest index to lowest to avoid reindex issues
        removeIndices.sort((a, b) => b - a).forEach(idx => Plotly.deleteTraces(dom.viz, idx));
    }
    const x = results.map(r => state.x[r.subsetIndex]);
    const y = results.map(r => state.y[r.subsetIndex]);
    const cds = results.map(r => r.subsetIndex);
    if (!x.length) return; // no neighbors -> nothing to draw
    const trace = {
        x: x, y: y,
        type: 'scattergl', mode: 'markers',
        marker: {color: 'red', size: 8, symbol: 'circle-open', line: {width: 2, color: 'red'}},
        hoverinfo: 'skip',
        customdata: cds,
        meta: 'neighbors'
    };
    // Insert neighbor trace just before a search trace if present (search trace assumed last)
    if (dom.viz.data.length > 2) {
        const lastIdx = dom.viz.data.length - 1;
        const lastTrace = dom.viz.data[lastIdx];
        if (lastTrace && lastTrace.meta === 'search') {
            Plotly.addTraces(dom.viz, trace, [lastIdx]);
            return;
        }
    }
    Plotly.addTraces(dom.viz, trace);
}

function renderMeta(data) {
    let html = `<h4>${data.Title || 'Unknown'}</h4>`;
    html += `<div class="detail-row"><span>Artist</span><span>${data.Artist}</span></div>`;
    html += `<div class="detail-row"><span>Creator</span><span>${data.Creator}</span></div>`;
    html += `<div class="detail-row"><span>Version</span><span>${data.Version}</span></div>`;

    const beatmapId = data.Id; // prefer explicit BeatmapId
    if (beatmapId) {
        html += `<a href="https://osu.ppy.sh/b/${beatmapId}" target="_blank" class="btn-small">Open in osu!</a>`;
    }

    html += `<hr>`;
    for (let k in data) {
        if (!['embedding', 'Title', 'Artist', 'Creator', 'Version'].includes(k)) {
            html += `<div style="font-size:0.8em; color:#666"><b>${k}:</b> ${data[k]}</div>`;
        }
    }
    dom.details.innerHTML = html;
}

function renderNeighbors(list) {
    // Sort ascending by cosine distance (lower = closer)
    const ordered = [...list].sort((a, b) => a.dist - b.dist);
    dom.neighbors.innerHTML = ordered.map((n, i) => {
        const m = state.meta[n.subsetIndex] || {};
        const name = `${m.Artist || 'Unknown'} - ${m.Title || 'Unknown'} [${m.Version || 'Unknown'}]`;
        return `<li onclick="selectNeighbor(${i})">${name} <span style=\"float:right; opacity:0.5\">d=${n.dist.toFixed(3)}</span></li>`;
    }).join('');

    window.selectNeighbor = (i) => {
        const n = ordered[i];
        state.selectedIndex = n.subsetIndex; // update selection
        addHighlight(n.subsetIndex, state.x[n.subsetIndex], state.y[n.subsetIndex]);
        state.worker.postMessage({type: 'GET_METADATA', payload: {index: n.subsetIndex}});
    };
}

function highlightSearch(indices) {
    // Remove existing search highlight
    if (dom.viz && Array.isArray(dom.viz.data)) {
        const removeIndices = [];
        for (let i = 1; i < dom.viz.data.length; i++) { // skip base scatter
            const tr = dom.viz.data[i];
            if (tr && tr.meta === 'search') removeIndices.push(i);
        }
        removeIndices.sort((a, b) => b - a).forEach(idx => Plotly.deleteTraces(dom.viz, idx));
    }
    const x = indices.map(si => state.x[si]);
    const y = indices.map(si => state.y[si]);
    const cds = indices.slice();
    if (!x.length) return;
    const trace = {
        x, y,
        type: 'scattergl', mode: 'markers',
        marker: {color: 'white', size: 6, symbol: 'circle-open', line: {width: 2, color: 'white'}},
        hoverinfo: 'skip',
        customdata: cds,
        meta: 'search'
    };
    Plotly.addTraces(dom.viz, trace);
}

// --- Init ---
initWorker();

dom.fileInput.addEventListener('change', async e => {
    const f = e.target.files[0];
    if (!f) return;
    dom.overlay.classList.add('visible');
    const buf = await f.arrayBuffer();
    state.worker.postMessage({type: 'LOAD_FILE', payload: buf});
});

// K-means slider: update value live, and on release/change trigger recluster only if coloring by K-means
function triggerReclusterIfClusterColor() {
    if (String(dom.colorSelect.value).toLowerCase() === 'cluster') {
        dom.overlay.classList.add('visible');
        state.worker.postMessage({type: 'RECLUSTER', payload: {k: state.k}});
    }
}

dom.kSlider.addEventListener('input', e => {
    state.k = parseInt(e.target.value);
    dom.kDisplay.innerText = state.k;
});
// Use change event to approximate slider release on most browsers
dom.kSlider.addEventListener('change', triggerReclusterIfClusterColor);

// Neighbor count slider: update and refresh neighbors for current selection when released
function triggerNeighborRefreshIfSelected() {
    if (state.selectedIndex !== null && state.selectedIndex !== undefined) {
        state.worker.postMessage({type: 'FIND_NEIGHBORS', payload: {index: state.selectedIndex, n: state.n_neighbors}});
    }
}

dom.nSlider.addEventListener('input', e => {
    state.n_neighbors = parseInt(e.target.value);
    dom.nDisplay.innerText = state.n_neighbors;
});
dom.nSlider.addEventListener('change', triggerNeighborRefreshIfSelected);

dom.colorSelect.addEventListener('change', () => {
    state.colorMode = dom.colorSelect.value;
    updateColor();
});

// Projection method change should auto update the view
if (dom.projSelect) {
    dom.projSelect.addEventListener('change', () => {
        triggerFilter();
    });
}

dom.radios.forEach(r => r.addEventListener('change', e => {
    state.viewMode = e.target.value;
    render(state.x.reduce((acc, curr, i) => {
        acc.push(curr, state.y[i]);
        return acc;
    }, [])); // Reconstruct interleaved
}));

dom.btnExport.addEventListener('click', () => {
    state.worker.postMessage({type: 'EXPORT_CSV'});
});

// Wire search input
// Trigger search on Enter; on blur only if changed
let lastSearchValue = '';

function triggerSearch() {
    const q = dom.searchInput.value;
    state.worker.postMessage({
        type: 'SEARCH',
        payload: {query: q}
    });
}

dom.searchInput.addEventListener('focus', () => {
    lastSearchValue = dom.searchInput.value;
});
dom.searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
        triggerSearch();
        lastSearchValue = dom.searchInput.value; // prevent immediate blur re-trigger
    }
});
dom.searchInput.addEventListener('blur', () => {
    const current = dom.searchInput.value;
    if (current !== lastSearchValue) {
        triggerSearch();
        lastSearchValue = current;
    }
});

let lastFilterValue = '';

function triggerFilter() {
    dom.overlay.classList.add('visible');
    const query = dom.filterInput.value;
    state.worker.postMessage({
        type: 'FILTER_AND_PROJECT',
        payload: {query: query, k: state.k, method: dom.projSelect.value}
    });
}

dom.filterInput.addEventListener('focus', () => {
    lastFilterValue = dom.filterInput.value;
});
dom.filterInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
        triggerFilter();
        lastFilterValue = dom.filterInput.value; // prevent immediate blur re-trigger
    }
});
dom.filterInput.addEventListener('blur', () => {
    const current = dom.filterInput.value;
    if (current !== lastFilterValue) {
        triggerFilter();
        lastFilterValue = current;
    }
});

function formatColorLabel(pointIndex) {
    const raw = dom.colorSelect.value || '';
    const modeNorm = String(raw).toLowerCase().replace(/\s+/g, '_');
    const f = state.fields[pointIndex] || {};
    const m = state.meta[pointIndex] || {};
    // Cluster and none: skip
    if (modeNorm === 'cluster' || modeNorm === 'none') return '';

    if (modeNorm === 'difficulty') {
        const v = typeof f.DifficultyRating === 'number' ? f.DifficultyRating : NaN;
        const bucket = Number.isNaN(v) ? 'Unknown' : (
            v < 2.0 ? 'Easy (<2.0)' :
                v < 2.7 ? 'Normal (2.0–2.69)' :
                    v < 4.0 ? 'Hard (2.7–3.99)' :
                        v < 5.3 ? 'Insane (4.0–5.29)' :
                            v < 6.5 ? 'Expert (5.3–6.49)' :
                                v <= 8.0 ? 'Expert+ (6.5–8.0)' :
                                    'Expert++ (>8.0)'
        );
        return `Difficulty: ${Number.isNaN(v) ? 'Unknown' : v.toFixed(2)} (${bucket})`;
    }

    // Map common fields
    const fieldMap = {
        artist: 'Artist',
        creator: 'Creator',
        gamemode: 'GameMode',
        mode: 'GameMode',
        ranked_status: 'RankedStatus',
        rankedstatus: 'RankedStatus',
        approved: 'RankedStatus'
    };
    const key = fieldMap[modeNorm] || raw;

    let val = undefined;
    if (key in f) val = f[key];
    else if (key in m) val = m[key];

    if (val === undefined || val === null || val === '') return '';
    return `${raw}: ${val}`;
}
