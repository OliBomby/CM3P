import {parquetRead} from 'https://cdn.jsdelivr.net/npm/hyparquet@1.22.1/+esm';
import {compressors} from 'https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.1/+esm';
import {UMAP} from 'https://cdn.jsdelivr.net/npm/umap-js@1.3.3/+esm';
import {WorkerPool} from './worker_pool.js';

// ============================================================================
// WASM Module Loading + Parallel Worker Pool
// ============================================================================
let wasmModule = null;
let useWasm = false;
let workerPool = null;
let useParallel = true;

async function initWasm() {
    try {
        const init = await import('./wasm/pkg/embeddings_wasm.js');
        await init.default();
        wasmModule = await import('./wasm/pkg/embeddings_wasm.js');
        useWasm = true;
        console.log('WASM module loaded successfully');
        self.postMessage({type: 'WASM_STATUS', loaded: true});
    } catch (err) {
        console.warn('WASM module not available, using JavaScript fallback:', err);
        useWasm = false;
        self.postMessage({type: 'WASM_STATUS', loaded: false});
    }
}

async function initWorkerPool() {
    try {
        workerPool = new WorkerPool();
        await workerPool.ready();
        console.log('Worker pool initialized for parallel processing');
        self.postMessage({type: 'PARALLEL_STATUS', loaded: true});
    } catch (err) {
        console.warn('Worker pool not available:', err);
        useParallel = false;
        self.postMessage({type: 'PARALLEL_STATUS', loaded: false});
    }
}

// Initialize both on worker startup
Promise.all([initWasm(), initWorkerPool()]).then(() => {
    console.log('All optimizations loaded');
});

// ============================================================================
// JavaScript Fallback Implementations
// ============================================================================

function calculatePCA_JS(embeddings) {
    const n = embeddings.length;
    if (n === 0) return [];
    const dims = embeddings[0].length;

    const mean = new Float32Array(dims);
    for (let i = 0; i < n; i++) {
        const v = embeddings[i];
        for (let j = 0; j < dims; j++) mean[j] += v[j];
    }
    for (let j = 0; j < dims; j++) mean[j] /= n;

    const components = [];
    const getCentered = (i, j) => embeddings[i][j] - mean[j];

    for (let c = 0; c < 2; c++) {
        let ev = new Float32Array(dims).map(() => Math.random() - 0.5);
        let mag = Math.sqrt(ev.reduce((a, b) => a + b * b, 0));
        ev = ev.map(x => x / mag);

        for (let iter = 0; iter < 8; iter++) {
            const scores = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                let s = 0;
                for (let j = 0; j < dims; j++) s += getCentered(i, j) * ev[j];
                scores[i] = s;
            }
            const next_ev = new Float32Array(dims);
            for (let j = 0; j < dims; j++) {
                let s = 0;
                for (let i = 0; i < n; i++) s += scores[i] * getCentered(i, j);
                next_ev[j] = s;
            }
            mag = Math.sqrt(next_ev.reduce((a, b) => a + b * b, 0));
            if (mag > 0) ev = next_ev.map(x => x / mag);
        }
        components.push(ev);

        if (c === 1) {
            const u = components[0];
            const v = ev;
            let dot = 0;
            for (let k = 0; k < dims; k++) dot += u[k] * v[k];
            for (let k = 0; k < dims; k++) v[k] -= dot * u[k];
            const mag2 = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
            ev = v.map(x => x / mag2);
            components[1] = ev;
        }
    }

    const projected = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
        let x = 0, y = 0;
        for (let j = 0; j < dims; j++) {
            const val = getCentered(i, j);
            x += val * components[0][j];
            y += val * components[1][j];
        }
        projected[i * 2] = x;
        projected[i * 2 + 1] = y;
    }
    return projected;
}

function calculateKMeans_JS(embeddings, k) {
    const n = embeddings.length;
    if (n === 0) return [];
    const dims = embeddings[0].length;

    let centroids = [];
    for (let i = 0; i < k; i++) centroids.push(embeddings[Math.floor(Math.random() * n)]);

    let labels = new Int8Array(n);
    for (let iter = 0; iter < 5; iter++) {
        const sums = Array(k).fill(0).map(() => new Float32Array(dims));
        const counts = new Int32Array(k);

        for (let i = 0; i < n; i++) {
            let minD = Infinity, best = 0;
            for (let c = 0; c < k; c++) {
                let d = 0;
                for (let j = 0; j < dims; j++) {
                    let diff = embeddings[i][j] - centroids[c][j];
                    d += diff * diff;
                }
                if (d < minD) {
                    minD = d;
                    best = c;
                }
            }
            labels[i] = best;
            counts[best]++;
            for (let j = 0; j < dims; j++) sums[best][j] += embeddings[i][j];
        }
        for (let c = 0; c < k; c++) {
            if (counts[c]) {
                for (let j = 0; j < dims; j++) centroids[c][j] = sums[c][j] / counts[c];
            }
        }
    }
    return labels;
}

function normalize_JS(v) {
    let sumSq = 0;
    for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
    if (sumSq === 0) return new Float32Array(v.length);
    const inv = 1 / Math.sqrt(sumSq);
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
}

// ============================================================================
// Unified API with WASM acceleration
// ============================================================================

function flattenEmbeddings(embeddings) {
    if (!embeddings || embeddings.length === 0) return {flat: new Float32Array(0), n: 0, dims: 0};
    const n = embeddings.length;
    const dims = embeddings[0].length;
    const flat = new Float32Array(n * dims);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < dims; j++) {
            flat[i * dims + j] = embeddings[i][j];
        }
    }
    return {flat, n, dims};
}

function calculatePCA(embeddings) {
    const t0 = performance.now();
    let result;

    if (useWasm && wasmModule) {
        const {flat, n, dims} = flattenEmbeddings(embeddings);
        result = wasmModule.pca_from_js(flat, n, dims);
    } else {
        result = calculatePCA_JS(embeddings);
    }

    const t1 = performance.now();
    console.log(`PCA (${useWasm ? 'WASM' : 'JS'}): ${(t1 - t0).toFixed(2)}ms`);
    return result;
}

async function calculateKMeans(embeddings, k) {
    const t0 = performance.now();
    let result;

    // Use parallel processing if worker pool is available
    if (useParallel && workerPool) {
        const {flat, n, dims} = flattenEmbeddings(embeddings);
        result = await workerPool.kmeans(flat, n, dims, k, 5);
        const t1 = performance.now();
        console.log(`K-Means (PARALLEL): ${(t1 - t0).toFixed(2)}ms`);
        return result;
    }

    if (useWasm && wasmModule) {
        const {flat, n, dims} = flattenEmbeddings(embeddings);
        result = wasmModule.kmeans_from_js(flat, n, dims, k);
    } else {
        result = calculateKMeans_JS(embeddings, k);
    }

    const t1 = performance.now();
    console.log(`K-Means (${useWasm ? 'WASM' : 'JS'}): ${(t1 - t0).toFixed(2)}ms`);
    return result;
}

function normalizeVectors(embeddings) {
    const t0 = performance.now();
    let result;

    if (useWasm && wasmModule) {
        const {flat, n, dims} = flattenEmbeddings(embeddings);
        const normalized = wasmModule.normalize_from_js(flat, n, dims);
        // Convert back to array of arrays
        result = [];
        for (let i = 0; i < n; i++) {
            const vec = new Float32Array(dims);
            for (let j = 0; j < dims; j++) {
                vec[j] = normalized[i * dims + j];
            }
            result.push(vec);
        }
    } else {
        result = embeddings.map(normalize_JS);
    }

    const t1 = performance.now();
    console.log(`Normalize (${useWasm ? 'WASM' : 'JS'}): ${(t1 - t0).toFixed(2)}ms`);
    return result;
}

function findNearestNeighbors(normalizedEmbeddings, queryIdx, k) {
    const t0 = performance.now();
    let results;

    if (useWasm && wasmModule) {
        const {flat, n, dims} = flattenEmbeddings(normalizedEmbeddings);
        const neighbors = wasmModule.neighbors_from_js(flat, n, dims, queryIdx, k);
        results = [];
        for (let i = 0; i < neighbors.indices.length; i++) {
            results.push({
                subsetIndex: neighbors.indices[i],
                dist: neighbors.distances[i],
                sim: 1 - neighbors.distances[i]
            });
        }
    } else {
        // JavaScript fallback
        const targetNorm = normalizedEmbeddings[queryIdx];
        if (!targetNorm) return [];

        results = [];
        for (let i = 0; i < normalizedEmbeddings.length; i++) {
            if (i === queryIdx) continue;
            const vec = normalizedEmbeddings[i];
            let dot = 0;
            for (let j = 0; j < vec.length; j++) dot += targetNorm[j] * vec[j];
            const sim = dot;
            const dist = 1 - sim;
            results.push({subsetIndex: i, sim, dist});
        }
        results.sort((a, b) => a.dist - b.dist);
        results = results.slice(0, k);
    }

    const t1 = performance.now();
    console.log(`Find Neighbors (${useWasm ? 'WASM' : 'JS'}): ${(t1 - t0).toFixed(2)}ms`);
    return results;
}

function calculateUMAP(embeddings) {
    // UMAP uses external library, keep as-is
    const umap = new UMAP({
        nNeighbors: 15,
        minDist: 0.1,
        nComponents: 2,
        nEpochs: 50
    });
    const embedding = umap.fit(embeddings);

    const projected = new Float32Array(embeddings.length * 2);
    for (let i = 0; i < embeddings.length; i++) {
        projected[i * 2] = embedding[i][0];
        projected[i * 2 + 1] = embedding[i][1];
    }
    return projected;
}

// ============================================================================
// Data Store
// ============================================================================
let fullData = [];
let filteredIndices = [];
let currentEmbeddings = [];
let currentEmbeddingsNorm = [];
let currentCoords = null;
let currentMethod = 'PCA';

function pickMeta(row) {
    return {
        Artist: row.Artist ?? 'Unknown',
        Title: row.Title ?? 'Unknown',
        Version: row.Version ?? 'Unknown',
        Creator: row.Creator ?? 'Unknown',
        Id: row.Id ?? -1
    };
}

function pickFields(row) {
    return {
        Artist: row.Artist ?? 'Unknown',
        Creator: row.Creator ?? 'Unknown',
        GameMode: row.GameMode ?? row.Mode ?? 'Unknown',
        RankedStatus: row.RankedStatus ?? row.Status ?? 'Unknown',
        DifficultyRating: row.DifficultyRating ?? 0,
        SubmittedDate: row.SubmittedDate ?? null,
    };
}

// ============================================================================
// Query Parsing & Matching (unchanged)
// ============================================================================
const OPS = new Set(['=', '!=', '<', '>', '<=', '>=']);
const FLOAT_TOL = 0.01;

function splitTokens(input) {
    const s = String(input || '').trim();
    const tokens = [];
    let buf = '';
    let inQ = false;
    let qChar = '';
    for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        if (inQ) {
            if (ch === qChar) {
                inQ = false;
                buf += ch;
                continue;
            }
            if (ch === '\\' && s[i + 1] === qChar) {
                buf += qChar;
                i++;
                continue;
            }
            buf += ch;
            continue;
        } else {
            if (ch === '"' || ch === '\'') {
                inQ = true;
                qChar = ch;
                buf += ch;
                continue;
            }
            if (ch === ' ') {
                if (buf.trim().length) {
                    tokens.push(buf.trim());
                }
                buf = '';
                continue;
            }
            buf += ch;
        }
    }
    if (buf.trim().length) tokens.push(buf.trim());
    return tokens;
}

function parseToken(token) {
    let inQ = false;
    let qChar = '';
    let opPos = -1;
    let opFound = '';
    for (let i = 0; i < token.length; i++) {
        const ch = token[i];
        if (inQ) {
            if (ch === qChar) inQ = false;
            if (ch === '\\' && token[i + 1] === qChar) {
                i++;
                continue;
            }
            continue;
        } else {
            if (ch === '"' || ch === '\'') {
                inQ = true;
                qChar = ch;
                continue;
            }
            for (const op of ['!=', '<=', '>=']) {
                if (token.startsWith(op, i)) {
                    opPos = i;
                    opFound = op;
                    i += op.length - 1;
                    break;
                }
            }
            if (opPos !== -1) break;
            if (OPS.has(ch)) {
                opPos = i;
                opFound = ch;
                break;
            }
        }
    }
    if (opPos === -1) return {type: 'value', value: unquote(token)};
    const colPart = token.slice(0, opPos).trim();
    const valPart = token.slice(opPos + opFound.length).trim();
    if (!colPart) return {type: 'value', value: unquote(valPart)};
    const column = colPart;
    const valueRaw = unquote(valPart);
    return {type: 'tuple', col: column, op: opFound, value: valueRaw};
}

function unquote(v) {
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith('\'') && v.endsWith('\''))) {
        return v.slice(1, -1);
    }
    return v;
}

function buildSubqueries(query) {
    const rawTokens = splitTokens(query);
    const merged = [];
    for (let i = 0; i < rawTokens.length; i++) {
        let tok = rawTokens[i];
        const lower = tok.toLowerCase();
        if (lower === 'and' || lower === 'or') continue;
        if (i + 2 < rawTokens.length) {
            const opTok = rawTokens[i + 1];
            const valTok = rawTokens[i + 2];
            if (OPS.has(opTok)) {
                merged.push(`${tok}${opTok}${valTok}`);
                i += 2;
                continue;
            }
        }
        merged.push(tok);
    }
    return merged.map(parseToken).filter(t => {
        if (t.type === 'tuple') return OPS.has(t.op);
        return t.value.length > 0;
    });
}

function findColumnCaseInsensitive(row, col) {
    const target = col.toLowerCase();
    for (const key of Object.keys(row)) {
        if (key.toLowerCase() === target) return key;
    }
    return null;
}

function numericCompare(lhs, rhs, op) {
    const ln = Number(lhs);
    const rn = Number(rhs);
    if (Number.isNaN(ln) || Number.isNaN(rn)) return null;
    switch (op) {
        case '=':
            return Math.abs(ln - rn) <= FLOAT_TOL;
        case '!=':
            return Math.abs(ln - rn) > FLOAT_TOL;
        case '<':
            return ln < rn;
        case '>':
            return ln > rn;
        case '<=':
            return ln <= rn + FLOAT_TOL;
        case '>=':
            return ln + FLOAT_TOL >= rn;
    }
    return false;
}

function stringCompare(lhs, rhs, op) {
    const ls = String(lhs ?? '').toLowerCase();
    const rs = String(rhs ?? '').toLowerCase();
    switch (op) {
        case '=':
            return ls.includes(rs);
        case '!=':
            return !ls.includes(rs);
        case '<':
            return ls < rs;
        case '>':
            return ls > rs;
        case '<=':
            return ls <= rs;
        case '>=':
            return ls >= rs;
    }
    return false;
}

function tupleMatches(row, sq) {
    const key = findColumnCaseInsensitive(row, sq.col);
    const val = key ? row[key] : undefined;
    const numericResult = numericCompare(val, sq.value, sq.op);
    if (numericResult !== null && (sq.op !== '=' && sq.op !== '!=' || typeof val === 'number')) {
        return numericResult;
    }
    return stringCompare(val, sq.value, sq.op);
}

function rowMatchesQuery(row, subqueries) {
    if (!subqueries || subqueries.length === 0) return true;
    for (const sq of subqueries) {
        if (sq.type === 'value') {
            const needle = String(sq.value).toLowerCase();
            let found = false;
            for (const key of Object.keys(row)) {
                if (key === 'embedding') continue;
                const txt = String(row[key] ?? '').toLowerCase();
                if (needle && txt.includes(needle)) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        } else if (sq.type === 'tuple') {
            if (!tupleMatches(row, sq)) return false;
        }
    }
    return true;
}

// ============================================================================
// Message Handlers
// ============================================================================

self.onmessage = async (e) => {
    const {type, payload} = e.data;

    if (type === 'LOAD_FILE') {
        try {
            console.log("Worker: Loading file...");
            self.postMessage({type: 'STATUS', msg: 'Parsing Parquet...'});

            await parquetRead({
                file: payload,
                rowFormat: 'object',
                compressors: compressors,
                onComplete: async (rows) => {
                    const totalRows = rows.length;
                    const colNames = Object.keys(rows[0] ?? {});

                    const validRows = [];
                    for (let i = 0; i < totalRows; i++) {
                        const r = rows[i];
                        const emb = r.embedding;
                        if (!emb || typeof emb.length !== 'number' || emb.length === 0) continue;
                        const idx = validRows.length;
                        validRows.push({originalIndex: idx, ...r});
                    }

                    const rowCount = validRows.length;
                    if (rowCount === 0) {
                        throw new Error("No rows with a valid 'embedding' column found");
                    }

                    console.log(`Worker: Loaded ${rowCount} rows (from ${totalRows}) and ${colNames.length} columns.`);
                    self.postMessage({type: 'STATUS', msg: `Processing ${rowCount} rows...`});

                    fullData = validRows;
                    const embeddings = fullData.map(r => r.embedding);

                    self.postMessage({type: 'STATUS', msg: 'Running Initial PCA...'});
                    currentCoords = calculatePCA(embeddings);

                    self.postMessage({type: 'STATUS', msg: 'Running K-Means clustering...'});
                    const clusters = await calculateKMeans(embeddings, 10);

                    filteredIndices = fullData.map((_, i) => i);
                    currentEmbeddings = embeddings;

                    self.postMessage({type: 'STATUS', msg: 'Normalizing vectors...'});
                    currentEmbeddingsNorm = normalizeVectors(embeddings);

                    const meta = filteredIndices.map(i => pickMeta(fullData[i]));
                    const fields = filteredIndices.map(i => pickFields(fullData[i]));

                    self.postMessage({
                        type: 'DATA_READY',
                        coords: currentCoords,
                        clusters: clusters,
                        indices: filteredIndices,
                        schema: colNames,
                        meta: meta,
                        fields: fields
                    });
                }
            });
        } catch (err) {
            console.error("Worker Error:", err);
            self.postMessage({type: 'ERROR', msg: err.message});
        }
    }

    if (type === 'FILTER_AND_PROJECT') {
        const {query, k, method} = payload;
        console.log(`Worker: Filtering with query "${query}" and projecting with ${method}...`);

        try {
            const subs = buildSubqueries(query);
            const newIndices = [];
            const newEmbeddings = [];

            for (let i = 0; i < fullData.length; i++) {
                try {
                    if (rowMatchesQuery(fullData[i], subs)) {
                        newIndices.push(i);
                        newEmbeddings.push(fullData[i].embedding);
                    }
                } catch (e) {
                }
            }

            self.postMessage({type: 'STATUS', msg: `Projecting ${newIndices.length} points (${method})...`});

            let coords;
            if (method === 'UMAP' || method === 'TSNE') {
                if (newIndices.length > 500000) {
                    self.postMessage({type: 'STATUS', msg: 'Dataset too large for UMAP (Limit 5k). Using PCA.'});
                    coords = calculatePCA(newEmbeddings);
                } else {
                    coords = calculateUMAP(newEmbeddings);
                }
            } else {
                coords = calculatePCA(newEmbeddings);
            }

            self.postMessage({type: 'STATUS', msg: 'Running K-Means clustering...'});
            const clusters = await calculateKMeans(newEmbeddings, k);

            filteredIndices = newIndices;
            currentEmbeddings = newEmbeddings;
            currentEmbeddingsNorm = normalizeVectors(newEmbeddings);
            currentCoords = coords;
            currentMethod = method;

            const meta = filteredIndices.map(i => pickMeta(fullData[i]));
            const fields = filteredIndices.map(i => pickFields(fullData[i]));

            self.postMessage({
                type: 'UPDATE_VIEW',
                coords: coords,
                clusters: clusters,
                indices: filteredIndices,
                meta: meta,
                fields: fields
            });

        } catch (err) {
            self.postMessage({type: 'ERROR', msg: "Filter/Proj Error: " + err.message});
        }
    }

    if (type === 'RECLUSTER') {
        const clusters = await calculateKMeans(currentEmbeddings, payload.k);
        self.postMessage({type: 'CLUSTERS_UPDATED', clusters: clusters});
    }

    if (type === 'GET_METADATA') {
        const fullIndex = filteredIndices[payload.index];
        const row = fullData[fullIndex];
        if (!row) {
            self.postMessage({type: 'METADATA', data: {}});
            return;
        }
        const {embedding, ...meta} = row;
        self.postMessage({type: 'METADATA', data: meta});
    }

    if (type === 'FIND_NEIGHBORS') {
        const results = findNearestNeighbors(currentEmbeddingsNorm, payload.index, payload.n);
        const enriched = results.map(r => ({
            ...r,
            realIndex: filteredIndices[r.subsetIndex],
            title: fullData[filteredIndices[r.subsetIndex]].Title || 'Unknown'
        }));
        self.postMessage({type: 'NEIGHBORS_FOUND', results: enriched});
    }

    if (type === 'EXPORT_CSV') {
        const headers = Object.keys(fullData[0]).filter(k => k !== 'embedding');
        let csv = headers.join(',') + '\n';

        for (let fullIndex of filteredIndices) {
            const row = fullData[fullIndex];
            const line = headers.map(h => {
                let val = row[h] === null ? '' : row[h];
                val = String(val).replace(/"/g, '""');
                if (val.search(/("|,|\n)/g) >= 0) val = `"${val}"`;
                return val;
            }).join(',');
            csv += line + '\n';
        }

        self.postMessage({type: 'CSV_READY', blob: csv});
    }

    if (type === 'SEARCH') {
        if (!payload.query || payload.query.trim() === '') {
            self.postMessage({type: 'SEARCH_RESULTS', indices: []});
            return;
        }
        console.log(`Worker: Searching with query "${payload.query}"...`);
        const subs = buildSubqueries(payload.query || '');
        const matches = [];
        for (let si = 0; si < filteredIndices.length; si++) {
            const row = fullData[filteredIndices[si]];
            if (rowMatchesQuery(row, subs)) {
                matches.push(si);
            }
        }
        self.postMessage({type: 'SEARCH_RESULTS', indices: matches});
    }
};
