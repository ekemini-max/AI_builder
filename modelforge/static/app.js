const API_BASE = 'https://your-modelforge-backend.onrender.com';

let currentSelectedModel = null;
let f1ChartInstance = null;

function getApiUrl(path) {
    if (!API_BASE || window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')) {
        return path;
    }
    return `${API_BASE.replace(/\/$/, '')}${path}`;
}

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

document.addEventListener('DOMContentLoaded', () => {
    loadModelsList();

    // Event Listeners
    document.getElementById('btn-refresh-models').addEventListener('click', loadModelsList);
    document.getElementById('btn-create-model-modal').addEventListener('click', () => {
        document.getElementById('modal-create-model').classList.remove('hidden');
    });
    document.getElementById('close-modal').addEventListener('click', () => {
        document.getElementById('modal-create-model').classList.add('hidden');
    });

    document.getElementById('form-create-model').addEventListener('submit', handleCreateModel);
    document.getElementById('form-autotrain').addEventListener('submit', handleAutoTrain);
    document.getElementById('form-upload-model').addEventListener('submit', handleUploadModel);

    // Copy API Key
    document.getElementById('btn-copy-key').addEventListener('click', () => {
        const keyInput = document.getElementById('detail-api-key');
        navigator.clipboard.writeText(keyInput.value);
        showToast('API Key copied to clipboard!', 'success');
    });

    // Tab Switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            e.target.classList.add('active');
            const targetTab = e.target.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Image preview for test
    document.getElementById('test-image-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                const img = document.getElementById('test-image-preview');
                img.src = evt.target.result;
                img.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    document.getElementById('btn-run-inference').addEventListener('click', handleRunInference);
});


function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}


async function loadModelsList() {
    const container = document.getElementById('models-list-container');
    container.innerHTML = '<p class="placeholder-text">Loading models...</p>';

    try {
        const res = await fetch(getApiUrl('/api/v1/models'));
        const data = await res.json();

        if (data.status === 'ok') {
            if (data.models.length === 0) {
                container.innerHTML = '<p class="placeholder-text">No models created yet. Click "+ Create New Model Slot" to start.</p>';
                return;
            }

            container.innerHTML = '';
            data.models.forEach(model => {
                const card = document.createElement('div');
                card.className = `model-card ${currentSelectedModel && currentSelectedModel.id === model.id ? 'selected' : ''}`;

                let accuracyText = 'Not trained/uploaded';
                let accuracyColor = 'var(--text-muted)';
                if (model.metrics && model.metrics.accuracy !== undefined) {
                    const accPct = (model.metrics.accuracy * 100).toFixed(1);
                    const isHeldOut = model.metrics.is_trustworthy_held_out;
                    accuracyText = `Acc: ${accPct}% (${isHeldOut ? 'Held-out' : 'Sanity'})`;
                    accuracyColor = 'var(--accent-green)';
                }

                const safeName = escapeHtml(model.name);
                const safeDesc = escapeHtml(model.description || 'No description provided');
                const safeTask = escapeHtml(model.task_type);

                card.innerHTML = `
                    <div class="model-card-header">
                        <h3>${safeName}</h3>
                        <span class="badge">${safeTask}</span>
                    </div>
                    <p class="model-desc-preview">${safeDesc}</p>
                    <div style="font-size:12px; font-weight:600; color: ${accuracyColor}; margin-bottom: 6px;">
                        ${accuracyText}
                    </div>
                    <div class="model-card-footer">
                        <span>Versions: ${model.version_count}</span>
                        <span>Threshold: ${model.confidence_threshold}</span>
                    </div>
                `;

                card.addEventListener('click', () => selectModel(model.id));
                container.appendChild(card);
            });
        }
    } catch (err) {
        container.innerHTML = `<p class="placeholder-text" style="color:var(--accent-red)">Failed to load models: ${escapeHtml(err.message)}</p>`;
    }
}


async function selectModel(modelId) {
    try {
        const res = await fetch(getApiUrl(`/api/v1/models/${modelId}`));
        const data = await res.json();

        if (data.status === 'ok') {
            currentSelectedModel = data.model;
            renderModelDetails(data.model);
            loadModelsList();
        }
    } catch (err) {
        showToast('Failed to load model details: ' + err.message, 'error');
    }
}


function renderModelDetails(model) {
    document.getElementById('model-detail-section').classList.remove('hidden');
    document.getElementById('detail-model-name').textContent = model.name;
    document.getElementById('detail-model-desc').textContent = model.description || 'No description provided.';
    document.getElementById('detail-task-type').textContent = (model.task_type || '').toUpperCase();
    document.getElementById('detail-model-id').textContent = `ID: ${model.id}`;
    document.getElementById('detail-api-key').value = model.api_key;

    // Toggle text column options
    const textColBox = document.getElementById('text-col-options');
    if (model.task_type === 'text') {
        textColBox.classList.remove('hidden');
    } else {
        textColBox.classList.add('hidden');
    }

    // Configure Test Tab interface
    const testVisionBox = document.getElementById('test-vision-box');
    const testTextBox = document.getElementById('test-text-box');
    if (model.task_type === 'vision') {
        testVisionBox.classList.remove('hidden');
        testTextBox.classList.add('hidden');
    } else {
        testTextBox.classList.remove('hidden');
        testVisionBox.classList.add('hidden');
    }

    renderVersionHistoryAndVisuals(model.versions, model.active_version_id);
}


function renderVersionHistoryAndVisuals(versions, activeVersionId) {
    const listContainer = document.getElementById('version-history-list');
    const visualPanel = document.getElementById('active-metrics-visual-panel');

    if (!versions || versions.length === 0) {
        listContainer.innerHTML = '<p class="subtext">No versions registered for this model yet.</p>';
        visualPanel.classList.add('hidden');
        return;
    }

    const activeVersion = versions.find(v => v.id === activeVersionId) || versions[versions.length - 1];

    if (activeVersion && activeVersion.evaluation_metrics) {
        visualPanel.classList.remove('hidden');
        const em = activeVersion.evaluation_metrics;

        const trustBadge = document.getElementById('trust-badge-active');
        if (em.is_trustworthy_held_out) {
            trustBadge.className = 'badge badge-free';
            trustBadge.textContent = '✅ Trustworthy: Held-Out Test Evaluation';
        } else {
            trustBadge.className = 'badge badge-warning';
            trustBadge.textContent = '⚠️ Optimistic: Training/Sanity Evaluation';
        }

        renderF1Chart(em.per_class_metrics);
        renderConfusionMatrix(em.classes || Object.keys(em.per_class_metrics || {}), em.confusion_matrix);
    } else {
        visualPanel.classList.add('hidden');
    }

    listContainer.innerHTML = '';
    versions.slice().reverse().forEach(v => {
        const item = document.createElement('div');
        item.className = `version-item ${v.is_active ? 'active-ver' : ''}`;

        const em = v.evaluation_metrics;
        let metricsHtml = '<p class="subtext">No evaluation metrics recorded for this version.</p>';

        if (em) {
            let perClassRows = '';
            if (em.per_class_metrics) {
                Object.entries(em.per_class_metrics).forEach(([cls, m]) => {
                    const safeCls = escapeHtml(cls);
                    perClassRows += `
                        <tr>
                            <td><strong>${safeCls}</strong></td>
                            <td>${(m.precision * 100).toFixed(1)}%</td>
                            <td>${(m.recall * 100).toFixed(1)}%</td>
                            <td>${(m.f1 * 100).toFixed(1)}%</td>
                            <td>${m.support}</td>
                        </tr>
                    `;
                });
            }

            metricsHtml = `
                <div class="metrics-grid">
                    <div class="metric-box"><label>Accuracy</label><span>${(em.accuracy * 100).toFixed(1)}%</span></div>
                    <div class="metric-box"><label>Precision</label><span>${(em.macro_precision * 100).toFixed(1)}%</span></div>
                    <div class="metric-box"><label>Recall</label><span>${(em.macro_recall * 100).toFixed(1)}%</span></div>
                    <div class="metric-box"><label>Macro F1</label><span>${(em.macro_f1 * 100).toFixed(1)}%</span></div>
                </div>
                <div style="font-size:11px; margin-bottom:8px;">
                    <strong>Integrity:</strong>
                    <span class="badge ${em.is_trustworthy_held_out ? 'badge-free' : 'badge-outline'}">
                        ${em.is_trustworthy_held_out ? '✅ Held-out test split' : '⚠️ Training/Sanity check'}
                    </span>
                </div>
                <table class="per-class-table">
                    <thead>
                        <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
                    </thead>
                    <tbody>${perClassRows}</tbody>
                </table>
            `;
        }

        const safeFormat = escapeHtml(v.format);
        const safeSource = escapeHtml(v.source);

        item.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong>Version ${v.version_number}</strong>
                    <span class="badge" style="margin-left:6px;">${safeFormat}</span>
                    <span class="subtext" style="margin-left:6px;">Source: ${safeSource}</span>
                </div>
                <div>
                    ${v.is_active ? '<span class="badge badge-free">ACTIVE</span>' : `<button class="btn btn-secondary btn-small" onclick="handleRollback(${v.version_number})">Rollback to v${v.version_number}</button>`}
                </div>
            </div>
            ${metricsHtml}
        `;

        listContainer.appendChild(item);
    });
}


function renderF1Chart(perClassMetrics) {
    if (!perClassMetrics) return;

    const labels = Object.keys(perClassMetrics);
    const f1Scores = labels.map(cls => (perClassMetrics[cls].f1 * 100).toFixed(1));

    const ctx = document.getElementById('f1-chart-canvas').getContext('2d');

    if (f1ChartInstance) {
        f1ChartInstance.destroy();
    }

    f1ChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'F1 Score (%)',
                data: f1Scores,
                backgroundColor: 'rgba(56, 189, 248, 0.6)',
                borderColor: '#38bdf8',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}


function renderConfusionMatrix(classes, confusionMatrix) {
    const wrap = document.getElementById('confusion-matrix-table-wrap');
    if (!confusionMatrix || !classes || confusionMatrix.length === 0) {
        wrap.innerHTML = '<p class="subtext">No confusion matrix data available.</p>';
        return;
    }

    let tableHtml = '<table class="matrix-table"><thead><tr><th>Actual \\ Pred</th>';
    classes.forEach(c => {
        tableHtml += `<th>${escapeHtml(c)}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';

    const maxVal = Math.max(...confusionMatrix.flat(), 1);

    confusionMatrix.forEach((row, i) => {
        const safeRowCls = escapeHtml(classes[i] || 'Class ' + i);
        tableHtml += `<tr><th>${safeRowCls}</th>`;
        row.forEach((val, j) => {
            const intensity = Math.min((val / maxVal), 1);
            const bgColor = i === j ? `rgba(34, 197, 94, ${0.2 + intensity * 0.6})` : `rgba(239, 68, 68, ${intensity * 0.5})`;
            tableHtml += `<td style="background-color: ${bgColor}; color: #fff;" class="matrix-cell">${val}</td>`;
        });
        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table>';
    wrap.innerHTML = tableHtml;
}


async function handleCreateModel(e) {
    e.preventDefault();
    const name = document.getElementById('create-name').value;
    const task_type = document.getElementById('create-task-type').value;
    const description = document.getElementById('create-desc').value;
    const confidence_threshold = parseFloat(document.getElementById('create-threshold').value);

    try {
        const res = await fetch(getApiUrl('/api/v1/models'), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name, task_type, description, confidence_threshold })
        });
        const data = await res.json();

        if (res.ok && data.status === 'ok') {
            document.getElementById('modal-create-model').classList.add('hidden');
            document.getElementById('form-create-model').reset();
            showToast(`Model '${data.model.name}' created successfully!`, 'success');
            await loadModelsList();
            selectModel(data.model.id);
        } else {
            const errDetail = data.detail ? (data.detail.message || JSON.stringify(data.detail)) : (data.message || 'Error');
            showToast('Failed to create model: ' + errDetail, 'error');
        }
    } catch (err) {
        showToast('Error creating model: ' + err.message, 'error');
    }
}


async function handleAutoTrain(e) {
    e.preventDefault();
    if (!currentSelectedModel) return;

    const fileInput = document.getElementById('autotrain-file');
    if (!fileInput.files[0]) return;

    const statusBox = document.getElementById('autotrain-status');
    const gpuNotice = document.getElementById('gpu-handoff-notice');
    const btnText = document.getElementById('train-btn-text');
    const spinner = document.getElementById('train-spinner');

    btnText.textContent = '⏳ Training in progress...';
    spinner.classList.remove('hidden');
    statusBox.classList.remove('hidden');
    statusBox.style.background = 'var(--bg-input)';
    statusBox.textContent = '⏳ Validating dataset, checking class balance, and executing CPU training pipeline...';
    gpuNotice.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    if (currentSelectedModel.task_type === 'text') {
        const textCol = document.getElementById('autotrain-text-col').value;
        const labelCol = document.getElementById('autotrain-label-col').value;
        if (textCol) formData.append('text_column', textCol);
        if (labelCol) formData.append('label_column', labelCol);
    }

    try {
        const res = await fetch(getApiUrl(`/api/v1/models/${currentSelectedModel.id}/autotrain`), {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            if (data.status === 'gpu_handoff_required') {
                statusBox.classList.add('hidden');
                gpuNotice.classList.remove('hidden');
                document.getElementById('gpu-handoff-reason').textContent = data.message;
                const link = document.getElementById('gpu-download-link');
                link.href = `/static/../notebooks/${data.gpu_handoff.notebook_filename}`;
                showToast('GPU Handoff required for large dataset.', 'error');
            } else {
                statusBox.style.background = 'rgba(34, 197, 94, 0.2)';
                statusBox.textContent = `✅ ${data.message} (Version ${data.version_number} created with Acc: ${(data.metrics.accuracy * 100).toFixed(1)}%)`;
                showToast(`Model auto-trained successfully! Acc: ${(data.metrics.accuracy * 100).toFixed(1)}%`, 'success');
                selectModel(currentSelectedModel.id);
            }
        } else {
            statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
            let issueText = data.detail ? (data.detail.message || JSON.stringify(data.detail)) : 'Training failed';
            if (data.detail && data.detail.issues) {
                issueText += ' Issues: ' + data.detail.issues.join('; ');
            }
            statusBox.textContent = `❌ ${escapeHtml(issueText)}`;
            showToast('Dataset validation or training failed.', 'error');
        }
    } catch (err) {
        statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
        statusBox.textContent = `❌ Network error: ${escapeHtml(err.message)}`;
        showToast('Network error during training request.', 'error');
    } finally {
        btnText.textContent = '⚡ Start Auto-Train Pipeline';
        spinner.classList.add('hidden');
    }
}


async function handleUploadModel(e) {
    e.preventDefault();
    if (!currentSelectedModel) return;

    const classes = document.getElementById('upload-classes').value;
    const fileInput = document.getElementById('upload-model-file');
    const prepInput = document.getElementById('upload-prep-file');

    if (!fileInput.files[0]) return;

    const statusBox = document.getElementById('upload-status');
    statusBox.classList.remove('hidden');
    statusBox.style.background = 'var(--bg-input)';
    statusBox.textContent = '⏳ Validating uploaded model file and running sanity check...';

    const formData = new FormData();
    formData.append('classes', classes);
    formData.append('file', fileInput.files[0]);
    if (prepInput.files[0]) {
        formData.append('preprocessor_file', prepInput.files[0]);
    }

    try {
        const res = await fetch(getApiUrl(`/api/v1/models/${currentSelectedModel.id}/upload`), {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            statusBox.style.background = 'rgba(34, 197, 94, 0.2)';
            statusBox.textContent = `✅ ${data.message} (Version ${data.version_number} registered as active)`;
            showToast('Model file uploaded and registered!', 'success');
            selectModel(currentSelectedModel.id);
        } else {
            statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
            statusBox.textContent = `❌ ${escapeHtml(data.detail ? data.detail.message : 'Upload failed')}`;
            showToast('Model upload validation failed.', 'error');
        }
    } catch (err) {
        statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
        statusBox.textContent = `❌ Network error: ${escapeHtml(err.message)}`;
        showToast('Upload network error.', 'error');
    }
}


async function handleRollback(versionNumber) {
    if (!currentSelectedModel) return;

    try {
        const res = await fetch(getApiUrl(`/api/v1/models/${currentSelectedModel.id}/rollback`), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ version_number: versionNumber })
        });
        const data = await res.json();

        if (res.ok) {
            showToast(`Rolled back model active version to v${versionNumber}`, 'success');
            selectModel(currentSelectedModel.id);
        } else {
            showToast('Rollback failed: ' + (data.detail ? data.detail.message : 'Unknown error'), 'error');
        }
    } catch (err) {
        showToast('Rollback error: ' + err.message, 'error');
    }
}


async function handleRunInference() {
    if (!currentSelectedModel) return;

    const resultsCard = document.getElementById('inference-results');
    const btnText = document.getElementById('infer-btn-text');
    const spinner = document.getElementById('infer-spinner');

    resultsCard.classList.add('hidden');
    btnText.textContent = '⏳ Executing Inference...';
    spinner.classList.remove('hidden');

    const headers = {
        'X-API-Key': currentSelectedModel.api_key
    };

    let bodyData = null;

    if (currentSelectedModel.task_type === 'vision') {
        const fileInput = document.getElementById('test-image-input');
        if (!fileInput.files[0]) {
            showToast('Please select an image file first.', 'error');
            btnText.textContent = '🚀 Run Inference';
            spinner.classList.add('hidden');
            return;
        }
        bodyData = new FormData();
        bodyData.append('image_file', fileInput.files[0]);
    } else {
        const textInput = document.getElementById('test-text-input').value;
        if (!textInput) {
            showToast('Please enter text for inference.', 'error');
            btnText.textContent = '🚀 Run Inference';
            spinner.classList.add('hidden');
            return;
        }
        headers['Content-Type'] = 'application/json';
        bodyData = JSON.stringify({ text: textInput });
    }

    try {
        const res = await fetch(getApiUrl(`/api/v1/models/${currentSelectedModel.id}/predict`), {
            method: 'POST',
            headers: headers,
            body: bodyData
        });
        const data = await res.json();

        if (res.ok && data.status === 'ok') {
            resultsCard.classList.remove('hidden');
            document.getElementById('res-predicted-class').textContent = data.predicted_class;
            document.getElementById('res-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;

            const uncertainBadge = document.getElementById('res-uncertain-badge');
            if (data.is_low_confidence) {
                uncertainBadge.classList.remove('hidden');
            } else {
                uncertainBadge.classList.add('hidden');
            }

            // Render probabilities
            const barsContainer = document.getElementById('res-probabilities-bars');
            barsContainer.innerHTML = '';
            if (data.class_probabilities) {
                Object.entries(data.class_probabilities).forEach(([cls, prob]) => {
                    const row = document.createElement('div');
                    row.className = 'prob-bar-row';
                    const pct = (prob * 100).toFixed(1);
                    const safeClsName = escapeHtml(cls);
                    row.innerHTML = `
                        <div class="prob-label-row">
                            <span>${safeClsName}</span>
                            <span>${pct}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                        </div>
                    `;
                    barsContainer.appendChild(row);
                });
            }
            showToast('Inference returned successfully!', 'success');
        } else {
            const errDetail = data.detail ? (data.detail.message || JSON.stringify(data.detail)) : JSON.stringify(data);
            showToast('Inference failed: ' + errDetail, 'error');
        }
    } catch (err) {
        showToast('Inference request failed: ' + err.message, 'error');
    } finally {
        btnText.textContent = '🚀 Run Inference';
        spinner.classList.add('hidden');
    }
}
