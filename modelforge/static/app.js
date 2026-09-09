let currentSelectedModel = null;

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
        alert('API Key copied to clipboard!');
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


async function loadModelsList() {
    const container = document.getElementById('models-list-container');
    container.innerHTML = '<p class="placeholder-text">Loading models...</p>';

    try {
        const res = await fetch('/api/v1/models');
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
                if (model.metrics && model.metrics.accuracy !== undefined) {
                    const accPct = (model.metrics.accuracy * 100).toFixed(1);
                    const isHeldOut = model.metrics.is_trustworthy_held_out;
                    accuracyText = `Acc: ${accPct}% (${isHeldOut ? 'Held-out test' : 'Sanity'})`;
                }

                card.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3>${model.name}</h3>
                        <span class="badge">${model.task_type}</span>
                    </div>
                    <p class="subtext" style="margin: 8px 0;">${model.description || 'No description'}</p>
                    <div style="font-size:13px; font-weight:600; color: var(--accent-green);">
                        ${accuracyText}
                    </div>
                    <div style="font-size:12px; color: var(--text-muted); margin-top: 4px;">
                        Versions: ${model.version_count}
                    </div>
                `;

                card.addEventListener('click', () => selectModel(model.id));
                container.appendChild(card);
            });
        }
    } catch (err) {
        container.innerHTML = `<p class="placeholder-text" style="color:var(--accent-red)">Failed to load models: ${err.message}</p>`;
    }
}


async function selectModel(modelId) {
    try {
        const res = await fetch(`/api/v1/models/${modelId}`);
        const data = await res.json();

        if (data.status === 'ok') {
            currentSelectedModel = data.model;
            renderModelDetails(data.model);
            loadModelsList(); // Highlight selected
        }
    } catch (err) {
        alert('Failed to load model details: ' + err.message);
    }
}


function renderModelDetails(model) {
    document.getElementById('model-detail-section').classList.remove('hidden');
    document.getElementById('detail-model-name').textContent = model.name;
    document.getElementById('detail-model-desc').textContent = model.description || 'No description provided.';
    document.getElementById('detail-task-type').textContent = model.task_type.toUpperCase();
    document.getElementById('detail-model-id').textContent = `ID: ${model.id}`;
    document.getElementById('detail-api-key').value = model.api_key;

    // Toggle text column options in Auto-Train form
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

    renderVersionHistory(model.versions);
}


function renderVersionHistory(versions) {
    const listContainer = document.getElementById('version-history-list');
    if (!versions || versions.length === 0) {
        listContainer.innerHTML = '<p class="subtext">No versions registered for this model yet.</p>';
        return;
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
                    perClassRows += `
                        <tr>
                            <td><strong>${cls}</strong></td>
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
                <div style="font-size:12px; margin-bottom:8px;">
                    <strong>Evaluation Split Integrity:</strong>
                    <span class="badge ${em.is_trustworthy_held_out ? 'badge-free' : 'badge-outline'}">
                        ${em.is_trustworthy_held_out ? '✅ Genuine Held-out Test Split' : '⚠️ Training/Sanity Check Only'}
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

        item.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong>Version ${v.version_number}</strong>
                    <span class="badge" style="margin-left:8px;">${v.format}</span>
                    <span class="subtext" style="margin-left:8px;">Source: ${v.source}</span>
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


async function handleCreateModel(e) {
    e.preventDefault();
    const name = document.getElementById('create-name').value;
    const task_type = document.getElementById('create-task-type').value;
    const description = document.getElementById('create-desc').value;
    const confidence_threshold = parseFloat(document.getElementById('create-threshold').value);

    try {
        const res = await fetch('/api/v1/models', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name, task_type, description, confidence_threshold })
        });
        const data = await res.json();

        if (data.status === 'ok') {
            document.getElementById('modal-create-model').classList.add('hidden');
            document.getElementById('form-create-model').reset();
            await loadModelsList();
            selectModel(data.model.id);
        } else {
            alert('Failed to create model: ' + data.message);
        }
    } catch (err) {
        alert('Error creating model: ' + err.message);
    }
}


async function handleAutoTrain(e) {
    e.preventDefault();
    if (!currentSelectedModel) return;

    const fileInput = document.getElementById('autotrain-file');
    if (!fileInput.files[0]) return;

    const statusBox = document.getElementById('autotrain-status');
    const gpuNotice = document.getElementById('gpu-handoff-notice');
    statusBox.classList.remove('hidden');
    statusBox.style.background = 'var(--bg-input)';
    statusBox.textContent = '⏳ Validating dataset and running training pipeline...';
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
        const res = await fetch(`/api/v1/models/${currentSelectedModel.id}/autotrain`, {
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
            } else {
                statusBox.style.background = 'rgba(34, 197, 94, 0.2)';
                statusBox.textContent = `✅ ${data.message} (Version ${data.version_number} created with Acc: ${(data.metrics.accuracy * 100).toFixed(1)}%)`;
                selectModel(currentSelectedModel.id);
            }
        } else {
            statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
            let issueText = data.detail ? (data.detail.message || JSON.stringify(data.detail)) : 'Training failed';
            statusBox.textContent = `❌ ${issueText}`;
        }
    } catch (err) {
        statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
        statusBox.textContent = `❌ Network error: ${err.message}`;
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
    statusBox.textContent = '⏳ Validating uploaded model file...';

    const formData = new FormData();
    formData.append('classes', classes);
    formData.append('file', fileInput.files[0]);
    if (prepInput.files[0]) {
        formData.append('preprocessor_file', prepInput.files[0]);
    }

    try {
        const res = await fetch(`/api/v1/models/${currentSelectedModel.id}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            statusBox.style.background = 'rgba(34, 197, 94, 0.2)';
            statusBox.textContent = `✅ ${data.message} (Version ${data.version_number} registered as active)`;
            selectModel(currentSelectedModel.id);
        } else {
            statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
            statusBox.textContent = `❌ ${data.detail ? data.detail.message : 'Upload failed'}`;
        }
    } catch (err) {
        statusBox.style.background = 'rgba(239, 68, 68, 0.2)';
        statusBox.textContent = `❌ Network error: ${err.message}`;
    }
}


async function handleRollback(versionNumber) {
    if (!currentSelectedModel) return;

    try {
        const res = await fetch(`/api/v1/models/${currentSelectedModel.id}/rollback`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ version_number: versionNumber })
        });
        const data = await res.json();

        if (res.ok) {
            alert(`Successfully rolled back to active version ${versionNumber}`);
            selectModel(currentSelectedModel.id);
        } else {
            alert('Rollback failed: ' + (data.detail ? data.detail.message : 'Unknown error'));
        }
    } catch (err) {
        alert('Rollback error: ' + err.message);
    }
}


async function handleRunInference() {
    if (!currentSelectedModel) return;

    const resultsCard = document.getElementById('inference-results');
    resultsCard.classList.add('hidden');

    const headers = {
        'X-API-Key': currentSelectedModel.api_key
    };

    let bodyData = null;

    if (currentSelectedModel.task_type === 'vision') {
        const fileInput = document.getElementById('test-image-input');
        if (!fileInput.files[0]) {
            alert('Please select an image file first.');
            return;
        }
        bodyData = new FormData();
        bodyData.append('image_file', fileInput.files[0]);
    } else {
        const textInput = document.getElementById('test-text-input').value;
        if (!textInput) {
            alert('Please enter text for inference.');
            return;
        }
        headers['Content-Type'] = 'application/json';
        bodyData = JSON.stringify({ text: textInput });
    }

    try {
        const res = await fetch(`/api/v1/models/${currentSelectedModel.id}/predict`, {
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
                    row.innerHTML = `
                        <div class="prob-label-row">
                            <span>${cls}</span>
                            <span>${pct}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                        </div>
                    `;
                    barsContainer.appendChild(row);
                });
            }
        } else {
            alert('Inference error: ' + (data.detail ? data.detail.message : JSON.stringify(data)));
        }
    } catch (err) {
        alert('Inference request failed: ' + err.message);
    }
}
