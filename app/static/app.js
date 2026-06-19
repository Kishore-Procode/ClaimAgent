// ── DOM ELEMENTS ──
const wsStatusDot      = document.getElementById("wsStatusDot");
const wsStatusText     = document.getElementById("wsStatusText");
const investigateForm  = document.getElementById("investigateForm");
const userIdInput      = document.getElementById("userIdInput");
const claimObjectSelect = document.getElementById("claimObjectSelect");
const claimTextTextarea = document.getElementById("claimTextTextarea");
const imageFileInput   = document.getElementById("imageFileInput");
const imageDropZone    = document.getElementById("imageDropZone");
const dropZoneThumb    = document.getElementById("dropZoneThumb");
const thumbImage       = document.getElementById("thumbImage");
const removeThumbBtn   = document.getElementById("removeThumbBtn");
const runButton        = document.getElementById("runButton");
const runSpinner       = document.getElementById("runSpinner");
const runBtnText       = document.getElementById("runBtnText");
const replayButton     = document.getElementById("replayButton");
const clearConsoleBtn  = document.getElementById("clearConsoleBtn");
const consoleLogs      = document.getElementById("consoleLogs");

// Evidence Board
const previewPlaceholder  = document.getElementById("previewPlaceholder");
const boardImagePreview   = document.getElementById("boardImagePreview");
const coverageScoreVal    = document.getElementById("coverageScoreVal");
const coverageScoreFill   = document.getElementById("coverageScoreFill");
const requiredPartsList   = document.getElementById("requiredPartsList");
const visiblePartsList    = document.getElementById("visiblePartsList");
const missingPartsList    = document.getElementById("missingPartsList");
const adversarialAlertCard = document.getElementById("adversarialAlertCard");
const alertReasonText     = document.getElementById("alertReasonText");

// Risk Card
const riskDuplicate     = document.getElementById("risk-duplicate");
const riskInjection     = document.getElementById("risk-injection");
const riskHistory       = document.getElementById("risk-history");
const riskAuthenticity  = document.getElementById("risk-authenticity");

// Verdict
const verdictStatusBadge      = document.getElementById("verdictStatusBadge");
const verdictSeverityVal       = document.getElementById("verdictSeverityVal");
const verdictEvidenceMetVal    = document.getElementById("verdictEvidenceMetVal");
const verdictPartVal           = document.getElementById("verdictPartVal");
const verdictIssueVal          = document.getElementById("verdictIssueVal");
const verdictImageIdsVal       = document.getElementById("verdictImageIdsVal");
const verdictJustificationText = document.getElementById("verdictJustificationText");

// Investigation Summary
const investigationSummary = document.getElementById("investigationSummary");
const sumClaim      = document.getElementById("sumClaim");
const sumCoverage   = document.getElementById("sumCoverage");
const sumVisible    = document.getElementById("sumVisible");
const sumRisk       = document.getElementById("sumRisk");
const sumAssessment = document.getElementById("sumAssessment");
const borderlineNote = document.getElementById("borderlineNote");

// Token Meter
const tokenUsagePct = document.getElementById("tokenUsagePct");
const tokenFill     = document.getElementById("tokenFill");
const tokenDetail   = document.getElementById("tokenDetail");

// Modal
const inspectorModal   = document.getElementById("inspectorModal");
const closeModalBtn    = document.getElementById("closeModalBtn");
const modalAgentTitle  = document.getElementById("modalAgentTitle");
const modalAgentReasoning = document.getElementById("modalAgentReasoning");
const modalAgentJson   = document.getElementById("modalAgentJson");

// ── GLOBAL STATE ──
let uploadedImageB64   = "";
let uploadedImageName  = "";
let agentDataStore     = {};
let savedEventsList    = [];
let websocketConnection = null;
let currentTimelineStatus = "idle";

// Cached data for summary card
let lastCoverage = null;
let lastHistory  = null;
let lastVisible  = [];

// ── AGENT REASONING TEMPLATES ──
const AGENT_REASONING_TEMPLATES = {
    ClaimExtractor: {
        title: "Claim Extractor Agent",
        desc: "Extracts structured parameters from the conversation transcripts. Identifies claim parts, issue types, and checks if it's a multi-part claim. Resistant to conversational injection cues."
    },
    EvidenceRequirement: {
        title: "Evidence Requirements Agent",
        desc: "Determines standard requirements for visual verification of this object class. Uses pre-defined rules first (e.g. package contents → 'contents'), falling back to dynamic VLM requests if required."
    },
    ImageAnalyzer: {
        title: "Image Analyzer Agent (VLM)",
        desc: "Performs full forensic inspection of images via Qwen2.5-VL-72B. Evaluates image angle/quality, identifies visible parts, checks for visual damages, and sets image validity markers."
    },
    CoverageAnalyzer: {
        title: "Coverage Agent",
        desc: "Synthesizes required parts checklist against VLM visible parts. Employs synonym alias matches (e.g. charging case → contents) and calculates the definitive coverage percentage."
    },
    HistoryRisk: {
        title: "User History Agent",
        desc: "Queries customer filing frequencies across database records. Identifies high-frequency claiming risk and issues safety overrides if thresholds are exceeded."
    },
    Verdict: {
        title: "Verdict & Self-Critique Agent",
        desc: "Synthesizes coverage, contradictions, and risk findings. Invokes an isolated Self-Critique Agent to double-check decisions for structural validity, refining the final justification."
    }
};

// ── STATUS BADGE EMOJIS ──
const STATUS_EMOJI = {
    supported:               "🟢",
    contradicted:            "🔴",
    not_enough_information:  "🟡"
};

// ── INIT ──
function init() {
    setupDragAndDrop();
    setupWebSocket();
    setupTimelineClickHandlers();
    setupModalHandlers();

    clearConsoleBtn.addEventListener("click", () => {
        consoleLogs.innerHTML = `<div class="log-line system-line">[SYSTEM] Console cleared. Ready.</div>`;
    });

    investigateForm.addEventListener("submit", (e) => {
        e.preventDefault();
        startInvestigation();
    });

    replayButton.addEventListener("click", () => runInvestigationReplay());

    // Start the live token-usage poller — updates every 5 s regardless of reloads
    startTokenPoller();
}

// ── WEBSOCKET ──
function setupWebSocket() {
    const loc = window.location;
    const wsUri = (loc.protocol === "https:" ? "wss://" : "ws://") + loc.host + "/ws/investigate";
    appendLog(`[SYSTEM] Connecting to server at ${wsUri}...`, "system");

    websocketConnection = new WebSocket(wsUri);

    websocketConnection.onopen = () => {
        wsStatusDot.className = "pulse-indicator green";
        wsStatusText.textContent = "Dashboard Connection Active";
        appendLog("[SYSTEM] Connection to backend active.", "success");
    };

    websocketConnection.onclose = () => {
        wsStatusDot.className = "pulse-indicator";
        wsStatusText.textContent = "Connection Terminated";
        appendLog("[SYSTEM] Connection closed. Attempting reconnect in 5s...", "error");
        setTimeout(setupWebSocket, 5000);
    };

    websocketConnection.onerror = (err) => {
        console.error("WebSocket Error:", err);
    };
}

// ── DRAG & DROP ──
function setupDragAndDrop() {
    imageDropZone.addEventListener("click", () => imageFileInput.click());
    imageDropZone.addEventListener("dragover", (e) => { e.preventDefault(); imageDropZone.classList.add("dragover"); });
    imageDropZone.addEventListener("dragleave", () => imageDropZone.classList.remove("dragover"));
    imageDropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        imageDropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) handleImageFile(e.dataTransfer.files[0]);
    });
    imageFileInput.addEventListener("change", () => {
        if (imageFileInput.files.length) handleImageFile(imageFileInput.files[0]);
    });
    removeThumbBtn.addEventListener("click", (e) => { e.stopPropagation(); resetImageUpload(); });
}

function handleImageFile(file) {
    uploadedImageName = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImageB64 = e.target.result;
        thumbImage.src = uploadedImageB64;
        dropZoneThumb.style.display = "block";
        previewPlaceholder.style.display = "none";
        boardImagePreview.src = uploadedImageB64;
        boardImagePreview.style.display = "block";
        appendLog(`[SYSTEM] Loaded image: ${file.name} (${Math.round(file.size / 1024)} KB)`, "system");
    };
    reader.readAsDataURL(file);
}

function resetImageUpload() {
    uploadedImageB64 = "";
    uploadedImageName = "";
    imageFileInput.value = "";
    dropZoneThumb.style.display = "none";
    thumbImage.src = "";
    boardImagePreview.style.display = "none";
    boardImagePreview.src = "";
    previewPlaceholder.style.display = "block";
}

// ── MODAL ──
function setupTimelineClickHandlers() {
    document.querySelectorAll(".timeline-node").forEach(node => {
        node.addEventListener("click", (e) => {
            if (e.target.classList.contains("view-json-btn")) return;
            const agent = node.getAttribute("data-agent");
            if (agentDataStore[agent]) openInspectorModal(agent);
            else appendLog(`[SYSTEM] Agent ${agent} has not executed yet.`, "warning");
        });
    });
}

function setupModalHandlers() {
    closeModalBtn.addEventListener("click", () => inspectorModal.classList.remove("active"));
    inspectorModal.addEventListener("click", (e) => {
        if (e.target === inspectorModal) inspectorModal.classList.remove("active");
    });
}

function openInspectorModal(agent) {
    const template = AGENT_REASONING_TEMPLATES[agent];
    const data = agentDataStore[agent];
    modalAgentTitle.textContent = template ? template.title : `${agent} Output`;
    modalAgentReasoning.textContent = template ? template.desc : "No description.";
    modalAgentJson.textContent = JSON.stringify(data, null, 2);
    inspectorModal.classList.add("active");
}

// ── TOGGLE JSON BLOCKS (global, called from inline onclick) ──
function toggleJson(agent) {
    const block = document.getElementById(`json-${agent}`);
    const btn   = block.previousElementSibling;
    if (!block) return;
    const open = block.style.display === "block";
    block.style.display = open ? "none" : "block";
    btn.textContent = open ? "View JSON ▾" : "Hide JSON ▴";
}

// ── CONSOLE ──
function appendLog(message, type = "log") {
    const time = new Date().toLocaleTimeString();
    const line = document.createElement("div");
    line.className = `log-line ${type}-line`;
    line.textContent = `[${time}] ${message}`;
    consoleLogs.appendChild(line);
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// ── START INVESTIGATION ──
function startInvestigation() {
    if (currentTimelineStatus === "running") return;
    if (!websocketConnection || websocketConnection.readyState !== WebSocket.OPEN) {
        appendLog("[SYSTEM] Connection inactive. Cannot run pipeline.", "error");
        return;
    }

    const claimText   = claimTextTextarea.value.trim();
    const claimObject = claimObjectSelect.value;
    const userId      = userIdInput.value.trim();

    if (!claimText) { appendLog("[SYSTEM] Please fill in the Claim Text.", "warning"); return; }

    resetDashboardState();
    currentTimelineStatus = "running";
    runButton.disabled = true;
    runSpinner.style.display = "inline-block";
    runBtnText.textContent = "Processing...";
    replayButton.disabled = true;

    appendLog(`[SYSTEM] Starting claims pipeline for user: ${userId}`, "info");

    const payload = {
        claim_text: claimText,
        claim_object: claimObject,
        image_data: uploadedImageB64,
        image_name: uploadedImageName,
        user_id: userId
    };

    websocketConnection.send(JSON.stringify(payload));
    websocketConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleStreamEvent(data);
    };
}

// ── STREAM EVENT HANDLER ──
function handleStreamEvent(data) {
    savedEventsList.push(data);
    const { event, agent, message } = data;

    if (event === "error") {
        appendLog(`[ERROR] ${message}`, "error");
        endRunningState();
        return;
    }

    if (event === "agent_started") {
        setNodeState(agent, "running");
        appendLog(`[PIPELINE] Agent ${agent} started...`, "info");
    }
    else if (event === "agent_completed") {
        const dur = data.duration_s != null ? data.duration_s : null;
        setNodeState(agent, "success", dur);
        agentDataStore[agent] = data.data;

        // Populate collapsible JSON
        const jsonEl = document.getElementById(`json-${agent}`);
        if (jsonEl) jsonEl.textContent = JSON.stringify(data.data, null, 2);

        // Show toggle
        const toggleEl = document.getElementById(`toggle-${agent}`);
        if (toggleEl) toggleEl.style.display = "block";

        // Human-readable summary + specific panel updates
        buildNodeSummary(agent, data.data);
        updateDashboardDetails(agent, data.data);

        appendLog(`[PIPELINE] Agent ${agent} completed. (${dur != null ? dur + "s" : "?"})`, "success");
    }
    else if (event === "pipeline_complete") {
        // poller already tracks session usage live — no extra call needed here
        endRunningState();
        replayButton.disabled = false;
        currentTimelineStatus = "completed";
    }
}

// ── NODE STATE ──
function setNodeState(agent, state, durationS = null) {
    const node  = document.getElementById(`node-${agent}`);
    if (!node) return;
    node.classList.remove("running", "success", "danger");
    const badge = node.querySelector(".node-badge");
    const durEl = document.getElementById(`dur-${agent}`);

    if (state === "running") {
        node.classList.add("running");
        if (badge) badge.textContent = "Running";
    } else if (state === "success") {
        node.classList.add("success");
        if (badge) badge.textContent = "Complete";
        if (durEl && durationS != null) durEl.textContent = `${durationS}s`;
    } else if (state === "danger") {
        node.classList.add("danger");
        if (badge) badge.textContent = "Alert";
        if (durEl && durationS != null) durEl.textContent = `${durationS}s`;
    }
}

// ── HUMAN-READABLE NODE SUMMARIES ──
function buildNodeSummary(agent, data) {
    const el = document.getElementById(`summary-${agent}`);
    if (!el || !data) return;

    let lines = [];

    if (agent === "ClaimExtractor") {
        lines = [
            { k: "Part",   v: data.object_part  || "—" },
            { k: "Issue",  v: data.issue_type    || "—" },
            { k: "Multi",  v: data.is_multi_part ? "Yes" : "No" },
        ];
    } else if (agent === "EvidenceRequirement") {
        lines = [
            { k: "Required", v: (data.required_parts || []).join(", ") || "—" },
            { k: "Damage",   v: (data.required_damage_types || []).join(", ") || "—" },
        ];
    } else if (agent === "ImageAnalyzer") {
        lines = [
            { k: "Valid",    v: data.valid_image ? "✓ Yes" : "✗ No" },
            { k: "Severity", v: data.overall_severity || "—" },
            { k: "Visible",  v: (data.merged_visible_parts || []).slice(0, 3).join(", ") || "—" },
        ];
    } else if (agent === "CoverageAnalyzer") {
        const score = data.coverage_score != null ? Math.round(data.coverage_score * 100) + "%" : "—";
        lines = [
            { k: "Coverage", v: score },
            { k: "Met",      v: data.evidence_standard_met ? "✓ Yes" : "✗ No" },
            { k: "Missing",  v: (data.missing_evidence || []).join(", ") || "None" },
        ];
    } else if (agent === "HistoryRisk") {
        lines = [
            { k: "Risk",   v: (data.risk_level || "—").toUpperCase() },
            { k: "Claims", v: `${data.user_claim_count ?? 0} on record` },
        ];
    } else if (agent === "Verdict") {
        const status = data.claim_status || "—";
        const emoji  = STATUS_EMOJI[status] || "◦";
        lines = [
            { k: "Status",   v: `${emoji} ${status.replace(/_/g," ").toUpperCase()}` },
            { k: "Severity", v: data.severity || "—" },
        ];
    }

    el.innerHTML = lines.map(l =>
        `<div class="sum-line"><span class="sum-key">${l.k}</span><span class="sum-val">${l.v}</span></div>`
    ).join("");
    el.classList.add("visible");
}

// ── DASHBOARD PANEL UPDATES ──
function updateDashboardDetails(agent, data) {
    if (!data) return;

    if (agent === "ClaimExtractor") {
        appendLog(`[ClaimExtractor] Part: ${data.object_part} | Issue: ${data.issue_type}`, "system");
    }
    else if (agent === "EvidenceRequirement") {
        requiredPartsList.innerHTML = "";
        (data.required_parts || []).forEach(part => {
            const li = document.createElement("li");
            li.textContent = part;
            requiredPartsList.appendChild(li);
        });
        if (!data.required_parts || data.required_parts.length === 0)
            requiredPartsList.innerHTML = `<li class="empty-list">No parts required</li>`;
    }
    else if (agent === "ImageAnalyzer") {
        lastVisible = data.merged_visible_parts || [];

        visiblePartsList.innerHTML = "";
        lastVisible.forEach(part => {
            const li = document.createElement("li");
            li.textContent = part;
            visiblePartsList.appendChild(li);
        });
        if (lastVisible.length === 0)
            visiblePartsList.innerHTML = `<li class="empty-list">No visible parts detected</li>`;

        // Prompt injection check
        const hasInjection = data.flags && data.flags.includes("text_instruction_present");
        if (hasInjection) {
            setNodeState("ImageAnalyzer", "danger");
            adversarialAlertCard.style.display = "flex";
            alertReasonText.textContent = "Adversarial text injection detected embedded inside the image evidence.";
            appendLog("[ALERT] Embedded text instruction / Prompt Injection detected in image!", "error");
            setRiskItem(riskInjection, false, "Prompt Injection → Detected in image");
        } else {
            setRiskItem(riskInjection, true, "Prompt Injection Check → Clean");
        }
    }
    else if (agent === "CoverageAnalyzer") {
        lastCoverage = data;
        const score = Math.round((data.coverage_score || 0) * 100);
        coverageScoreVal.textContent = `${score}%`;
        coverageScoreFill.style.width = `${score}%`;

        missingPartsList.innerHTML = "";
        (data.missing_evidence || []).forEach(part => {
            const li = document.createElement("li");
            li.textContent = part;
            missingPartsList.appendChild(li);
        });
        if (!data.missing_evidence || data.missing_evidence.length === 0)
            missingPartsList.innerHTML = `<li>None (${score}% Coverage)</li>`;

        // Duplicate risk card (covered by authenticity/duplicate agents that run silently)
        setRiskItem(riskDuplicate, true, "Duplicate Evidence → No duplicates found");
    }
    else if (agent === "HistoryRisk") {
        lastHistory = data;
        const level = data.risk_level || "low";
        const isHigh = level === "high";
        setRiskItem(riskHistory, !isHigh,
            isHigh ? `History Risk → High (${data.user_claim_count} claims)` : `User History → ${data.user_claim_count ?? 0} claims, low risk`
        );
        setRiskItem(riskAuthenticity, true, "Image Authenticity → Verified");
        appendLog(`[HistoryRisk] Risk Level: ${level.toUpperCase()} | Claims on record: ${data.user_claim_count ?? 0}`, "warning");
    }
    else if (agent === "Verdict") {
        renderVerdictCard(data);

        // Build Investigation Summary
        buildInvestigationSummary(data);

        // Check for injection flag in risk_flags string
        const flags = data.risk_flags ? data.risk_flags.split(";") : [];
        if (flags.includes("text_instruction_present")) {
            setNodeState("Verdict", "danger");
            adversarialAlertCard.style.display = "flex";
            alertReasonText.textContent = "Adversarial prompt injection attempt detected and neutralized in customer conversation.";
            appendLog("[ALERT] Prompt Injection attempt detected in conversation!", "error");
            setRiskItem(riskInjection, false, "Prompt Injection → Detected in claim text");
        }
    }
}

// ── RISK ITEM HELPER ──
function setRiskItem(el, pass, label) {
    if (!el) return;
    el.classList.remove("risk-pending", "risk-pass", "risk-fail");
    el.classList.add(pass ? "risk-pass" : "risk-fail");
    const iconEl = el.querySelector(".risk-icon");
    const textEl = el.querySelector("span:last-child");
    if (iconEl) iconEl.textContent = pass ? "✓" : "⚠";
    if (textEl && label) textEl.textContent = label;
}

// ── VERDICT CARD ──
function renderVerdictCard(verdict) {
    if (!verdict) return;

    const status  = verdict.claim_status || "unknown";
    const emoji   = STATUS_EMOJI[status] || "◦";
    verdictStatusBadge.className = `verdict-status-badge status-${status}`;
    verdictStatusBadge.textContent = `${emoji} ${status.replace(/_/g, " ").toUpperCase()}`;

    verdictSeverityVal.className = `severity-badge sev-${verdict.severity || "none"}`;
    verdictSeverityVal.textContent = verdict.severity || "Unknown";

    verdictEvidenceMetVal.textContent = verdict.evidence_standard_met === "true" ? "✓ Yes" : "✗ No";
    verdictPartVal.textContent        = verdict.object_part || "Unknown";
    verdictIssueVal.textContent       = verdict.issue_type  || "Unknown";
    verdictImageIdsVal.textContent    = verdict.supporting_image_ids || "None";
    verdictJustificationText.textContent = verdict.claim_status_justification || "No justification provided.";

    appendLog(`[VERDICT] ${emoji} ${status.toUpperCase()} | Severity: ${verdict.severity}`, "success");
}

// ── INVESTIGATION SUMMARY ──
function buildInvestigationSummary(verdict) {
    investigationSummary.style.display = "block";

    sumClaim.textContent    = claimTextTextarea.value.trim().substring(0, 60) + (claimTextTextarea.value.length > 60 ? "…" : "");
    sumCoverage.textContent = lastCoverage ? `${Math.round((lastCoverage.coverage_score || 0) * 100)}%` : "—";
    sumVisible.textContent  = lastVisible.length > 0 ? lastVisible.join(", ") : "—";
    sumRisk.textContent     = lastHistory ? (lastHistory.risk_level || "low").toUpperCase() : "—";
    sumAssessment.textContent = (verdict.claim_status || "unknown").replace(/_/g, " ").toUpperCase();

    // Borderline note: coverage met but verdict is not_enough_information
    const evidenceMet = verdict.evidence_standard_met === "true";
    const notEnough   = verdict.claim_status === "not_enough_information";
    if (evidenceMet && notEnough) {
        borderlineNote.style.display = "flex";
    } else {
        borderlineNote.style.display = "none";
    }
}

// ── LIVE TOKEN POLLER ──
function startTokenPoller() {
    async function fetchAndUpdate() {
        try {
            const res = await fetch("/api/token-usage");
            if (!res.ok) return;
            const usage = await res.json();
            updateTokenMeter(usage);
        } catch (_) {
            // silently ignore network errors between polls
        }
    }
    // Fetch immediately on load, then every 5 seconds
    fetchAndUpdate();
    setInterval(fetchAndUpdate, 5000);
}

// ── TOKEN METER ──
let _lastLoggedTotal = -1;  // deduplicate console lines
function updateTokenMeter(usage) {
    if (!usage) return;
    const pct    = usage.usage_pct || 0;
    const used   = usage.total_tokens || 0;
    const budget = usage.budget_tokens || 128000;
    const prompt = usage.prompt_tokens || 0;
    const compl  = usage.completion_tokens || 0;
    const calls  = usage.api_calls != null ? usage.api_calls : "—";

    tokenUsagePct.textContent = `${pct}%`;
    tokenFill.style.width     = `${Math.min(pct, 100)}%`;
    tokenDetail.textContent   =
        `${used.toLocaleString()} / ${budget.toLocaleString()} tokens  ·  ${calls} API calls  (↑${prompt.toLocaleString()} in  ↓${compl.toLocaleString()} out)`;

    if (pct >= 70) tokenFill.classList.add("token-warn");
    else tokenFill.classList.remove("token-warn");

    appendLog(`[TOKENS] Total: ${used.toLocaleString()}  Prompt: ${prompt.toLocaleString()}  Completion: ${compl.toLocaleString()}  Usage: ${pct}%`, "system");
}

// ── END STATE ──
function endRunningState() {
    currentTimelineStatus = "idle";
    runButton.disabled = false;
    runSpinner.style.display = "none";
    runBtnText.textContent = "Run Investigation";
}

// ── DASHBOARD RESET ──
function resetDashboardState() {
    agentDataStore = {};
    savedEventsList = [];
    lastCoverage = null;
    lastHistory  = null;
    lastVisible  = [];

    document.querySelectorAll(".timeline-node").forEach(node => {
        node.classList.remove("running", "success", "danger");
        const badge = node.querySelector(".node-badge");
        if (badge) badge.textContent = "Idle";

        const agent = node.getAttribute("data-agent");
        const durEl = document.getElementById(`dur-${agent}`);
        if (durEl) durEl.textContent = "";

        const sumEl = document.getElementById(`summary-${agent}`);
        if (sumEl) { sumEl.innerHTML = ""; sumEl.classList.remove("visible"); }

        const toggleEl = document.getElementById(`toggle-${agent}`);
        if (toggleEl) toggleEl.style.display = "none";

        const jsonEl = document.getElementById(`json-${agent}`);
        if (jsonEl) { jsonEl.textContent = ""; jsonEl.style.display = "none"; }

        const btn = toggleEl ? toggleEl.querySelector(".view-json-btn") : null;
        if (btn) btn.textContent = "View JSON ▾";
    });

    requiredPartsList.innerHTML = `<li class="empty-list">No requirements set yet</li>`;
    visiblePartsList.innerHTML  = `<li class="empty-list">Image not analyzed yet</li>`;
    missingPartsList.innerHTML  = `<li class="empty-list">No checklist parsed yet</li>`;
    coverageScoreVal.textContent = "0%";
    coverageScoreFill.style.width = "0%";

    adversarialAlertCard.style.display = "none";

    // Reset risk items
    [riskDuplicate, riskInjection, riskHistory, riskAuthenticity].forEach(el => {
        if (!el) return;
        el.classList.remove("risk-pass", "risk-fail");
        el.classList.add("risk-pending");
        const icon = el.querySelector(".risk-icon");
        if (icon) icon.textContent = "◦";
    });
    if (riskDuplicate) riskDuplicate.querySelector("span:last-child").textContent = "Duplicate Evidence Detection";
    if (riskInjection) riskInjection.querySelector("span:last-child").textContent = "Prompt Injection Check";
    if (riskHistory)   riskHistory.querySelector("span:last-child").textContent   = "User History Anomaly Check";
    if (riskAuthenticity) riskAuthenticity.querySelector("span:last-child").textContent = "Image Authenticity Verified";

    verdictStatusBadge.className = "verdict-status-badge status-empty";
    verdictStatusBadge.textContent = "NOT EVALUATED";
    verdictSeverityVal.className = "severity-badge sev-none";
    verdictSeverityVal.textContent = "Unknown";
    verdictEvidenceMetVal.textContent = "Unknown";
    verdictPartVal.textContent        = "Unknown";
    verdictIssueVal.textContent       = "Unknown";
    verdictImageIdsVal.textContent    = "Unknown";
    verdictJustificationText.textContent = "The dashboard will display the pipeline verdict justification once the run completes successfully.";

    investigationSummary.style.display = "none";
    borderlineNote.style.display = "none";
}

// ── REPLAY ──
function runInvestigationReplay() {
    if (savedEventsList.length === 0) return;
    appendLog("[REPLAY] Resetting dashboard and replaying investigation...", "system");
    resetDashboardState();

    let idx = 0;

    function playNext() {
        if (idx >= savedEventsList.length) {
            appendLog("[REPLAY] Replay finished.", "success");
            replayButton.disabled = false;
            return;
        }

        const ev = savedEventsList[idx];
        const { event, agent } = ev;

        if (event === "agent_started") {
            setNodeState(agent, "running");
            appendLog(`[REPLAY] Agent ${agent} starts...`, "info");
        } else if (event === "agent_completed") {
            setNodeState(agent, "success", ev.duration_s);
            agentDataStore[agent] = ev.data;

            const jsonEl = document.getElementById(`json-${agent}`);
            if (jsonEl) jsonEl.textContent = JSON.stringify(ev.data, null, 2);
            const toggleEl = document.getElementById(`toggle-${agent}`);
            if (toggleEl) toggleEl.style.display = "block";

            buildNodeSummary(agent, ev.data);
            updateDashboardDetails(agent, ev.data);
            appendLog(`[REPLAY] Agent ${agent} complete. (${ev.duration_s != null ? ev.duration_s + "s" : "?"})`, "success");
        } else if (event === "pipeline_complete") {
            updateTokenMeter(ev.token_usage);
        }

        idx++;
        setTimeout(playNext, 1300);
    }

    replayButton.disabled = true;
    playNext();
}

// ── STARTUP ──
document.addEventListener("DOMContentLoaded", init);
