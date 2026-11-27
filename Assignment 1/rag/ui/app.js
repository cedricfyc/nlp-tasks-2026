// Chat page: sends the question (and the chosen number of sources, top_k) to POST /ask,
// shows the answer with clickable [n] citations, its sources and any refusal, and keeps a
// history of this session's questions in the browser (no API call to reopen one).
{
  const K_CHOICES = [3, 4, 5];
  const CANDIDATES = 20;
  const SUGGESTIONS = [
    "What is the maximum rental deposit a landlord may ask for?",
    "How much is the broadcasting licence fee in a shared flat?",
    "Which number do I call in a medical emergency?",
    "What is a Sperrkonto?",
  ];
  const REASONS = {
    model: "The retrieved passages do not contain the answer.",
    no_valid_citations: "No statement could be backed by a source.",
    no_context: "No matching passages were found.",
  };

  const state = { history: [], sel: null, k: 3, busy: false };
  const el = id => document.getElementById(id);
  const content = el("content"), input = el("question"), send = el("send");

  function esc(text) {
    return String(text ?? "").replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------- sidebar and k picker ----------
  function renderHistory() {
    const box = el("history");
    if (!state.history.length) {
      box.innerHTML = '<div class="history-empty">Your questions will appear here.</div>';
      return;
    }
    box.innerHTML = state.history.map((h, i) => {
      let tag = "Answering…", cls = "";
      if (h.error) tag = "Error";
      else if (h.data && h.data.refused) { tag = "Refused"; cls = " refused"; }
      else if (h.data) tag = `${(h.data.citations || []).length} sources`;
      return `<button type="button" class="history-item${i === state.sel ? " active" : ""}" data-i="${i}">`
           + `<span class="history-q">${esc(h.q)}</span><span class="history-tag${cls}">${tag}</span></button>`;
    }).join("");
  }

  function renderK() {
    el("k-options").innerHTML = K_CHOICES.map(k =>
      `<button type="button" data-k="${k}" aria-pressed="${k === state.k}">${k}</button>`).join("");
  }

  // ---------- main area ----------
  function welcome() {
    return `<div class="welcome">
      <h1>Ask about studying and living in Germany</h1>
      <p>Every answer cites the guide it comes from. If the guides don't cover a question, the assistant says so instead of guessing.</p>
      <div class="try-label">Try one</div>
      <div class="suggestions">${SUGGESTIONS.map(q =>
        `<button type="button" data-ask="${esc(q)}">${esc(q)}</button>`).join("")}</div>
    </div>`;
  }

  function refusalText(data) {
    const r = data.refusal;
    if (data.refused) {
      if (r && r.refuse) return `Refusal check (${esc(r.method)}): P(unanswerable) = ${r.p_unanswerable} ≥ ${r.threshold}`;
      return esc(REASONS[(data.citation_check || {}).refusal_reason] || "Not answerable from the guides.");
    }
    return r ? `refusal check passed (${esc(r.method)})` : "refusal check off";
  }

  function answerCard(h) {
    const data = h.data, cites = data.citations || [];
    const ids = new Set(cites.map(c => c.id));
    const text = esc(data.answer).replace(/\[(\d+)\]/g, (m, n) => ids.has(Number(n))
      ? `<button type="button" class="cite${h.focus === Number(n) ? " on" : ""}" data-cite="${n}" aria-label="Show source ${n}">${n}</button>`
      : m);

    let html = `<div class="card${data.refused ? " refused" : ""}">`;
    if (data.refused) {
      html += `<div class="refused-label"><svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.5"/><path d="M8 4.5v4M8 11v.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>Not covered by the guides</div>`;
    }
    html += `<p class="answer">${text}</p>`;
    const used = data.retrieved ? data.retrieved.length : data.top_k;
    html += `<div class="meta">${data.seconds} s · ${cites.length} cited of ${used} passages used · ${refusalText(data)}</div>`;

    if (cites.length) {
      html += `<div class="sources"><div class="sources-title">Sources</div>` + cites.map(c => {
        const title = esc(c.source_title || c.filename);
        const link = c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">${title}</a>` : title;
        const sub = [c.filename !== c.source_title ? c.filename : "", c.section, c.pages ? `p. ${c.pages}` : ""]
          .filter(Boolean).map(esc).join(" · ");
        return `<div class="source${h.focus === c.id ? " on" : ""}" id="src-${h.n}-${c.id}">`
             + `<span class="source-n">${c.id}</span><div class="source-body">`
             + `<span class="source-title">${link}</span>`
             + (sub ? `<span class="source-sub">${sub}</span>` : "")
             + (c.snippet ? `<span class="source-snippet">“${esc(c.snippet)}”</span>` : "")
             + `</div></div>`;
      }).join("") + `</div>`;
    }

    const passages = data.retrieved || [];
    if (passages.length) {
      html += `<details class="passages"><summary>Passages used: best ${passages.length} of ${data.candidates || CANDIDATES}`
            + `${passages.some(p => p.score != null) ? " after re-ranking" : ""}</summary>`
            + passages.map(p => {
                const name = [p.filename, p.section, p.pages ? `p. ${p.pages}` : ""].filter(Boolean).map(esc).join(" · ");
                return `<div class="passage"><span>${name}</span><span>${p.score != null ? "score " + p.score : ""}</span></div>`;
              }).join("") + `</details>`;
    }
    return html + `</div>`;
  }

  let shownKey = "";
  function renderMain() {
    const h = state.sel === null ? null : state.history[state.sel];
    if (!h) { content.innerHTML = welcome(); el("scroll").scrollTop = 0; shownKey = ""; return; }
    let html = `<div class="bubble">${esc(h.q)}</div>`;
    if (h.error) html += `<div class="card error">${esc(h.error)}</div>`;
    else if (!h.data) html += `<div class="card pending">Searching ${CANDIDATES} candidate passages, keeping the best ${h.k} and writing an answer…</div>`;
    else html += answerCard(h);
    content.innerHTML = html;
    const key = `${h.n}:${!!h.data}:${!!h.error}`;      // a new question or its answer arrived
    if (key !== shownKey) { el("scroll").scrollTop = 0; shownKey = key; }
  }

  function render() { renderHistory(); renderK(); renderMain(); send.disabled = state.busy; }

  // ---------- asking ----------
  let count = 0;
  async function ask(question) {
    const q = String(question || "").trim();
    if (!q || state.busy) return;
    const h = { q, k: state.k, n: ++count, data: null, error: null, focus: 0 };
    state.history.unshift(h);
    state.sel = 0;
    state.busy = true;
    input.value = "";
    render();
    try {
      const response = await fetch("ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: h.k }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
      h.data = data;
    } catch (error) {
      h.error = error.message;
    } finally {
      state.busy = false;
      render();
      input.focus();
    }
  }

  // ---------- events ----------
  el("form").addEventListener("submit", event => { event.preventDefault(); ask(input.value); });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(input.value); }
  });
  el("new-chat").addEventListener("click", () => { state.sel = null; render(); input.focus(); });
  el("history").addEventListener("click", event => {
    const item = event.target.closest("[data-i]");
    if (item) { state.sel = Number(item.dataset.i); render(); }
  });
  el("k-options").addEventListener("click", event => {
    const b = event.target.closest("[data-k]");
    if (b) { state.k = Number(b.dataset.k); renderK(); }
  });
  content.addEventListener("click", event => {
    const s = event.target.closest("[data-ask]");
    if (s) { ask(s.dataset.ask); return; }
    const c = event.target.closest("[data-cite]");
    if (c) {
      const h = state.history[state.sel], n = Number(c.dataset.cite);
      h.focus = h.focus === n ? 0 : n;
      renderMain();
      const target = document.getElementById(`src-${h.n}-${n}`);
      if (target && h.focus) target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });

  render();
}
