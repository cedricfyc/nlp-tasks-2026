// Sends the question to POST /ask and shows the answer with its [n] citations and sources.

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("question");
const send = document.getElementById("send");

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function addMessage(html, cls = "") {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = html;
  chat.appendChild(div);
  div.scrollIntoView({ block: "end" });
  return div;
}

let answerCount = 0;

function sourceLink(s) {
  const name = escapeHtml(s.source_title || s.filename);
  return s.url ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${name}</a>` : name;
}

function renderSource(s) {
  const section = s.section ? ` - ${escapeHtml(s.section)}` : "";
  const pages = s.pages ? ` (p. ${escapeHtml(s.pages)})` : "";
  return `<li>${sourceLink(s)}${section}${pages}</li>`;
}

function renderCitation(c, prefix) {
  const section = c.section ? ` - ${escapeHtml(c.section)}` : "";
  const pages = c.pages ? ` (p. ${escapeHtml(c.pages)})` : "";
  return `<li id="${prefix}-${c.id}" value="${c.id}">${sourceLink(c)}${section}${pages}`
       + `<blockquote>${escapeHtml(c.snippet)}</blockquote></li>`;
}

// Turn [n] in the (escaped) answer into a link to source n.
function linkCitations(answerHtml, citations, prefix) {
  const ids = new Set(citations.map(c => c.id));
  return answerHtml.replace(/\[(\d+)\]/g, (match, n) => ids.has(Number(n))
    ? `<a class="cite" href="#${prefix}-${n}">[${n}]</a>` : match);
}

function renderAnswer(data) {
  const prefix = `a${++answerCount}`;
  const citations = data.citations || [];
  const cls = data.refused ? "answer refused" : "answer";
  let html = `<div class="${cls}">${linkCitations(escapeHtml(data.answer), citations, prefix)}</div>`;

  if (citations.length) {
    html += `<div class="sources"><strong>Sources</strong><ol>`
          + citations.map(c => renderCitation(c, prefix)).join("") + `</ol></div>`;
  }
  const check = data.citation_check || {};
  const dropped = (check.dropped_sentences || []).length;
  if (dropped) {
    html += `<div class="meta">${dropped} statement(s) without a source were removed.</div>`;
  }
  if (data.retrieved.length) {
    html += `<details class="sources"><summary>All retrieved context (${data.retrieved.length})</summary><ol>`
          + data.retrieved.map(renderSource).join("") + `</ol></details>`;
  }
  return html + `<div class="meta">${data.seconds} s</div>`;
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) { input.focus(); return; }

  addMessage(escapeHtml(question), "q");
  input.value = "";
  send.disabled = true;
  const pending = addMessage('<span class="meta">Searching and writing an answer...</span>');

  try {
    const response = await fetch("ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    pending.innerHTML = renderAnswer(data);
  } catch (error) {
    pending.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
  } finally {
    send.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
