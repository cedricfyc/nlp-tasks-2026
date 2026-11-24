// Sends the question to POST /ask and shows the answer with the retrieved sources.

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

function renderSource(s) {
  const name = s.url
    ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.filename)}</a>`
    : escapeHtml(s.filename);
  const section = s.section ? ` - ${escapeHtml(s.section)}` : "";
  const pages = s.pages ? ` (p. ${escapeHtml(s.pages)})` : "";
  return `<li>${name}${section}${pages}</li>`;
}

function renderAnswer(data) {
  let html = `<div class="answer">${escapeHtml(data.answer)}</div>`;
  if (data.retrieved.length) {
    html += `<div class="sources"><strong>Retrieved context</strong><ol>`
          + data.retrieved.map(renderSource).join("") + `</ol></div>`;
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
