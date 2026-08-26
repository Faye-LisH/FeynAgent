// Three jobs: collect the sidebar, stream a run, render the report it ends with.
// The Markdown is the report the CLI writes, so the dialect handled here is
// exactly the one graph.report produces — headings, bullets, fences, $$ blocks.

const $ = (id) => document.getElementById(id);
const SETTINGS = ["neatibp", "singular", "spasm", "provider", "model", "base_url", "api_key"];
const OPTIONS = ["kind", "neatibp_mode", "extract", "arxiv", "max_papers", "workdir",
                 "offline", "reduce"];
const KEPT = "feynman-agent";  // localStorage, minus the key
const MARKS = {ok: "✓", hit: "✓", miss: "·", partial: "~",
               failed: "✗", error: "✗"};

let server = {};   // what the process is configured with
let job = null;    // the run the page is watching

// ---------------------------------------------------------------- settings

function collect(ids) {
  const out = {};
  for (const id of ids) {
    const el = $(id);
    out[id] = el.type === "checkbox" ? el.checked : el.value.trim();
  }
  return out;
}

function restore(values) {
  for (const [id, value] of Object.entries(values)) {
    const el = $(id);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = value;
    else el.value = value;
  }
}

function remember() {
  const kept = {...collect(SETTINGS), ...collect(OPTIONS)};
  delete kept.api_key;  // a key belongs in .env, not in a browser store
  localStorage.setItem(KEPT, JSON.stringify(kept));
}

const TOOLS = ["neatibp", "singular", "spasm"];

async function checkTools() {
  const found = await post("/api/check", collect(TOOLS));
  const missing = TOOLS.filter((name) => !found[name]);
  for (const name of TOOLS) $(name).classList.toggle("missing", !found[name]);
  $("toolhint").textContent = missing.length
    ? `not found: ${missing.join(", ")} — the reduction will be mocked instead`
    : "all three found";
  $("toolhint").classList.toggle("bad", missing.length > 0);
}

function keyHint() {
  const chosen = $("provider").value;
  const name = server.key_vars[chosen];
  const known = server.has_key[chosen];
  $("keyhint").textContent = known
    ? `${name} is already set — leave this blank to keep it`
    : `${name} is not set; type it here, or put it in .env`;
}

// Model and base URL belong to a provider, so switching provider drops them
// back to that provider's defaults — kimi's model at deepseek's endpoint is a
// confusing 404, not a run. The placeholder says what blank will use.
function pickProvider() {
  const fallback = server.defaults[$("provider").value] || {};
  for (const [id, value] of [["model", fallback.model], ["base_url", fallback.base_url]]) {
    $(id).value = "";
    $(id).placeholder = value || "provider default";
  }
  $("figurehint").hidden = fallback.images !== false;
  keyHint();
}

// --------------------------------------------------------------- rendering

const esc = (s) => s.replace(/[&<>"]/g,
  (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// The diagram is not a file yet — it is in the run's state, one request away.
const figure = (src) => (src.endsWith(".svg") ? `/api/figure?job=${job}` : src);

function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, (_, t) => `<code>${t}</code>`)
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => `<img alt="${alt}" src="${figure(src)}">`)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g,
             (_, t, url) => `<a href="${url}" target="_blank" rel="noreferrer">${t}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function render(md) {
  const lines = md.split("\n"), out = [];
  let para = [], quote = [], list = 0;
  const flushPara = () => { if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`), para = []; };
  const flushQuote = () => {
    // One line each: the model's reading is GIVES / WHERE / MATCH, not a paragraph.
    if (quote.length) out.push(`<blockquote>${quote.map(inline).join("<br>")}</blockquote>`), quote = [];
  };
  const flush = () => { flushPara(); flushQuote(); };
  const closeList = () => { while (list > 0) { out.push("</ul>"); list--; } };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const fence = line.match(/^(\s*)```/);   // a master's closed form is indented
    if (fence) {
      flush(); closeList();
      const body = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i].slice(fence[1].length));
      out.push(`<pre class="code"><code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (line.trim() === "$$") {             // left for KaTeX, below
      flush(); closeList();
      const body = [];
      while (++i < lines.length && lines[i].trim() !== "$$") body.push(lines[i]);
      out.push(`<div class="math">${esc(body.join("\n"))}</div>`);
      continue;
    }
    const head = line.match(/^(#{1,4}) +(.*)$/);
    if (head) {
      flush(); closeList();
      out.push(`<h${head[1].length}>${inline(head[2])}</h${head[1].length}>`);
      continue;
    }
    const item = line.match(/^( *)- +(.*)$/);
    if (item) {
      flush();
      const depth = item[1].length >= 2 ? 2 : 1;
      while (list < depth) { out.push("<ul>"); list++; }
      while (list > depth) { out.push("</ul>"); list--; }
      out.push(`<li>${inline(item[2])}</li>`);
      continue;
    }
    if (line.startsWith(">")) { flushPara(); quote.push(line.replace(/^>\s?/, "")); continue; }
    if (!line.trim()) { flush(); closeList(); continue; }
    flushQuote();
    para.push(line.trim());   // the report wraps its prose; the browser rewraps it
  }
  flush(); closeList();
  return out.join("\n");
}

function typeset(root) {
  for (const el of root.querySelectorAll(".math")) {
    if (!window.katex) { el.classList.add("plain"); continue; }  // offline: show the source
    katex.render(el.textContent, el, {displayMode: true, throwOnError: false});
  }
}

// -------------------------------------------------------------------- run

const post = async (path, body) => (await fetch(path, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify(body),
})).json();

function line(cls, mark, html) {
  const li = document.createElement("li");
  li.className = cls;
  li.innerHTML = `<span class="mark">${mark}</span>${html}`;
  $("progress").append(li);
}

function running(on) {
  $("go").disabled = on;
  $("go").textContent = on ? "Running…" : "Run";
}

function handle(event, stream) {
  if (event.kind === "note") line("note", "…", esc(event.text));
  else if (event.kind === "step") {
    const done = event.step;
    const bad = done.status === "failed" || done.status === "error";
    line(bad ? "bad" : "ok", MARKS[done.status] || "·",
         `<b>${esc(done.step)}</b> — ${esc(done.detail.split("\n")[0])}`);
  } else if (event.kind === "ask") {
    $("ask").hidden = false;
    $("askText").textContent = event.question.question;
    $("askList").textContent = (event.question.unresolved || []).join(", ");
  } else if (event.kind === "report") {
    $("out").hidden = false;
    $("report").innerHTML = render(event.markdown);
    typeset($("report"));
    $("out").scrollIntoView({behavior: "smooth", block: "start"});
  } else if (event.kind === "error") {
    line("bad", "✗", esc(event.text));
  } else if (event.kind === "done") {
    stream.close();
    running(false);
  }
}

async function run() {
  const input = $("input").value.trim();
  if (!input) return;
  remember();
  $("progress").innerHTML = "";
  $("out").hidden = true;
  $("ask").hidden = true;
  $("saved").textContent = "";
  running(true);

  const {job: started} = await post("/api/run", {
    input, settings: collect(SETTINGS), options: collect(OPTIONS),
  });
  job = started;
  const stream = new EventSource(`/api/events?job=${job}`);
  stream.onmessage = (e) => handle(JSON.parse(e.data), stream);
  stream.onerror = () => { stream.close(); running(false); };
}

async function upload(file) {
  if (!file) return;
  const res = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`,
                          {method: "POST", body: file});
  const {path} = await res.json();
  $("input").value = path;      // the agent opens figures by path, as the CLI does
  $("kind").value = "figure";
  $("dropnote").textContent = path;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
}

// ------------------------------------------------------------------ wiring

async function main() {
  server = await (await fetch("/api/settings")).json();
  $("provider").innerHTML = server.providers.map((p) => `<option>${p}</option>`).join("");
  restore(server);                                            // what the process uses now
  restore(JSON.parse(localStorage.getItem(KEPT) || "{}"));     // what this browser last chose
  const fallback = server.defaults[$("provider").value] || {};
  $("figurehint").hidden = fallback.images !== false;
  keyHint();

  for (const name of TOOLS) $(name).addEventListener("change", checkTools);
  checkTools();

  $("go").onclick = run;
  $("input").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  $("provider").addEventListener("change", pickProvider);
  $("api_key").addEventListener("input", keyHint);

  const zone = $("drop");
  for (const name of ["dragenter", "dragover"]) {
    zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.add("over"); });
  }
  for (const name of ["dragleave", "drop"]) {
    zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.remove("over"); });
  }
  zone.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
  zone.addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => upload(e.target.files[0]));

  $("search").onclick = () => answer("search");
  $("stop").onclick = () => answer("stop");
  $("save").onclick = async () => {
    const {path} = await post("/api/save", {job, path: $("savepath").value.trim()});
    $("saved").textContent = `written to ${path}`;
  };
}

async function answer(reply) {
  $("ask").hidden = true;
  await post("/api/answer", {job, reply});
}

main();
