"use strict";

/* ---------- helpers ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get: k => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch {} },
};
const LOCALES = { en: "en-IN", hi: "hi-IN", ta: "ta-IN" };
const STATES = ["Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Other"];

const state = {
  user: null, guest: false,
  lang: store.get("lang") || "en", view: null, last: null, authMode: "in",
};
try { state.guest = sessionStorage.getItem("guest") === "1"; } catch {}
if (!STR.brand_sub[state.lang]) state.lang = "en";

const tr = (k, vars) => {
  let s = STR[k]?.[state.lang] ?? STR[k]?.en ?? k;
  if (vars) for (const [a, b] of Object.entries(vars)) s = s.replaceAll(`{${a}}`, b);
  return s;
};

class ApiError extends Error { constructor(m, status) { super(m); this.status = status; } }
async function api(path, { method = "GET", json, form } = {}) {
  const opt = { method, credentials: "same-origin", headers: { "X-Requested-With": "PlantDoc" } };
  if (json !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(json); }
  if (form) opt.body = form;
  let res;
  try { res = await fetch(path, opt); } catch { throw new ApiError(tr("err_net"), 0); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(typeof data.detail === "string" ? data.detail : tr("err_generic"), res.status);
  return data;
}

let toastTimer;
function toast(msg, ok = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (ok ? " ok" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), 4500);
}

function applyI18n(root = document) {
  $$("[data-i18n]", root).forEach(el => (el.textContent = tr(el.dataset.i18n)));
  document.documentElement.lang = state.lang;
  $$(".lang-seg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.lang === state.lang));
}

function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  store.set("theme", t || "auto");
}
applyTheme(store.get("theme") || "auto");

/* ---------- session / routing ---------- */
const VIEWS = ["diagnose", "advisor", "history", "profile", "login"];
const initials = u => ((u.name || u.email).trim().split(/\s+/).map(w => w[0]).slice(0, 2).join("") || "?").toUpperCase();

function setUser(user) {
  state.user = user;
  if (user) {
    state.guest = false;
    try { sessionStorage.removeItem("guest"); } catch {}
    const p = user.profile;
    if (p.language && STR.brand_sub[p.language]) { state.lang = p.language; store.set("lang", p.language); }
    applyTheme(p.theme);
  }
  paintUser();
}

function paintUser() {
  const u = state.user;
  $("#avatar").textContent = u ? initials(u) : "?";
  $("#whoName").textContent = u ? (u.name || u.email).split(" ")[0] : tr("sign_in");
  const hello = $("#hello");
  hello.hidden = !u;
  if (u) hello.textContent = tr("hello", { name: (u.name || u.email.split("@")[0]).split(" ")[0] });
}

function route() {
  let v = (location.hash.replace(/^#\/?/, "") || "diagnose");
  if (!VIEWS.includes(v)) v = "diagnose";
  if (!state.user && !state.guest && v !== "login") { location.replace("#/login"); return; }
  if (state.user && v === "login") { location.replace("#/"); return; }
  show(v);
}

function show(v) {
  state.view = v;
  $("#view-login").hidden = v !== "login";
  $("#shell").hidden = v === "login";
  VIEWS.forEach(n => { const el = $("#view-" + n); if (el && n !== "login") el.hidden = n !== v; });
  $$("[data-route]").forEach(a => a.classList.toggle("on", a.dataset.route === v));
  $("#userBtn").classList.toggle("on", v === "profile");
  window.scrollTo(0, 0);
  if (v === "login") paintAuth();
  if (v === "advisor") loadAdvisor();
  if (v === "history") loadHistory();
  if (v === "profile") renderProfile();
}

async function setLang(l) {
  if (l === state.lang) return;
  state.lang = l; store.set("lang", l);
  applyI18n(); paintUser();
  if (state.user) {
    state.user.profile.language = l;
    api("/api/profile", { method: "PATCH", json: { language: l } }).catch(() => {});
  }
  if (state.view === "login") paintAuth();
  if (state.view === "diagnose" && state.last) analyse(null, true);
  else if (state.view === "advisor") loadAdvisor(true);
  else if (state.view === "history") loadHistory();
  else if (state.view === "profile") renderProfile();
}
document.addEventListener("click", e => {
  const b = e.target.closest(".lang-seg button");
  if (b) setLang(b.dataset.lang);
});

/* ---------- auth ---------- */
function paintAuth() {
  const up = state.authMode === "up";
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.mode === state.authMode));
  $$(".up-only").forEach(el => (el.hidden = !up));
  $("#authTitle").textContent = tr(up ? "auth_up_t" : "auth_in_t");
  $("#authDesc").textContent = tr(up ? "auth_up_d" : "auth_in_d");
  $("#authSubmit").textContent = tr(up ? "btn_signup" : "btn_signin");
  $("#aPass").autocomplete = up ? "new-password" : "current-password";
  $("#pwToggle").textContent = tr($("#aPass").type === "password" ? "show" : "hide");
}
$$(".tabs button").forEach(b => (b.onclick = () => { state.authMode = b.dataset.mode; $("#authErr").hidden = true; paintAuth(); }));
$("#pwToggle").onclick = () => { const i = $("#aPass"); i.type = i.type === "password" ? "text" : "password"; paintAuth(); };
$("#guestBtn").onclick = () => {
  state.guest = true;
  try { sessionStorage.setItem("guest", "1"); } catch {}
  paintUser(); location.hash = "#/";
};
$("#authForm").onsubmit = async e => {
  e.preventDefault();
  const err = $("#authErr"), btn = $("#authSubmit");
  const email = $("#aEmail").value.trim(), password = $("#aPass").value;
  err.hidden = true;
  if (!email || !password) { err.textContent = tr("err_fill"); err.hidden = false; return; }
  btn.disabled = true;
  try {
    const up = state.authMode === "up";
    const { user } = await api(up ? "/api/auth/register" : "/api/auth/login", {
      method: "POST", json: { email, password, name: $("#aName").value, language: state.lang },
    });
    $("#aPass").value = "";
    setUser(user); applyI18n();
    location.hash = "#/";
    route();
  } catch (x) { err.textContent = x.message; err.hidden = false; }
  btn.disabled = false;
};

async function signOut() {
  await api("/api/auth/logout", { method: "POST" }).catch(() => {});
  state.user = null; state.guest = false;
  try { sessionStorage.removeItem("guest"); } catch {}
  resetDiagnose(); paintUser();
  location.hash = "#/login";
}

/* ---------- expert plan widget ---------- */
let widgetSeq = 0;
class PlanWidget {
  constructor(root) { this.root = root; this.base = null; this.mounted = false; this.seq = 0; this.timer = null; }

  async load(base) {
    this.base = base;
    this.root.hidden = false; this.mounted = false;
    this.root.innerHTML = '<div class="card"><div class="skeleton"></div></div>';
    try { const { plan } = await this.fetch(base, {}); this.mount(base, plan); }
    catch (e) { this.root.innerHTML = ""; toast(e.message); }
  }

  fetch(base, ctx) {
    return api("/api/advisor", { method: "POST", json: { ...base, ctx, lang: state.lang } });
  }

  mount(base, plan) {
    this.base = base; this.mounted = true; this.root.hidden = false;
    const c = plan.ctx, id = ++widgetSeq;
    const sel = (key, opts, val) => `<select data-ctx="${key}">${opts.map(([v, l]) => `<option value="${v}"${String(v) === String(val) ? " selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
    const form = plan.kind === "uncertain" ? "" : `
      <details class="plan-form" open><summary>${esc(tr("cond_t"))} <small>${esc(tr("cond_hint"))}</small></summary>
        <div class="cond-grid">
          <label class="field"><span>${esc(tr("f_sev"))}</span>${sel("severity", [["auto", tr("opt_auto")], ["mild", tr("sev_mild")], ["moderate", tr("sev_moderate")], ["severe", tr("sev_severe")]], c.severity)}</label>
          <label class="field"><span>${esc(tr("f_hum"))}</span>${sel("humidity", [["low", tr("hum_low")], ["medium", tr("hum_medium")], ["high", tr("hum_high")]], c.humidity)}</label>
          <label class="field"><span>${esc(tr("f_rain"))}</span>${sel("rain", [["0", tr("no")], ["1", tr("yes")]], c.rain ? "1" : "0")}</label>
          <label class="field"><span>${esc(tr("f_stage"))}</span>${sel("stage", ["seedling", "vegetative", "flowering", "fruiting", "harvest"].map(s => [s, tr("st_" + s)]), c.stage)}</label>
          <label class="field"><span>${esc(tr("f_acres"))}</span><input data-ctx="acres" type="number" min="0.1" max="5000" step="0.1" inputmode="decimal" value="${esc(c.acres)}"></label>
          <label class="check"><input data-ctx="organic_only" type="checkbox"${c.organic_only ? " checked" : ""}><span>${esc(tr("f_org"))}</span></label>
        </div>
      </details>`;
    this.root.innerHTML = `<article class="card plan" data-w="${id}">
      <div class="plan-top" data-top></div><div class="facts" data-facts></div>${form}
      <div class="plan-body" data-body></div></article>`;
    this.root.querySelectorAll("[data-ctx]").forEach(el => {
      el.addEventListener(el.type === "number" ? "input" : "change", () => {
        clearTimeout(this.timer); this.timer = setTimeout(() => this.refresh(), 350);
      });
    });
    this.paint(plan);
  }

  readCtx() {
    const ctx = {};
    this.root.querySelectorAll("[data-ctx]").forEach(el => {
      const k = el.dataset.ctx;
      if (el.type === "checkbox") ctx[k] = el.checked;
      else if (k === "rain") ctx[k] = el.value === "1";
      else if (k === "acres") { const n = parseFloat(el.value); if (n > 0) ctx[k] = n; }
      else ctx[k] = el.value;
    });
    return ctx;
  }

  async refresh() {
    if (!this.mounted) return;
    const seq = ++this.seq, body = $("[data-body]", this.root);
    body.classList.add("busy");
    try {
      const { plan } = await this.fetch(this.base, this.readCtx());
      if (seq === this.seq) this.paint(plan);
    } catch (e) { toast(e.message); }
    body.classList.remove("busy");
  }

  paint(p) {
    const top = $("[data-top]", this.root), facts = $("[data-facts]", this.root), body = $("[data-body]", this.root);
    top.innerHTML = `<div class="row"><span class="urg urg-${p.urgency.level}">${esc(p.urgency.label)}</span>
        <span class="pill">${esc(tr("track_l"))}: ${esc(p.track.label)}</span>${p.pathogen ? `<span class="pill">${esc(p.pathogen)}</span>` : ""}</div>
      <h2>${esc(p.headline)}</h2>
      ${p.kind === "uncertain" && p.candidates?.length ? `<p class="muted small">${esc(tr("cands", { list: p.candidates.join(", ") }))}</p>` : ""}`;

    facts.hidden = !p.facts.length;
    facts.innerHTML = p.facts.map(f => {
      const sub = f.key === "severity" ? tr("src_" + f.source) : f.detail || "";
      return `<div class="fact"><span>${esc(tr("fact_" + f.key))}</span><b class="lv-${esc(f.level || "")}">${esc(f.value)}</b>${sub ? `<small>${esc(sub)}</small>` : ""}</div>`;
    }).join("");

    const e = p.escalate;
    const escHtml = e ? `<div class="esc ${esc(e.level)}"><svg class="ic"><use href="#i-phone"/></svg>
      <div class="msg"><b>${esc(tr("esc_" + e.level))}</b>${esc(e.message)}</div>
      <div class="btns"><a class="btn btn-primary btn-sm" href="tel:${esc(e.phone.replace(/\D/g, ""))}"><svg class="ic"><use href="#i-phone"/></svg>${esc(e.phone_label)} ${esc(e.phone)}</a>
      <a class="btn btn-ghost btn-sm" href="${esc(e.kvk_url)}" target="_blank" rel="noopener">${esc(e.kvk_label)}</a></div></div>` : "";

    const actions = `<div class="sec"><h4>${esc(tr("actions_t"))}</h4><ol class="actions">${p.actions.map(a =>
      `<li><div><b>${esc(a.title)}</b><span class="pr ${esc(a.priority)}">${esc(tr("pr_" + a.priority))}</span><p>${esc(a.detail)}</p></div></li>`).join("")}</ol></div>`;

    let treat = "";
    if (p.kind !== "uncertain") {
      treat = `<div class="sec"><h4>${esc(tr("treat_t"))}</h4>` + (p.treatment.length
        ? `<div class="tbl-wrap"><table><thead><tr><th>${esc(tr("col_product"))}</th><th>${esc(tr("col_dose"))}</th><th>${esc(tr("col_total"))}</th></tr></thead><tbody>${p.treatment.map(t =>
          `<tr><td><i class="kind ${esc(t.kind)}"></i>${esc(t.name)}</td><td>${esc(t.dose)}</td><td><b>${esc(t.total)}</b></td></tr>`).join("")}</tbody></table></div>
          <p class="note">${esc(tr("spray_vol", { n: p.spray_volume.total, p: p.spray_volume.per_acre }))}</p>`
        : `<p class="muted">${esc(tr("no_treat"))}</p>`) + "</div>";
    }

    const cal = p.schedule.length ? `<div class="sec"><h4>${esc(tr("cal_t"))}</h4><ul class="cal">${p.schedule.map(s =>
      `<li><b>${esc(tr("day_n", { n: s.day }))}</b>${esc(s.task)}</li>`).join("")}</ul></div>` : "";

    const prev = p.prevention.length ? `<div class="sec"><h4>${esc(tr("prev_t"))}</h4><ul class="bul">${p.prevention.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>` : "";
    const safe = p.safety.length ? `<div class="sec"><h4>${esc(tr("safe_t"))}</h4><ul class="bul safe">${p.safety.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>` : "";
    const two = (prev || safe) ? `<div class="two">${prev}${safe}</div>` : "";

    const why = p.reasoning.length ? `<details class="why"><summary>${esc(tr("why_t"))}</summary><ul>${p.reasoning.map(r => `<li><code>${esc(r.rule)}</code>${esc(r.why)}</li>`).join("")}</ul></details>` : "";

    body.innerHTML = `${escHtml}${actions}${treat}${cal}${two}${why}
      <div class="plan-foot"><span>${esc(p.disclaimer)}</span><button class="btn btn-ghost btn-sm" data-print><svg class="ic"><use href="#i-print"/></svg>${esc(tr("print"))}</button></div>`;
    $("[data-print]", body).onclick = () => {
      document.body.classList.add("printing-plan");
      window.addEventListener("afterprint", () => document.body.classList.remove("printing-plan"), { once: true });
      window.print();
    };
  }
}

/* ---------- diagnose ---------- */
const dxPlan = new PlanWidget($("#dxPlan"));
let busy = false, stepTimer;

function resetDiagnose() {
  state.last = null;
  $("#dropEmpty").hidden = false; $("#dropPreview").hidden = true;
  $("#resEmpty").hidden = false; $("#steps").hidden = true; $("#resBody").hidden = true; $("#resBadge").hidden = true;
  $("#dxPlan").hidden = true; $("#dxPlan").innerHTML = ""; dxPlan.mounted = false;
}

function setSteps(s) {
  $("#resEmpty").hidden = true; $("#resBody").hidden = true; $("#steps").hidden = false;
  $$("#steps li").forEach(li => (li.className = s[li.dataset.step] || ""));
}
function setTag(kind, text) {
  const t = $("#previewTag");
  t.hidden = !kind; t.className = "tag " + (kind || ""); t.textContent = text || "";
}

/* Downscale on-device first: smaller upload on rural networks. HEIC can't be decoded by most browsers, so send it raw. */
async function shrink(file, max = 1600) {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise(r => c.toBlob(r, "image/jpeg", 0.9));
    return blob ? new File([blob], "leaf.jpg", { type: "image/jpeg" }) : file;
  } catch { return file; }
}

async function analyse(file, relang = false) {
  if (busy || (relang && !state.last)) return;
  busy = true;
  $$("#view-diagnose .btn").forEach(b => (b.disabled = true));
  clearTimeout(stepTimer);
  try {
    const upload = relang ? state.last.upload : await shrink(file);
    if (!relang) {
      $("#dropEmpty").hidden = true; $("#dropPreview").hidden = false;
      const url = URL.createObjectURL(upload);
      $("#previewImg").src = url; $("#camOrig").src = url;
      $("#scanline").hidden = false; setTag("", "");
      $("#dxPlan").hidden = true;
      setSteps({ quality: "run" });
      stepTimer = setTimeout(() => busy && setSteps({ quality: "done", model: "run" }), 700);
    }
    const form = new FormData();
    form.append("file", upload); form.append("lang", state.lang); form.append("save", relang ? "0" : "1");
    if (relang && dxPlan.mounted) form.append("context", JSON.stringify(dxPlan.readCtx()));
    const d = await api("/api/diagnose", { method: "POST", form });
    state.last = { upload };
    renderResult(d);
  } catch (e) {
    toast(e.message);
    if (!relang) { $("#steps").hidden = true; $("#resEmpty").hidden = false; $("#scanline").hidden = true; }
  } finally {
    busy = false; clearTimeout(stepTimer);
    $$("#view-diagnose .btn").forEach(b => (b.disabled = false));
  }
}

function setMeter(id, v) {
  $("#m" + id).textContent = v.toFixed(2);
  $("#m" + id + "Bar").style.width = Math.round(v * 100) + "%";
}

function renderPreds(ul, list) {
  ul.innerHTML = "";
  list.forEach((p, i) => {
    const li = document.createElement("li");
    if (i === 0) li.className = "top";
    const pct = Math.round(p.confidence * 100);
    li.innerHTML = `<div class="row"><span></span><span>${pct < 1 && p.confidence > 0 ? "<1" : pct}%</span></div><small></small><div class="bar"><i></i></div>`;
    $("span", li).textContent = p.label;
    $("small", li).textContent = p.pathogen || "";
    ul.append(li);
    requestAnimationFrame(() => requestAnimationFrame(() => ($("i", li).style.width = pct + "%")));
  });
}

function renderResult(d) {
  $("#scanline").hidden = true;
  $("#steps").hidden = true; $("#resEmpty").hidden = true; $("#resBody").hidden = false;
  const v = $("#verdict"), badge = $("#resBadge");
  const msg = (title, text) => { v.innerHTML = "<b></b><span></span>"; $("b", v).textContent = title; $("span", v).textContent = text || ""; };
  const badgeSet = (cls, text) => { badge.hidden = false; badge.className = "pill " + cls; badge.textContent = text; };
  v.className = "verdict";
  $("#camBlock").hidden = true;
  $("#predBlock").hidden = true;
  $("#dxPlan").hidden = true;
  $("#mTrust").parentElement.hidden = false;
  setMeter("Quality", d.quality.score);
  if (matchMedia("(max-width:820px)").matches) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });

  if (d.status === "REJECTED_QUALITY") {
    v.classList.add("bad"); msg(tr("v_quality"), d.tip); setTag("fail", tr("tag_fail")); badgeSet("red", tr("tag_fail"));
    $("#predBlock").hidden = false; $("#preds").innerHTML = ""; $(".cols-label").hidden = true; $("#mTrust").parentElement.hidden = true;
    return;
  }
  if (d.status === "REJECTED_OOD") {
    v.classList.add("bad"); msg(tr("v_ood"), d.message); setTag("fail", tr("tag_fail")); badgeSet("red", tr("tag_fail"));
    return;
  }
  $("#predBlock").hidden = false; $(".cols-label").hidden = false;
  renderPreds($("#preds"), d.predictions);
  setMeter("Trust", d.trust);

  if (d.status === "UNCERTAIN") {
    v.classList.add("warn"); msg(tr("v_uncertain"), d.reason); setTag("warn", tr("tag_warn")); badgeSet("amber", tr("tag_warn"));
  } else {
    msg(`${tr("v_trusted")}: ${d.predictions[0].label}`, ""); setTag("ok", tr("tag_ok")); badgeSet("", tr("tag_ok"));
    if (d.gradcam) { $("#camMap").src = d.gradcam; $("#camBlock").hidden = false; }
  }
  if (d.plan) {
    const base = d.plan.kind === "uncertain"
      ? { uncertain: true, candidates: d.predictions.slice(0, 2).map(p => p.name), brown_ratio: d.brown_ratio }
      : { disease: d.plan.disease, brown_ratio: d.brown_ratio };
    dxPlan.mount(base, d.plan);
  }
}

const dz = $("#dropzone"), fUp = $("#fileUpload"), fCam = $("#fileCamera");
$("#btnUpload").onclick = () => fUp.click();
$("#btnCamera").onclick = () => fCam.click();
dz.onclick = () => fUp.click();
dz.onkeydown = e => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), fUp.click());
[fUp, fCam].forEach(f => (f.onchange = () => { if (f.files[0]) analyse(f.files[0]); f.value = ""; }));
["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) analyse(f); });

/* ---------- advisor ---------- */
const advPlan = new PlanWidget($("#advPlan"));
async function loadAdvisor(relang = false) {
  const sel = $("#advDisease"), prev = sel.value;
  try {
    const list = await api("/api/diseases?lang=" + state.lang);
    sel.innerHTML = list.map(d => `<option value="${esc(d.name)}">${esc(d.label)}</option>`).join("");
    if (prev) sel.value = prev;
  } catch (e) { toast(e.message); return; }
  if (advPlan.mounted && relang) advPlan.refresh();
  else if (!advPlan.mounted) advPlan.load({ disease: sel.value });
}
$("#advDisease").onchange = e => {
  advPlan.base = { disease: e.target.value };
  advPlan.mounted ? advPlan.refresh() : advPlan.load(advPlan.base);
};

/* ---------- history ---------- */
function gate(root) {
  root.innerHTML = `<div class="card state"><span class="empty-ic"><svg class="ic"><use href="#i-user"/></svg></span><b>${esc(tr("gate_t"))}</b><p class="muted">${esc(tr("gate_d"))}</p>
    <a class="btn btn-primary" href="#/login">${esc(tr("sign_in"))}</a></div>`;
}
const fmtDate = t => new Date(t * 1000).toLocaleDateString(LOCALES[state.lang], { day: "numeric", month: "short", year: "numeric" });

async function loadHistory() {
  const body = $("#histBody"), clear = $("#histClear");
  clear.hidden = true;
  if (!state.user) return gate(body);
  body.innerHTML = '<div class="skeleton"></div>';
  try {
    const { items } = await api("/api/history?lang=" + state.lang);
    if (!items.length) {
      body.innerHTML = `<div class="card state"><span class="empty-ic"><svg class="ic"><use href="#i-history"/></svg></span><b>${esc(tr("hist_empty_t"))}</b><p class="muted">${esc(tr("hist_empty_d"))}</p><a class="btn btn-primary" href="#/">${esc(tr("hist_new"))}</a></div>`;
      return;
    }
    clear.hidden = false;
    body.innerHTML = `<div class="hist-grid">${items.map(it => `
      <button class="card hcard" data-id="${it.id}"><span class="pill ${it.status === "UNCERTAIN" ? "amber" : ""}">${esc(tr(it.status === "UNCERTAIN" ? "tag_warn" : "tag_ok"))}</span>
        <img src="${esc(it.thumb || "")}" alt="" loading="lazy">
        <div class="meta"><b>${esc(it.label || tr("unconfirmed"))}</b><small>${esc(fmtDate(it.created))} · ${Math.round((it.confidence || 0) * 100)}%</small></div></button>`).join("")}</div>`;
    $$(".hcard", body).forEach(c => (c.onclick = () => openHistory(+c.dataset.id)));
  } catch (e) { body.innerHTML = ""; toast(e.message); }
}
$("#histClear").onclick = async () => {
  if (!confirm(tr("hist_clear_q"))) return;
  try { await api("/api/history", { method: "DELETE" }); loadHistory(); } catch (e) { toast(e.message); }
};

async function openHistory(id) {
  const dlg = $("#dlg"), body = $("#dlgBody");
  try {
    const it = await api(`/api/history/${id}?lang=${state.lang}`);
    body.innerHTML = `<div class="dlg-head"><h3>${esc(it.predictions[0]?.label || tr("unconfirmed"))} <small class="muted">· ${esc(fmtDate(it.created))}</small></h3>
        <div><button class="btn btn-ghost btn-sm" id="hDel"><svg class="ic"><use href="#i-trash"/></svg>${esc(tr("delete"))}</button>
        <button class="btn btn-primary btn-sm" id="hClose">${esc(tr("close"))}</button></div></div>
      <div class="card"><div class="grid"><img src="${esc(it.thumb || "")}" alt="" style="width:100%;border-radius:12px"><div><ul class="preds" id="hPreds"></ul></div></div></div><div id="hPlan"></div>`;
    renderPreds($("#hPreds", body), it.predictions);
    const w = new PlanWidget($("#hPlan", body));
    w.mount(it.plan.kind === "uncertain"
      ? { uncertain: true, candidates: it.predictions.slice(0, 2).map(p => p.name), brown_ratio: it.brown_ratio }
      : { disease: it.disease, brown_ratio: it.brown_ratio }, it.plan);
    $("#hClose").onclick = () => dlg.close();
    $("#hDel").onclick = async () => {
      try { await api(`/api/history/${id}`, { method: "DELETE" }); dlg.close(); loadHistory(); } catch (e) { toast(e.message); }
    };
    dlg.showModal();
  } catch (e) { toast(e.message); }
}
$("#dlg").addEventListener("click", e => { if (e.target === $("#dlg")) $("#dlg").close(); });

/* ---------- profile ---------- */
function renderProfile() {
  const root = $("#profBody");
  if (!state.user) return gate(root);
  const u = state.user, p = u.profile;
  const opts = (list, val, label) => list.map(v => `<option value="${esc(v)}"${v === val ? " selected" : ""}>${esc(label ? label(v) : v)}</option>`).join("");
  root.innerHTML = `
  <form id="profForm" class="prof-grid" novalidate>
    <section class="card">
      <div class="me"><span class="avatar">${esc(initials(u))}</span><div><h3>${esc(u.name || u.email)}</h3><p class="muted small">${esc(u.email)} · ${esc(tr("member_since", { d: fmtDate(u.created) }))}</p></div></div>
      <div class="fields-2">
        <label class="field"><span>${esc(tr("f_name"))}</span><input name="name" maxlength="80" value="${esc(u.name)}" autocomplete="name"></label>
        <label class="field"><span>${esc(tr("f_phone"))}</span><input name="phone" type="tel" inputmode="tel" value="${esc(p.phone)}" autocomplete="tel"></label>
      </div>
      <h3 style="margin-top:22px">${esc(tr("sec_pref"))}</h3>
      <div class="field"><span>${esc(tr("f_lang"))}</span><div class="seg seg-wide lang-seg"><button type="button" data-lang="en">English</button><button type="button" data-lang="hi">हिन्दी</button><button type="button" data-lang="ta">தமிழ்</button></div></div>
      <div class="field"><span>${esc(tr("f_theme"))}</span><div class="seg seg-wide" id="themeSeg">${["auto", "light", "dark"].map(t => `<button type="button" data-theme="${t}" aria-pressed="${p.theme === t}">${esc(tr("th_" + t))}</button>`).join("")}</div></div>
      <label class="check" style="margin-top:16px"><input name="organic_only" type="checkbox"${p.organic_only ? " checked" : ""}><span>${esc(tr("pref_org"))}</span></label>
    </section>
    <section class="card">
      <h3>${esc(tr("sec_farm"))}</h3><p class="muted">${esc(tr("sec_farm_d"))}</p>
      <div class="fields-2">
        <label class="field"><span>${esc(tr("f_state"))}</span><select name="state"><option value=""></option>${opts(STATES, p.state)}</select></label>
        <label class="field"><span>${esc(tr("f_district"))}</span><input name="district" maxlength="60" value="${esc(p.district)}"></label>
        <label class="field"><span>${esc(tr("f_acres"))}</span><input name="farm_acres" type="number" min="0.1" max="5000" step="0.1" inputmode="decimal" value="${esc(p.farm_acres)}"></label>
        <label class="field"><span>${esc(tr("f_irrig"))}</span><select name="irrigation">${opts(["", "drip", "flood", "sprinkler", "rainfed"], p.irrigation, v => tr("irr_" + v))}</select></label>
        <label class="field"><span>${esc(tr("f_soil"))}</span><select name="soil">${opts(["", "red", "black", "alluvial", "loam", "sandy", "clay"], p.soil, v => tr("soil_" + v))}</select></label>
      </div>
      <div class="field"><span>${esc(tr("f_crops"))}</span><div class="chips">${["tomato", "potato"].map(c => `<label class="chip"><input type="checkbox" name="crop" value="${c}"${p.crops.includes(c) ? " checked" : ""}>${esc(tr("crop_" + c))}</label>`).join("")}</div></div>
      <div class="save-row"><button class="btn btn-primary" type="submit">${esc(tr("btn_save"))}</button><span class="ok" id="savedMsg" hidden>✓ ${esc(tr("saved"))}</span></div>
    </section>
  </form>
  <div class="prof-grid" style="margin-top:18px">
    <form class="card" id="pwForm" novalidate>
      <h3>${esc(tr("sec_sec"))}</h3>
      <label class="field"><span>${esc(tr("pw_current"))}</span><input name="current" type="password" autocomplete="current-password"></label>
      <label class="field"><span>${esc(tr("pw_new"))}</span><input name="new" type="password" autocomplete="new-password" minlength="8"><small class="muted">${esc(tr("pw_hint"))}</small></label>
      <p class="form-err" id="pwErr" hidden></p>
      <div class="save-row"><button class="btn btn-ghost" type="submit">${esc(tr("btn_changepw"))}</button></div>
    </form>
    <div class="card danger">
      <h3>${esc(tr("sec_danger"))}</h3>
      <div class="save-row"><button class="btn btn-ghost" id="soBtn" type="button"><svg class="ic"><use href="#i-logout"/></svg>${esc(tr("sign_out"))}</button></div>
      <form id="delForm" novalidate>
        <label class="field"><span>${esc(tr("del_q"))}</span><input name="password" type="password" autocomplete="current-password"></label>
        <p class="form-err" id="delErr" hidden></p>
        <div class="save-row"><button class="btn btn-danger" type="submit">${esc(tr("btn_delacct"))}</button></div>
      </form>
    </div>
  </div>`;
  applyI18n(root);

  let theme = p.theme;
  $$("#themeSeg button").forEach(b => (b.onclick = () => {
    theme = b.dataset.theme; applyTheme(theme);
    $$("#themeSeg button").forEach(x => x.setAttribute("aria-pressed", x === b));
  }));
  $("#soBtn").onclick = signOut;

  $("#profForm").onsubmit = async e => {
    e.preventDefault();
    const f = e.target, v = n => f.elements[n].value;
    try {
      const { user } = await api("/api/profile", { method: "PATCH", json: {
        name: v("name"), phone: v("phone"), state: v("state"), district: v("district"),
        farm_acres: v("farm_acres") || 1, irrigation: v("irrigation"), soil: v("soil"),
        crops: $$("[name=crop]:checked", f).map(c => c.value), organic_only: f.elements.organic_only.checked,
        theme, language: state.lang } });
      state.user = user; paintUser();
      $("#savedMsg").hidden = false; setTimeout(() => ($("#savedMsg") && ($("#savedMsg").hidden = true)), 2500);
    } catch (x) { toast(x.message); }
  };
  $("#pwForm").onsubmit = async e => {
    e.preventDefault();
    const f = e.target, err = $("#pwErr"); err.hidden = true;
    try {
      await api("/api/auth/password", { method: "POST", json: { current: f.elements.current.value, new: f.elements.new.value } });
      f.reset(); toast(tr("pw_changed"), true);
    } catch (x) { err.textContent = x.message; err.hidden = false; }
  };
  $("#delForm").onsubmit = async e => {
    e.preventDefault();
    const err = $("#delErr"); err.hidden = true;
    try {
      await api("/api/account/delete", { method: "POST", json: { password: e.target.elements.password.value } });
      state.user = null; resetDiagnose(); paintUser(); location.hash = "#/login";
    } catch (x) { err.textContent = x.message; err.hidden = false; }
  };
}

/* ---------- boot ---------- */
(async function boot() {
  applyI18n();
  try { const { user } = await api("/api/auth/me"); if (user) setUser(user); } catch {}
  applyI18n(); paintUser();
  window.addEventListener("hashchange", route);
  route();
  api("/api/health").then(h => { if (h.demo) toast(tr("demo")); }).catch(() => {});
})();
