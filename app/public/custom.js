(function () {
  const HINT_ID = "export-all-welcome-hint";
  const MODAL_ID = "settings-modal-overlay";
  const TERMS_MODAL_ID = "terms-modal-overlay";
  const LEGAL_FOOTER_ID = "legal-footer";
  const SETTINGS_URL = "/settings/app";
  const ADMIN_URL = "/admin/app";
  // Virtual href for the header link: no server route, custom.js intercepts it.
  const CODE_URL = "/admin/code";
  const CODE_MODAL_ID = "code-modal-overlay";
  const IFRAME_URLS = [SETTINGS_URL, ADMIN_URL];
  let settingsDirty = false;
  let cachedRole = null;
  let rolePromise = null;
  let termsChecked = false;

  function markLoginPage() {
    const onLogin = /^\/login(\b|\/)/.test(window.location.pathname);
    document.body.classList.toggle("on-login-page", onLogin);
  }
  if (document.body) markLoginPage();
  else document.addEventListener("DOMContentLoaded", markLoginPage);
  window.addEventListener("popstate", markLoginPage);

  function fetchRole() {
    if (rolePromise) return rolePromise;
    rolePromise = fetch("/api/me", { credentials: "include" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        cachedRole = (data && data.role) || "user";
        return cachedRole;
      })
      .catch(function () { return "user"; });
    return rolePromise;
  }

  window.addEventListener("message", function (e) {
    if (e && e.data && e.data.type === "settings-saved") {
      settingsDirty = true;
    }
  });

  function ensureSettingsModal() {
    let overlay = document.getElementById(MODAL_ID);
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.id = MODAL_ID;
    overlay.className = "settings-modal-overlay";
    overlay.innerHTML =
      '<div class="settings-modal" role="dialog" aria-modal="true" aria-label="Einstellungen">' +
      '<button type="button" class="settings-modal-close" aria-label="Schließen">×</button>' +
      '<iframe class="settings-modal-iframe" title="Einstellungen" src="about:blank"></iframe>' +
      "</div>";
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeSettingsModal();
    });
    overlay
      .querySelector(".settings-modal-close")
      .addEventListener("click", closeSettingsModal);
    document.body.appendChild(overlay);
    return overlay;
  }

  function detectChainlitTheme() {
    const root = document.documentElement;
    if (root.classList.contains("dark")) return "dark";
    if (root.classList.contains("light")) return "light";
    const bg = window.getComputedStyle(document.body).backgroundColor;
    const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(bg);
    if (m) {
      const luma = 0.2126 * +m[1] + 0.7152 * +m[2] + 0.0722 * +m[3];
      return luma < 128 ? "dark" : "light";
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function openSettingsModal(url) {
    const overlay = ensureSettingsModal();
    const iframe = overlay.querySelector("iframe");
    const base = url || SETTINGS_URL;
    // Cache-bust: without this the browser kept serving a stale admin/settings
    // page after deploys (no Cache-Control on the response, heuristic caching).
    const target = base + "?theme=" + detectChainlitTheme() + "&v=" + Date.now();
    iframe.src = target;
    settingsDirty = false;
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", escSettingsHandler);
  }

  function closeSettingsModal() {
    const overlay = document.getElementById(MODAL_ID);
    if (!overlay) return;
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    document.removeEventListener("keydown", escSettingsHandler);
    const iframe = overlay.querySelector("iframe");
    if (iframe) iframe.src = "about:blank";
    if (settingsDirty) {
      settingsDirty = false;
      window.location.reload();
    }
  }

  function escSettingsHandler(e) {
    if (e.key === "Escape") closeSettingsModal();
  }

  function interceptSettingsLinks() {
    IFRAME_URLS.forEach(function (url) {
      const links = document.querySelectorAll('a[href="' + url + '"]');
      links.forEach(function (link) {
        if (link.dataset.settingsModalBound === "1") return;
        link.dataset.settingsModalBound = "1";
        link.addEventListener("click", function (e) {
          e.preventDefault();
          e.stopPropagation();
          openSettingsModal(url);
        });
      });
    });
  }

  // ---- Workshop-Code modal (admin only) ------------------------------------
  function ensureCodeModal() {
    let overlay = document.getElementById(CODE_MODAL_ID);
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.id = CODE_MODAL_ID;
    overlay.className = "settings-modal-overlay";
    overlay.innerHTML =
      '<div class="code-modal" role="dialog" aria-modal="true" aria-label="Workshop-Code">' +
      '<button type="button" class="settings-modal-close" aria-label="Schließen">×</button>' +
      '<div class="code-modal-title">Workshop-Code</div>' +
      '<div class="code-modal-code">…</div>' +
      '<div class="code-modal-meta"></div>' +
      '<button type="button" class="code-modal-users">Benutzer verwalten</button>' +
      "</div>";
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeCodeModal();
    });
    overlay.querySelector(".settings-modal-close").addEventListener("click", closeCodeModal);
    overlay.querySelector(".code-modal-users").addEventListener("click", function () {
      closeCodeModal();
      openSettingsModal(ADMIN_URL);
    });
    document.body.appendChild(overlay);
    return overlay;
  }

  function escCodeHandler(e) {
    if (e.key === "Escape") closeCodeModal();
  }

  function closeCodeModal() {
    const overlay = document.getElementById(CODE_MODAL_ID);
    if (!overlay) return;
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    document.removeEventListener("keydown", escCodeHandler);
  }

  function openCodeModal() {
    const overlay = ensureCodeModal();
    const codeEl = overlay.querySelector(".code-modal-code");
    const metaEl = overlay.querySelector(".code-modal-meta");
    codeEl.textContent = "…";
    metaEl.textContent = "";
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", escCodeHandler);
    fetch("/api/admin/invite-code", { credentials: "include", cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (info) {
        if (!info.enabled) {
          codeEl.textContent = "Registrierung geschlossen";
          metaEl.textContent = "REGISTRATION_INVITE_SECRET ist nicht gesetzt.";
          return;
        }
        codeEl.textContent = info.code;
        const until = new Date(info.rotates_at);
        metaEl.textContent =
          "Gültig bis " + until.toLocaleString("de-DE", { dateStyle: "short", timeStyle: "short" }) +
          " · der Code von gestern gilt ebenfalls noch.";
      })
      .catch(function (err) {
        codeEl.textContent = "Konnte nicht geladen werden";
        metaEl.textContent = err.message;
      });
  }

  function interceptCodeLinks() {
    document.querySelectorAll('a[href="' + CODE_URL + '"]').forEach(function (link) {
      if (link.dataset.codeModalBound === "1") return;
      link.dataset.codeModalBound = "1";
      link.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        openCodeModal();
      });
    });
  }

  function hideAdminLinksIfNotAdmin() {
    if (cachedRole === "admin") return;
    const links = document.querySelectorAll(
      'a[href="' + ADMIN_URL + '"], a[href="' + CODE_URL + '"]'
    );
    links.forEach(function (link) {
      const container = link.closest("li, button, div, span") || link;
      container.style.display = "none";
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderMarkdown(md) {
    // Minimal markdown → HTML. Supports headings, lists, bold, italic,
    // links, horizontal rules, blank-line paragraphs. Good enough for the
    // HPI Nutzungsvertrag; falls back gracefully for anything else.
    const lines = String(md || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let para = [];
    let list = [];

    function flushPara() {
      if (para.length) {
        const text = para.join(" ").trim();
        if (text) out.push("<p>" + inline(text) + "</p>");
        para = [];
      }
    }
    function flushList() {
      if (list.length) {
        out.push("<ul>" + list.map(function (i) { return "<li>" + inline(i) + "</li>"; }).join("") + "</ul>");
        list = [];
      }
    }
    function inline(s) {
      s = escapeHtml(s);
      s = s.replace(/\\_/g, "_");
      s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)!?]|$)/g, "$1<em>$2</em>");
      s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_m, t, u) {
        const safeUrl = /^(https?:|mailto:|\/)/i.test(u) ? u : "#";
        return '<a href="' + escapeHtml(safeUrl) + '" target="_blank" rel="noopener noreferrer">' + t + "</a>";
      });
      return s;
    }

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];
      const line = raw.replace(/\s+$/, "");
      const h = /^(#{1,6})\s+(.*)$/.exec(line);
      if (h) {
        flushPara(); flushList();
        const lvl = h[1].length;
        out.push("<h" + lvl + ">" + inline(h[2]) + "</h" + lvl + ">");
        continue;
      }
      if (/^\s*[-*]\s+/.test(line)) {
        flushPara();
        list.push(line.replace(/^\s*[-*]\s+/, ""));
        continue;
      }
      if (/^\s*---+\s*$/.test(line)) {
        flushPara(); flushList();
        out.push("<hr/>");
        continue;
      }
      if (!line.trim()) {
        flushPara(); flushList();
        continue;
      }
      flushList();
      para.push(line);
    }
    flushPara(); flushList();
    return out.join("\n");
  }

  function ensureTermsModal() {
    let overlay = document.getElementById(TERMS_MODAL_ID);
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.id = TERMS_MODAL_ID;
    overlay.className = "terms-modal-overlay";
    overlay.innerHTML =
      '<div class="terms-modal" role="dialog" aria-modal="true" aria-label="Nutzungsbedingungen">' +
      '  <div class="terms-modal-logo">' +
      '    <img src="/public/logo_light.png" alt="KI-Servicezentrum Berlin-Brandenburg · gefördert durch BMFTR" />' +
      '  </div>' +
      '  <div class="terms-modal-header">' +
      '    <h2 class="terms-modal-title">Nutzungsvertrag KI-Servicezentrum Berlin-Brandenburg</h2>' +
      '    <div class="terms-modal-langs">' +
      '      <button type="button" class="terms-lang-btn" data-lang="de">Deutsch</button>' +
      '      <button type="button" class="terms-lang-btn" data-lang="en">English</button>' +
      '    </div>' +
      '  </div>' +
      '  <div class="terms-modal-body"></div>' +
      '  <div class="terms-modal-pdf-block">' +
      '    <a class="terms-modal-pdf-btn" target="_blank" rel="noopener noreferrer"></a>' +
      '  </div>' +
      '  <div class="terms-modal-footer">' +
      '    <label class="terms-modal-consent">' +
      '      <input type="checkbox" class="terms-consent-checkbox" />' +
      '      <span class="terms-consent-label"></span>' +
      '    </label>' +
      '    <div class="terms-modal-actions">' +
      '      <button type="button" class="terms-btn-decline"></button>' +
      '      <button type="button" class="terms-btn-accept" disabled></button>' +
      '    </div>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(overlay);
    return overlay;
  }

  function openTermsModal(data) {
    const overlay = ensureTermsModal();
    const body = overlay.querySelector(".terms-modal-body");
    const acceptBtn = overlay.querySelector(".terms-btn-accept");
    const declineBtn = overlay.querySelector(".terms-btn-decline");
    const pdfBtn = overlay.querySelector(".terms-modal-pdf-btn");
    const checkbox = overlay.querySelector(".terms-consent-checkbox");
    const consentLabel = overlay.querySelector(".terms-consent-label");
    const langBtns = overlay.querySelectorAll(".terms-lang-btn");

    let currentLang = (navigator.language || "de").toLowerCase().startsWith("en") ? "en" : "de";

    function applyLang(lang) {
      currentLang = lang;
      langBtns.forEach(function (b) {
        b.classList.toggle("active", b.dataset.lang === lang);
      });
      const md = (data.summary && data.summary[lang]) || "";
      body.innerHTML = renderMarkdown(md);
      body.scrollTop = 0;

      declineBtn.textContent = lang === "en" ? "Decline" : "Ablehnen";
      acceptBtn.textContent = lang === "en" ? "Accept" : "Akzeptieren";
      consentLabel.textContent = lang === "en"
        ? "I have read, understood, and agree to the Terms of Use."
        : "Ich habe die Nutzungsbedingungen gelesen, verstanden und stimme ihnen zu.";
      pdfBtn.textContent = lang === "en" ? "Full Terms of Use (PDF)" : "Vollständige Nutzungsbedingungen als PDF";
      pdfBtn.href = (data.pdf && data.pdf[lang]) || "#";
    }

    function refreshAcceptEnabled() {
      acceptBtn.disabled = !checkbox.checked;
    }

    checkbox.checked = false;
    checkbox.onchange = refreshAcceptEnabled;
    refreshAcceptEnabled();

    langBtns.forEach(function (b) {
      b.onclick = function () { applyLang(b.dataset.lang); };
    });

    declineBtn.onclick = function () {
      window.location.href = "/logout";
    };

    acceptBtn.onclick = function () {
      if (!checkbox.checked) return;
      acceptBtn.disabled = true;
      fetch("/api/terms/accept", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: data.version }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error("accept failed");
          overlay.classList.remove("open");
          document.body.style.overflow = "";
        })
        .catch(function () {
          refreshAcceptEnabled();
          alert(currentLang === "en"
            ? "Could not record acceptance. Please try again."
            : "Die Zustimmung konnte nicht gespeichert werden. Bitte erneut versuchen.");
        });
    };

    applyLang(currentLang);
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function isOnLoginPage() {
    return /^\/login(\b|\/)/.test(window.location.pathname);
  }

  function loadAndMaybeShowTerms() {
    if (termsChecked) return;
    // /api/terms requires auth. Calling it from /login where no auth cookie
    // exists returns 401 — and if we latched termsChecked=true on that, the
    // subsequent post-login client-side route transition would never re-run
    // the check, and the modal would only appear on a full refresh.
    if (isOnLoginPage()) return;
    fetch("/api/terms", { credentials: "include" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data) return;
        // Only latch once we actually got a valid answer — a 401 while
        // session is still settling shouldn't permanently disable the check.
        termsChecked = true;
        if (data.up_to_date) return;
        openTermsModal(data);
      })
      .catch(function () {});
  }

  function isVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden";
  }

  function findComposerAnchor() {
    const input =
      document.querySelector("textarea") ||
      document.querySelector('input[type="text"]');
    if (!input || !isVisible(input)) return null;

    const form = input.closest("form");
    if (form && form.parentNode) return form;
    return input.parentElement;
  }

  function findDisclaimerAnchor() {
    // Chainlit's "LLMs can make mistakes…" (or translated / "Built with Chainlit") — a small
    // caption element near the composer. Scope the search to composer ancestors for speed.
    const composer = findComposerAnchor();
    if (!composer) return null;
    const patterns = [
      /^LLMs can make mistakes/i,
      /Check important info/i,
      /LLMs k(ö|oe)nnen Fehler machen/i,
      /Built with Chainlit/i,
    ];
    let scope = composer.parentNode;
    for (let i = 0; i < 5 && scope; i++, scope = scope.parentNode) {
      const els = scope.querySelectorAll("p, small, span, div");
      for (let j = 0; j < els.length; j++) {
        const el = els[j];
        if (el.children.length > 0) continue;
        const t = (el.textContent || "").trim();
        if (!t || t.length > 200) continue;
        for (let k = 0; k < patterns.length; k++) {
          if (patterns[k].test(t)) return el;
        }
      }
    }
    return null;
  }

  function ensureLegalFooter() {
    let footer = document.getElementById(LEGAL_FOOTER_ID);
    if (!footer) {
      footer = document.createElement("div");
      footer.id = LEGAL_FOOTER_ID;
      footer.className = "legal-footer";
      footer.innerHTML =
        '<a href="https://aisc.hpi.de/portal/cfp/pages/imprint/" target="_blank" rel="noopener noreferrer">Impressum</a>' +
        '<span class="legal-footer-sep" aria-hidden="true"> · </span>' +
        '<a href="/public/privacy.html" target="_blank" rel="noopener noreferrer">Datenschutz</a>' +
        '<span class="legal-footer-sep" aria-hidden="true"> · </span>' +
        '<a href="/public/terms.html" target="_blank" rel="noopener noreferrer">Nutzungsbedingungen</a>';
    }
    const disclaimer = findDisclaimerAnchor();
    if (disclaimer && disclaimer.parentNode) {
      // Place immediately after the Chainlit disclaimer so it sits at the very bottom of the chat column.
      if (disclaimer.nextSibling !== footer) {
        disclaimer.parentNode.insertBefore(footer, disclaimer.nextSibling);
      }
      return;
    }
    // Fallback for /login and any screen without a disclaimer — pin to body.
    if (footer.parentNode !== document.body) {
      document.body.appendChild(footer);
    }
  }

  function ensureHint() {
    // The login page has no composer; without this guard the invite-code
    // input (type="text") gets mistaken for it and the hint lands under the login form.
    if (isOnLoginPage()) return;
    const anchor = findComposerAnchor();
    if (!anchor || !anchor.parentNode) return;

    let hint = document.getElementById(HINT_ID);
    if (!hint) {
      hint = document.createElement("div");
      hint.id = HINT_ID;
      hint.style.margin = "0.55rem 0 0 0";
      hint.style.fontSize = "0.9rem";
      hint.style.lineHeight = "1.35";
      hint.style.color = "inherit";
      hint.style.opacity = "0.9";
      hint.innerHTML = 'Nutze "<code>/export all</code>" um alle Chats zu exportieren.';
    }

    if (anchor.nextSibling !== hint) {
      anchor.parentNode.insertBefore(hint, anchor.nextSibling);
    }
  }

  // ---- Self-registration ("Konto erstellen") -------------------------------
  // Chainlit's login page has no signup UI, so we add one button under the
  // form. It reuses the existing #email / #password inputs, POSTs them to
  // /auth/register, then logs in via Chainlit's own /login endpoint.
  const SIGNUP_ID = "signup-block";
  const CODE_INPUT_ID = "invite-code";

  async function doSignup(button, status) {
    const emailInput = document.getElementById("email");
    const passwordInput = document.getElementById("password");
    const codeInput = document.getElementById(CODE_INPUT_ID);
    if (!emailInput || !passwordInput) return;
    const email = (emailInput.value || "").trim().toLowerCase();
    const password = passwordInput.value || "";
    const inviteCode = codeInput ? (codeInput.value || "").trim() : "";

    function fail(msg) {
      status.style.color = "#dc2626";
      status.textContent = msg;
    }
    if (!email || email.indexOf("@") < 1) return fail("Bitte eine gültige E-Mail-Adresse eingeben.");
    if (password.length < 8) return fail("Das Passwort muss mindestens 8 Zeichen haben.");
    if (!inviteCode) return fail("Bitte den Einladungscode eingeben.");

    button.disabled = true;
    status.style.color = "inherit";
    status.textContent = "Konto wird erstellt …";
    try {
      const res = await fetch("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: email, password: password, invite_code: inviteCode }),
      });
      if (!res.ok) {
        let detail = "Registrierung fehlgeschlagen.";
        try {
          const data = await res.json();
          if (data && data.detail) detail = data.detail;
        } catch (e) { /* keep default */ }
        button.disabled = false;
        if (res.status === 403 && codeInput) {
          // Wrong code — clear it so a retry can't resend the same value.
          codeInput.value = "";
          codeInput.focus();
        }
        if ((res.status === 429 || res.status === 503) && codeInput) {
          codeInput.disabled = true; // not coming back this minute
        }
        return fail(detail);
      }
      // Registered — log straight in with the same credentials.
      status.textContent = "Konto erstellt, Anmeldung läuft …";
      const form = new FormData();
      form.append("username", email);
      form.append("password", password);
      const login = await fetch("/login", { method: "POST", credentials: "include", body: form });
      if (login.ok) {
        window.location.href = "/";
        return;
      }
      status.style.color = "inherit";
      status.textContent = "Konto erstellt. Bitte jetzt anmelden.";
      button.disabled = false;
    } catch (e) {
      button.disabled = false;
      fail("Registrierung fehlgeschlagen. Bitte später erneut versuchen.");
    }
  }

  function ensureSignup() {
    if (!isOnLoginPage()) {
      const stale = document.getElementById(SIGNUP_ID);
      if (stale && stale.parentNode) stale.parentNode.removeChild(stale);
      return;
    }
    if (document.getElementById(SIGNUP_ID)) return;

    const passwordInput = document.getElementById("password");
    if (!passwordInput) return; // password auth disabled — nothing to hook onto
    const form = passwordInput.closest("form");
    const submit = form && form.querySelector('button[type="submit"]');
    if (!submit || !submit.parentNode) return;

    const block = document.createElement("div");
    block.id = SIGNUP_ID;
    block.style.display = "grid";
    block.style.gap = "0.5rem";

    const hint = document.createElement("p");
    hint.textContent = "Noch kein Konto? E-Mail und Passwort oben eingeben, dazu den Einladungscode:";
    hint.style.fontSize = "0.8rem";
    hint.style.opacity = "0.7";
    hint.style.margin = "0";

    const codeLabel = document.createElement("label");
    codeLabel.htmlFor = CODE_INPUT_ID;
    codeLabel.textContent = "Einladungscode";
    codeLabel.style.fontSize = "0.8rem";

    const codeInput = document.createElement("input");
    codeInput.type = "text";
    codeInput.id = CODE_INPUT_ID;
    codeInput.name = CODE_INPUT_ID;
    codeInput.autocomplete = "off";
    codeInput.spellcheck = false;
    // Chainlit renders no such field, so this one is ours — but borrow the
    // password input's classes so it looks native in both themes.
    codeInput.className = passwordInput.className;

    const button = document.createElement("button");
    button.type = "button";
    // Match the look of the OAuth buttons (outline variant) when present,
    // otherwise fall back to the submit button's classes.
    const oauthButton = form.querySelector('button[type="button"][class*="border"]');
    button.className = (oauthButton || submit).className;
    button.textContent = "Konto erstellen";

    const status = document.createElement("p");
    status.style.fontSize = "0.8rem";
    status.style.margin = "0";

    button.addEventListener("click", function () {
      doSignup(button, status);
    });

    // Our block sits inside Chainlit's <form>, so a bare Enter here would
    // submit the login form instead of registering.
    codeInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        doSignup(button, status);
      }
    });

    block.appendChild(hint);
    block.appendChild(codeLabel);
    block.appendChild(codeInput);
    block.appendChild(button);
    block.appendChild(status);
    submit.parentNode.insertBefore(block, submit.nextSibling);
  }

  const observer = new MutationObserver(function () {
    ensureSignup();
    ensureHint();
    ensureLegalFooter();
    interceptSettingsLinks();
    interceptCodeLinks();
    hideAdminLinksIfNotAdmin();
    // Chainlit navigates /login → / via client-side routing with no page
    // reload. The observer fires on the DOM churn that follows the nav,
    // so this is where we retry the terms fetch once we're actually
    // authenticated. Guarded by termsChecked so it fetches at most once.
    loadAndMaybeShowTerms();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  window.addEventListener("load", function () {
    ensureSignup();
    ensureHint();
    ensureLegalFooter();
    interceptSettingsLinks();
    interceptCodeLinks();
    fetchRole().then(hideAdminLinksIfNotAdmin);
    loadAndMaybeShowTerms();
  });
  ensureSignup();
  ensureHint();
  ensureLegalFooter();
  interceptSettingsLinks();
  interceptCodeLinks();
  fetchRole().then(hideAdminLinksIfNotAdmin);
  loadAndMaybeShowTerms();
})();
