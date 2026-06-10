/* Deep PTA — comportamientos compartidos del sitio (sin librerías):
   tema claro/oscuro, idioma ES/EN, TOC autogenerado, carrusel y Mermaid. */

/* ---------- tema ---------- */
(function () {
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle");
  const saved = localStorage.getItem("deep-pta-theme");
  if (saved) root.dataset.theme = saved;
  if (btn) {
    btn.textContent = (saved || "auto").toUpperCase();
    btn.addEventListener("click", () => {
      const next = { "": "dark", dark: "light", light: "" }[root.dataset.theme || ""];
      if (next) { root.dataset.theme = next; localStorage.setItem("deep-pta-theme", next); }
      else { delete root.dataset.theme; localStorage.removeItem("deep-pta-theme"); }
      location.reload(); // re-render mermaid con el tema correcto
    });
  }
})();

/* ---------- idioma (es | en) ---------- */
(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("deep-pta-lang") || "es";
  function apply(lang) {
    root.setAttribute("lang", lang);
    document.querySelectorAll("[data-lang]").forEach((el) => {
      el.style.display = el.dataset.lang === lang ? "" : "none";
    });
    document.querySelectorAll(".lang-toggle button").forEach((b) => {
      b.classList.toggle("active", b.dataset.setLang === lang);
    });
    localStorage.setItem("deep-pta-lang", lang);
  }
  document.querySelectorAll(".lang-toggle button").forEach((b) => {
    b.addEventListener("click", () => apply(b.dataset.setLang));
  });
  apply(saved);
})();

/* ---------- TOC autogenerado + scroll-spy ---------- */
(function () {
  const toc = document.querySelector("#toc ol");
  if (!toc) return;
  const lang = localStorage.getItem("deep-pta-lang") || "es";
  const heads = [...document.querySelectorAll("article h2, article h3")].filter(
    (h) => !h.closest("[data-lang]") || h.closest("[data-lang]").dataset.lang === lang
  );
  const links = new Map();
  heads.forEach((h, i) => {
    if (!h.id)
      h.id = "s" + i + "-" + h.textContent.toLowerCase()
        .normalize("NFD").replace(/[^a-z0-9]+/g, "-").slice(0, 40);
    const li = document.createElement("li");
    li.className = "lvl-" + h.tagName[1];
    const a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    toc.appendChild(li);
    links.set(h.id, a);
  });
  const spy = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.remove("active"));
          links.get(e.target.id)?.classList.add("active");
        }
      });
    },
    { rootMargin: "0px 0px -75% 0px" }
  );
  heads.forEach((h) => spy.observe(h));
})();

/* ---------- carrusel ---------- */
(function () {
  document.querySelectorAll("[data-carousel]").forEach((root) => {
    const track = root.querySelector(".carousel-track");
    const slides = [...track.children];
    const dotsBox = root.querySelector(".dots");
    const dots = slides.map((_, i) => {
      const d = document.createElement("span");
      d.className = "dot" + (i === 0 ? " active" : "");
      d.addEventListener("click", () => go(i));
      dotsBox.appendChild(d);
      return d;
    });
    let idx = 0;
    function go(i) {
      idx = Math.max(0, Math.min(slides.length - 1, i));
      track.scrollTo({ left: slides[idx].offsetLeft - track.offsetLeft, behavior: "smooth" });
    }
    root.querySelector("[data-prev]").addEventListener("click", () => go(idx - 1));
    root.querySelector("[data-next]").addEventListener("click", () => go(idx + 1));
    track.addEventListener(
      "scroll",
      () => {
        const i = Math.round(track.scrollLeft / track.clientWidth);
        if (i !== idx) { idx = i; dots.forEach((d, j) => d.classList.toggle("active", j === i)); }
      },
      { passive: true }
    );
  });
})();

/* ---------- mermaid (tema según modo) ---------- */
(function () {
  if (!document.querySelector(".mermaid")) return;
  const forced = document.documentElement.dataset.theme;
  const dark =
    forced === "dark" || (!forced && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const vars = dark
    ? { primaryColor: "#1d2026", primaryTextColor: "#e8e6e3",
        primaryBorderColor: "#8d8a85", lineColor: "#8d8a85", tertiaryColor: "#14161a" }
    : { primaryColor: "#ffffff", primaryTextColor: "#1a1a1a",
        primaryBorderColor: "#6e6e6e", lineColor: "#6e6e6e", tertiaryColor: "#f7f7f7" };
  import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs").then(({ default: m }) =>
    m.initialize({
      startOnLoad: true,
      theme: "base",
      themeVariables: {
        fontFamily: "Source Sans 3, Helvetica, Arial, sans-serif",
        fontSize: "14px",
        ...vars,
      },
    })
  );
})();
