/**
 * Fix left (primary) and right (secondary) sidebar toggle buttons on wide screens.
 * sphinx-book-theme 1.4.0 only binds querySelector(".primary-toggle") (missing
 * the article-header button) and does not handle .secondary-toggle on desktop.
 */
(function () {
  function initSidebarToggles() {
    const primarySidebar = document.querySelector("#pst-primary-sidebar");
    const secondarySidebar = document.querySelector("#pst-secondary-sidebar");

    if (primarySidebar) {
      document.querySelectorAll(".primary-toggle").forEach((btn) => {
        btn.addEventListener(
          "click",
          (event) => {
            if (window.matchMedia("(min-width: 992px)").matches) {
              event.preventDefault();
              event.stopImmediatePropagation();
              primarySidebar.classList.toggle("pst-sidebar-hidden");
              btn.blur();
            }
          },
          true
        );
      });
    }

    if (secondarySidebar) {
      document.querySelectorAll(".secondary-toggle").forEach((btn) => {
        btn.addEventListener(
          "click",
          (event) => {
            if (window.matchMedia("(min-width: 1200px)").matches) {
              event.preventDefault();
              event.stopImmediatePropagation();
              secondarySidebar.classList.toggle("pst-sidebar-hidden");
              btn.blur();
            }
          },
          true
        );
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSidebarToggles);
  } else {
    initSidebarToggles();
  }
})();
