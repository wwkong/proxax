/* Lightweight, dependency-free interactive sorting for all Sphinx doc tables. */
(function () {
  function parseSortValue(text) {
    var cleaned = text.trim().replace(/,/g, "");
    if (cleaned === "" || cleaned === "-" || cleaned === "—") {
      return { isNum: false, num: 0, str: "" };
    }
    var fracMatch = cleaned.match(/^([-+]?\d*\.?\d+)\s*\/\s*([-+]?\d*\.?\d+)$/);
    if (fracMatch && parseFloat(fracMatch[2]) !== 0) {
      return {
        isNum: true,
        num: parseFloat(fracMatch[1]) / parseFloat(fracMatch[2]),
        str: cleaned.toLowerCase(),
      };
    }
    var stripped = cleaned.replace(/\s*(?:ms|s|×|x|%)$/i, "").trim();
    var numMatch = stripped.match(/^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$/);
    if (numMatch) {
      return { isNum: true, num: parseFloat(stripped), str: cleaned.toLowerCase() };
    }
    return { isNum: false, num: 0, str: cleaned.toLowerCase() };
  }

  function initSortableTables() {
    document.querySelectorAll("table.table, table.docutils").forEach(function (table) {
      if (table.getAttribute("data-px-sortable") === "1") return;
      var thead = table.querySelector("thead");
      var tbody = table.querySelector("tbody");
      if (!thead || !tbody) return;
      table.setAttribute("data-px-sortable", "1");

      var headers = Array.from(thead.querySelectorAll("th"));
      headers.forEach(function (th, colIdx) {
        th.classList.add("px-sortable-th");
        th.title = "Click to sort";
        var indicator = document.createElement("span");
        indicator.className = "px-sort-indicator";
        indicator.textContent = " ⇅";
        th.appendChild(indicator);

        th.addEventListener("click", function () {
          var currentDir = th.getAttribute("data-sort-dir");
          var nextDir = currentDir === "asc" ? "desc" : "asc";

          headers.forEach(function (other) {
            other.removeAttribute("data-sort-dir");
            var ind = other.querySelector(".px-sort-indicator");
            if (ind) ind.textContent = " ⇅";
          });

          th.setAttribute("data-sort-dir", nextDir);
          indicator.textContent = nextDir === "asc" ? " ▲" : " ▼";

          var rows = Array.from(tbody.querySelectorAll("tr"));
          var allNumeric = rows.every(function (row) {
            var cell = row.children[colIdx];
            if (!cell) return true;
            var val = parseSortValue(cell.textContent || "");
            return val.isNum || val.str === "";
          });

          rows.sort(function (a, b) {
            var aText = a.children[colIdx] ? a.children[colIdx].textContent || "" : "";
            var bText = b.children[colIdx] ? b.children[colIdx].textContent || "" : "";
            var va = parseSortValue(aText);
            var vb = parseSortValue(bText);

            var cmp = 0;
            if (allNumeric && va.isNum && vb.isNum) {
              cmp = va.num - vb.num;
            } else {
              cmp = va.str.localeCompare(vb.str);
            }
            return nextDir === "asc" ? cmp : -cmp;
          });

          rows.forEach(function (row) {
            tbody.appendChild(row);
          });
        });
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSortableTables);
  } else {
    initSortableTables();
  }
})();
