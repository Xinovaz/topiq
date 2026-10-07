// Search over the documentation: the box at the top of every page finds
// units, declarations and methods by name, and then by their summaries. The
// index is `search-index.js`, which sets `window.TOPIQ_SEARCH` to entries of
// [name, unit, kind, address from the root, summary].
(function () {
  "use strict";
  var input = document.getElementById("search");
  var results = document.getElementById("results");
  var content = document.getElementById("content");
  var root = document.body.getAttribute("data-root") || "";
  if (!input || !results || !content) {
    return;
  }

  // lower ranks come first: the name itself, then a name beginning with the
  // query, holding it, its path holding it, and last its summary
  function rank(entry, q) {
    var name = entry[0].toLowerCase();
    var path = (entry[1] ? entry[1] + "::" : "") + name;
    if (name === q || path === q) return 0;
    if (name.indexOf(q) === 0) return 1;
    if (name.indexOf(q) >= 0) return 2;
    if (path.indexOf(q) >= 0) return 3;
    if (entry[4].toLowerCase().indexOf(q) >= 0) return 4;
    return -1;
  }

  function cell(row, text, className) {
    var td = document.createElement("td");
    if (className) td.className = className;
    if (text !== null) td.textContent = text;
    row.appendChild(td);
    return td;
  }

  function show(query) {
    var q = query.trim().toLowerCase();
    if (!q) {
      results.hidden = true;
      content.hidden = false;
      return;
    }
    var found = [];
    (window.TOPIQ_SEARCH || []).forEach(function (entry) {
      var r = rank(entry, q);
      if (r >= 0) found.push([r, entry]);
    });
    found.sort(function (a, b) {
      return a[0] - b[0] || a[1][0].length - b[1][0].length || a[1][0].localeCompare(b[1][0]);
    });

    results.textContent = "";
    var heading = document.createElement("h1");
    heading.className = "title";
    heading.textContent = "Results for “" + query.trim() + "”";
    results.appendChild(heading);
    if (!found.length) {
      var none = document.createElement("p");
      none.textContent = "Nothing documented has that name.";
      results.appendChild(none);
    } else {
      var table = document.createElement("table");
      table.className = "items";
      found.slice(0, 200).forEach(function (f) {
        var entry = f[1];
        var row = document.createElement("tr");
        var link = document.createElement("a");
        link.className = "name";
        link.href = root + entry[3];
        link.textContent = (entry[1] ? entry[1] + "::" : "") + entry[0];
        cell(row, null, "item-name").appendChild(link);
        cell(row, entry[2], "kind");
        cell(row, entry[4], "summary");
        table.appendChild(row);
      });
      results.appendChild(table);
    }
    results.hidden = false;
    content.hidden = true;
  }

  input.addEventListener("input", function () {
    show(input.value);
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "/" && document.activeElement !== input) {
      event.preventDefault();
      input.focus();
    } else if (event.key === "Escape" && document.activeElement === input) {
      input.value = "";
      show("");
    }
  });
  var asked = new URLSearchParams(window.location.search).get("search");
  if (asked) {
    input.value = asked;
    show(asked);
  }
})();
