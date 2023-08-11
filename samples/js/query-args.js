/*
Provides a simple way to get values from the query string if they're present
and use a default value if not.

Example:
For the URL http://example.com/index.html?particleCount=1000

QueryArgs.getInt("particleCount", 100); // URL overrides, returns 1000
QueryArgs.getInt("particleSize", 10); // Not in URL, returns default of 10
*/

let searchParams = null;
function clearArgsCache() {
  // Force re-parsing on next access
  searchParams = null;
}
window.addEventListener('popstate', clearArgsCache);
window.addEventListener('hashchange', clearArgsCache);

function ensureArgsCached() {
  if (!searchParams) {
    searchParams = new URLSearchParams(window.location.search);
  }
}

export class QueryArgs {
  static getString(name, defaultValue) {
    ensureArgsCached();
    return searchParams.get(name) || defaultValue;
  }

  static getInt(name, defaultValue) {
    ensureArgsCached();
    return searchParams.has(name) ? parseInt(searchParams.has(name), 10) : defaultValue;
  }

  static getFloat(name, defaultValue) {
    ensureArgsCached();
    return searchParams.has(name) ? parseFloat(searchParams.has(name)) : defaultValue;
  }

  static getBool(name, defaultValue) {
    ensureArgsCached();
    return searchParams.has(name) ? parseInt(searchParams.has(name), 10) != 0 : defaultValue;
  }
}
