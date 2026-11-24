(function (global) {
  "use strict";

  function retryDelayMilliseconds(response) {
    const value = response.headers.get("Retry-After");
    const text = value ? value.trim() : "";
    if (/^\d+$/.test(text)) return Number(text) * 1000;
    const retryAt = Date.parse(text);
    if (Number.isFinite(retryAt)) return Math.max(0, retryAt - Date.now());
    return 1000;
  }

  global.retryDelayMilliseconds = retryDelayMilliseconds;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { retryDelayMilliseconds };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
