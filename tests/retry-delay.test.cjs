const test = require("node:test");
const assert = require("node:assert/strict");
const { retryDelayMilliseconds } = require("../static/retry-delay.js");

function responseWithRetryAfter(value) {
  return { headers: { get: (name) => name === "Retry-After" ? value : null } };
}

test("uses the server Retry-After delay in milliseconds", () => {
  assert.equal(retryDelayMilliseconds(responseWithRetryAfter("1")), 1000);
});

test("supports HTTP-date Retry-After values", () => {
  const originalNow = Date.now;
  Date.now = () => 0;
  try {
    assert.equal(retryDelayMilliseconds(responseWithRetryAfter("Thu, 01 Jan 1970 00:00:02 GMT")), 2000);
  } finally {
    Date.now = originalNow;
  }
});

test("falls back to one second when Retry-After is absent or invalid", () => {
  assert.equal(retryDelayMilliseconds(responseWithRetryAfter(null)), 1000);
  assert.equal(retryDelayMilliseconds(responseWithRetryAfter("later")), 1000);
  assert.equal(retryDelayMilliseconds(responseWithRetryAfter("1second")), 1000);
});
