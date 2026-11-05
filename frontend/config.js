const localHosts = ["localhost", "127.0.0.1"];
window.SALES_API_BASE = localHosts.includes(window.location.hostname)
  ? "http://127.0.0.1:8000"
  : "https://northstar-sales-api.onrender.com";
