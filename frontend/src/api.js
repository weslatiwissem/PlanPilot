// Every call to the backend goes through here. The login token is kept in localStorage.
const TOKEN_KEY = "planpilot-token";

export function getToken() {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}

export function setToken(token) {
  try { token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
}

// POST/GET/PUT/DELETE /api/<path>. Throws an Error with the backend's message if it fails.
export async function api(path, { method = "GET", body } = {}) {
  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(`/api${path}`, { method, headers, body: body ? JSON.stringify(body) : undefined });
  } catch {
    throw new Error("Cannot reach the backend. Is uvicorn running on port 8765?");
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && path !== "/auth/login") {
    setToken(null);
    window.dispatchEvent(new Event("planpilot-logout")); // App goes back to the login page
  }
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(" · ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export const post = (path, body) => api(path, { method: "POST", body });
