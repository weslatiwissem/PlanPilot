// The shell: login gate + a sidebar (can be hidden) + the three pages.
import { useEffect, useState } from "react";
import { api, getToken, post, setToken } from "./api.js";
import Login from "./pages/Login.jsx";
import Plan from "./pages/Plan.jsx";
import Team from "./pages/Team.jsx";
import Account from "./pages/Account.jsx";

const SIDEBAR_KEY = "planpilot-sidebar";

// simple line icon: a panel with a side column (the sidebar toggle)
const PanelIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <rect x="3" y="4" width="18" height="16" rx="4" /><line x1="9" y1="4" x2="9" y2="20" />
  </svg>
);

export default function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(Boolean(getToken()));
  const [page, setPage] = useState("plan");
  const [open, setOpen] = useState(() => {
    try { return localStorage.getItem(SIDEBAR_KEY) !== "closed"; } catch { return true; }
  });

  // Already logged in on this browser? Ask the backend who we are.
  useEffect(() => {
    if (!getToken()) return;
    api("/auth/me").then(setUser).catch(() => setToken(null)).finally(() => setChecking(false));
  }, []);

  // The api helper fires this when the token is no longer valid.
  useEffect(() => {
    const out = () => setUser(null);
    window.addEventListener("planpilot-logout", out);
    return () => window.removeEventListener("planpilot-logout", out);
  }, []);

  useEffect(() => { try { localStorage.setItem(SIDEBAR_KEY, open ? "open" : "closed"); } catch { /* ignore */ } }, [open]);

  if (checking) return <div className="center muted">Loading…</div>;
  if (!user) return <Login onLogin={(u) => { setUser(u); setPage(u.role === "member" ? "team" : "plan"); }} />;

  const canPlan = user.role === "lead" || user.role === "admin";
  const tabs = [canPlan && ["plan", "Plan a project"], ["team", "Team"], ["account", "Account"]].filter(Boolean);
  const current = !canPlan && page === "plan" ? "team" : page;

  async function logout() {
    await post("/auth/logout").catch(() => {});
    setToken(null);
    setUser(null);
  }

  return (
    <div className={`shell ${open ? "" : "closed"}`}>
      <aside className="sidebar">
        <div className="side-top">
          <div className="brand">PlanPilot</div>
          <button className="icon" title="Hide sidebar" onClick={() => setOpen(false)}><PanelIcon /></button>
        </div>
        <nav>
          {tabs.map(([id, label]) => (
            <button key={id} className={`nav-item ${current === id ? "active" : ""}`} onClick={() => setPage(id)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="side-user">
          <div className="avatar small">{user.name.slice(0, 1).toUpperCase()}</div>
          <div className="grow">
            <div className="side-name">{user.name}</div>
            <div className="role">{user.role}</div>
          </div>
          <button className="link" onClick={logout}>Log out</button>
        </div>
      </aside>
      {open && <div className="scrim" onClick={() => setOpen(false)} />}
      <main>
        {!open && <button className="icon open-side" title="Show sidebar" onClick={() => setOpen(true)}><PanelIcon /></button>}
        {current === "plan" && <Plan user={user} />}
        {current === "team" && <Team user={user} />}
        {current === "account" && <Account user={user} />}
      </main>
    </div>
  );
}
