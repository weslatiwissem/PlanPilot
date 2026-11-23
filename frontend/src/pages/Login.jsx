// Login page. There is no sign-up: the admin creates every account (Team page).
import { useState } from "react";
import { post, setToken } from "../api.js";

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const { token, user } = await post("/auth/login", { email, password });
      setToken(token);
      onLogin(user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <form className="card login-card" onSubmit={submit}>
        <div className="brand big">PlanPilot</div>
        <p className="muted">AI proposes, the team lead decides.</p>
        <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required /></label>
        {error && <div className="error">{error}</div>}
        <button className="primary wide" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        <p className="muted small">No account? Ask your admin to add you.</p>
      </form>
    </div>
  );
}
