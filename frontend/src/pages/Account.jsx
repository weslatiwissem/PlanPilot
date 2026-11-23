// Account page: who you are + change your own password.
import { useState } from "react";
import { post } from "../api.js";

export default function Account({ user }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [msg, setMsg] = useState(null);

  async function save(e) {
    e.preventDefault();
    try {
      await post("/auth/password", { current_password: current, new_password: next });
      setMsg({ ok: true, text: "Password changed." });
      setCurrent("");
      setNext("");
    } catch (err) {
      setMsg({ ok: false, text: err.message });
    }
  }

  return (
    <section className="narrow">
      <h1>Account</h1>
      <div className="card">
        <h3>{user.name} <span className="role">{user.role}</span></h3>
        <p className="muted">{user.email}</p>
        {user.skills.length > 0 && <div className="chips">{user.skills.map((s) => <span key={s} className="chip">{s}</span>)}</div>}
      </div>
      <form className="card" onSubmit={save}>
        <h3>Change password</h3>
        <label>Current password<input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} required /></label>
        <label>New password (min 6)<input type="password" value={next} onChange={(e) => setNext(e.target.value)} required minLength={6} /></label>
        {msg && <div className={msg.ok ? "success" : "error"}>{msg.text}</div>}
        <button className="primary">Save</button>
      </form>
    </section>
  );
}
