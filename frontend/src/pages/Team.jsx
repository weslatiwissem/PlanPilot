// Team page: everyone in the database. The admin adds, edits and removes accounts here.
import { useEffect, useState } from "react";
import { api, post } from "../api.js";

const EMPTY = { name: "", email: "", password: "", role: "member", skills: "", capacity_points: 10,
  current_load: 0, trello_username: "", in_team: true };

export default function Team({ user }) {
  const isAdmin = user.role === "admin";
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState(null);      // null = closed, else the account being added/edited
  const [error, setError] = useState("");
  const [confirmId, setConfirmId] = useState(null);

  const load = () => api("/users").then(setUsers).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  function edit(u) {
    setForm({ ...u, skills: u.skills.join(", "), password: "", trello_username: u.trello_username || "" });
    setError("");
  }

  async function save(e) {
    e.preventDefault();
    const body = {
      name: form.name, role: form.role, in_team: form.in_team,
      skills: form.skills.split(",").map((s) => s.trim()).filter(Boolean),
      capacity_points: Number(form.capacity_points) || 1, current_load: Number(form.current_load) || 0,
      trello_username: form.trello_username.trim() || null,
    };
    try {
      if (form.id) {
        if (form.password) body.password = form.password;   // empty = keep the old password
        await api(`/users/${form.id}`, { method: "PUT", body });
      } else {
        await post("/users", { ...body, email: form.email, password: form.password });
      }
      setForm(null);
      load();
    } catch (err) {
      setError(err.message);
    }
  }

  async function remove(id) {
    try {
      await api(`/users/${id}`, { method: "DELETE" });
      setConfirmId(null);
      load();
    } catch (err) {
      setError(err.message);
    }
  }

  const set = (key) => (e) => setForm({ ...form, [key]: e.target.type === "checkbox" ? e.target.checked : e.target.value });

  return (
    <section>
      <div className="page-head">
        <div>
          <h1>Team</h1>
          <p className="muted">Everyone who can log in. Plans use the people marked “works on stories”.</p>
        </div>
        {isAdmin && !form && <button className="primary" onClick={() => { setForm(EMPTY); setError(""); }}>+ Add teammate</button>}
      </div>
      {error && !form && <div className="error">{error}</div>}

      {form && (
        <form className="card form-grid" onSubmit={save}>
          <h3 className="span2">{form.id ? `Edit ${form.name}` : "New teammate"}</h3>
          <label>Name<input value={form.name} onChange={set("name")} required /></label>
          <label>Email<input type="email" value={form.email} onChange={set("email")} required disabled={Boolean(form.id)} /></label>
          <label>{form.id ? "New password (leave empty to keep)" : "First password (min 6)"}
            <input type="password" value={form.password} onChange={set("password")} required={!form.id} minLength={6} />
          </label>
          <label>Role
            <select value={form.role} onChange={set("role")}>
              <option value="member">member</option><option value="lead">lead</option><option value="admin">admin</option>
            </select>
          </label>
          <label className="span2">Skills (comma separated)<input value={form.skills} onChange={set("skills")} placeholder="backend, python, api" /></label>
          <label>Capacity (points / sprint)<input type="number" min="1" value={form.capacity_points} onChange={set("capacity_points")} /></label>
          <label>Current load (points)<input type="number" min="0" value={form.current_load} onChange={set("current_load")} /></label>
          <label>Trello username (optional)<input value={form.trello_username} onChange={set("trello_username")} /></label>
          <label className="check"><input type="checkbox" checked={form.in_team} onChange={set("in_team")} /> Works on stories</label>
          {error && <div className="error span2">{error}</div>}
          <div className="row span2">
            <button type="button" onClick={() => setForm(null)}>Cancel</button>
            <button className="primary">{form.id ? "Save changes" : "Create account"}</button>
          </div>
        </form>
      )}

      <div className="grid">
        {users.map((u) => (
          <div key={u.id} className={`card person ${u.in_team ? "" : "faded"}`}>
            <div className="person-head">
              <div className="avatar">{u.name.slice(0, 1).toUpperCase()}</div>
              <div className="person-id">
                <h3>{u.name}</h3>
                <div className="muted small">{u.email}</div>
                <div className="role">{u.role}{u.in_team ? "" : " · manages only"}</div>
              </div>
            </div>
            <div className="chips">
              {u.skills.length ? u.skills.map((s) => <span key={s} className="chip">{s}</span>) : <span className="muted small">no skills listed</span>}
            </div>
            <div className="facts">
              <span><b>{u.capacity_points}</b> pts / sprint</span>
              <span><b>{u.current_load}</b> already taken</span>
            </div>
            {isAdmin && (
              <div className="row">
                <button onClick={() => edit(u)}>Edit</button>
                {u.id !== user.id && (confirmId === u.id
                  ? <button className="danger" onClick={() => remove(u.id)}>Really delete?</button>
                  : <button className="ghost" onClick={() => setConfirmId(u.id)}>Delete</button>)}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
