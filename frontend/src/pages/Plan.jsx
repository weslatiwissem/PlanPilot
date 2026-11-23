// Planning: 5 steps. The AI proposes each step, the lead edits it, the next step starts from the lead's version.
//   0 Project  -> chat: the description (+ answers) becomes the backlog     POST /steps/scope
//   1 Backlog  -> lead edits points / descriptions -> sprints are packed      POST /steps/plan-sprints
//   2 Sprints  -> lead moves stories -> people are matched inside them        POST /steps/assign
//   3 Team     -> lead changes people (auto-locked) -> the rest is re-matched POST /steps/assign, /steps/judge
//   4 Approve  -> Trello cards + how much of the AI's plan was kept          POST /steps/approve
// The team comes from the database (Team page). The plan is kept in this browser until "New plan".
import { useEffect, useRef, useState } from "react";
import { api, post } from "../api.js";

const STEPS = ["Project", "Backlog", "Sprints", "Team", "Approve"];
const POINTS = [1, 2, 3, 5, 8, 13];
const SAVE_KEY = "planpilot-plan";
const START = { step: 0, board: "PlanPilot Project", desc: "", draft: "", chat: [], questions: [], answers: {}, qIndex: 0, scope: null,
  tickets: [], result: null, judge: null, modelVersion: "", approved: null };

function loadSaved() {
  try { return { ...START, ...JSON.parse(localStorage.getItem(SAVE_KEY) || "{}") }; } catch { return START; }
}

const sprintsOf = (tickets) => [...new Set(tickets.map((t) => t.sprint))].sort((a, b) => a - b);

// ------------------------------------------------------------------ small pieces
function Bar({ label, pct, tone }) {
  const color = tone || (pct > 100 ? "var(--red)" : pct >= 90 ? "var(--amber)" : "var(--marine)");
  return (
    <div className="bar-wrap">
      <div className="bar-label">{label}</div>
      <div className="bar"><div style={{ width: `${Math.min(Math.max(pct, 0), 100)}%`, background: color }} /></div>
    </div>
  );
}

function Risks({ result }) {
  if (!result) return null;
  const { risks = [], forecasts = [] } = result;
  return (
    <div className="card soft">
      <h3>Risks</h3>
      {forecasts.length > 0 && (
        <div className="forecasts">
          {forecasts.map((f) => (
            <div key={f.sprint} className="stat"><b>{Math.round(f.probability * 100)}%</b><span>Sprint {f.sprint} may slip</span></div>
          ))}
        </div>
      )}
      {risks.length === 0 ? <p className="muted small">No risks found.</p> : (
        <ul className="risks">
          {risks.slice(0, 6).map((r, i) => <li key={i}><b>{r.item}</b> · {r.reason}</li>)}
        </ul>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ the page
export default function Plan({ user }) {
  const [p, setP] = useState(loadSaved);
  const [team, setTeam] = useState([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const update = (changes) => setP((old) => ({ ...old, ...changes }));
  const logRef = useRef(null);

  useEffect(() => { try { localStorage.setItem(SAVE_KEY, JSON.stringify(p)); } catch { /* ignore */ } }, [p]);
  useEffect(() => { api("/users").then((us) => setTeam(us.filter((u) => u.in_team))).catch((e) => setError(e.message)); }, []);

  // what the planner needs about each person (from the database)
  const members = team.map((u) => ({ name: u.name, skills: u.skills, capacity_points: u.capacity_points,
    current_load: u.current_load, trello_username: u.trello_username }));
  const base = () => ({ scope: p.scope, board_name: p.board, team_members: members });

  async function run(label, fn) {
    setBusy(label);
    setError("");
    try { await fn(); } catch (e) { setError(e.message); } finally { setBusy(""); }
  }

  const setTicket = (title, changes) =>
    update({ tickets: p.tickets.map((t) => (t.title === title ? { ...t, ...changes } : t)), judge: null });

  // keep the newest chat message in view
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [p.chat, busy, error]);

  // ---------------- step actions
  // The chat, one question at a time:
  //   1. the lead describes the project      -> the assistant checks what is missing (dots while it thinks)
  //   2. it asks the questions ONE BY ONE     -> each reply is kept in `answers`
  //   3. after the last reply (or "skip")     -> the backlog is built ("Creating the backlog" + dots)
  // Everything is saved in the browser, so a page reload keeps the conversation (see "Create the backlog" below).
  function send(skip = false) {
    const text = p.draft.trim();
    if (!skip && !text) return;

    if (p.questions.length === 0) {                       // first message = the description
      const chat = [...p.chat, { from: "you", text }];
      update({ chat, draft: "", desc: text, answers: {}, qIndex: 0 });
      askScope({ project_description: text }, chat, "thinking");
      return;
    }

    let chat = p.chat;
    let answers = p.answers;
    if (skip) {
      chat = [...chat, { from: "you", text: "Skip the rest, make reasonable assumptions." }];
    } else {
      answers = { ...answers, [p.questions[p.qIndex]]: text };
      chat = [...chat, { from: "you", text }];
    }
    const next = p.qIndex + 1;
    if (!skip && next < p.questions.length) {           // ask the next question
      chat = [...chat, { from: "ai", text: p.questions[next] }];
      update({ chat, answers, qIndex: next, draft: "" });
      return;
    }
    update({ chat, answers, qIndex: p.questions.length, draft: "" });
    buildBacklog(answers, chat);
  }

  function buildBacklog(answers = p.answers, chat = p.chat) {
    const clarifications = p.questions.filter((q) => answers[q]).map((q) => ({ question: q, answer: answers[q] }));
    askScope({ project_description: p.desc, clarifications, skip_clarification: clarifications.length === 0 },
      chat, "Creating the backlog");
  }

  function askScope(body, chat, label) {
    run(label, async () => {
      const out = await post("/steps/scope", body);
      if (out.status === "needs_clarification") {
        const qs = out.clarifying_questions;
        update({ questions: qs, qIndex: 0, answers: {}, chat: [...chat,
          { from: "ai", text: `Thanks. I have ${qs.length} quick question${qs.length > 1 ? "s" : ""} before I plan.` },
          { from: "ai", text: qs[0] }] });
      } else {
        update({ scope: out.scope, step: 1, chat: [...chat, { from: "ai", text: "Your backlog is ready." }] });
      }
    });
  }

  const loadFile = (file) => file && file.text().then((text) => {
    try {
      const data = JSON.parse(text);
      update({ scope: data.scope || data, questions: [], step: 1 });
    } catch { setError("That file is not a saved backlog (JSON)."); }
  });

  const proposeSprints = () => run("Packing the stories into sprints…", async () => {
    const scope = { epics: p.scope.epics.map((e) => ({ ...e, title: e.title.trim(),
      stories: e.stories.map((x) => ({ ...x, title: x.title.trim() })) })) };
    update({ scope });
    const out = await post("/steps/plan-sprints", { ...base(), scope });
    update({ result: out, tickets: out.ops_result.tickets, step: 2 });
  });

  const assign = (nextStep = 3) => run("Matching people to stories…", async () => {
    const out = await post("/steps/assign", { ...base(), tickets: p.tickets });
    update({ result: out, tickets: out.ops_result.tickets, modelVersion: out.ops_result.model_version,
      judge: null, step: nextStep });
  });

  const askJudge = () => run("The judge is reviewing the assignments…", async () => {
    update({ judge: await post("/steps/judge", { ...base(), tickets: p.tickets }) });
  });

  const approve = () => run("Writing to Trello…", async () => {
    const out = await post("/steps/approve", { ...base(), tickets: p.tickets, approved_by: user.name,
      model_version: p.modelVersion || null });
    update({ approved: out });
  });

  // backlog edits (by position, so renaming never loses the cursor)
  const setEpics = (epics) => update({ scope: { ...p.scope, epics } });
  const editEpic = (ei, changes) => setEpics(p.scope.epics.map((e, i) => (i === ei ? { ...e, ...changes } : e)));
  const removeEpic = (ei) => setEpics(p.scope.epics.filter((_, i) => i !== ei));
  const addEpic = () => setEpics([...p.scope.epics, { title: "New epic", stories: [{ title: "", description: "", estimate_points: 3 }] }]);
  const editStory = (ei, si, changes) => editEpic(ei, {
    stories: p.scope.epics[ei].stories.map((x, i) => (i === si ? { ...x, ...changes } : x)) });
  const removeStory = (ei, si) => editEpic(ei, { stories: p.scope.epics[ei].stories.filter((_, i) => i !== si) });
  const addStory = (ei) => editEpic(ei, { stories: [...p.scope.epics[ei].stories, { title: "", description: "", estimate_points: 3 }] });

  // ---------------- render
  const sprints = sprintsOf(p.tickets);
  const capacity = p.result?.sprint_plan?.sprints?.[0]?.capacity_points
    || members.reduce((sum, m) => sum + m.capacity_points, 0) || 1;

  return (
    <section>
      <div className="plan-head">
        <ol className="stepper">
          {STEPS.map((name, i) => (
            <li key={name} className={i === p.step ? "now" : i < p.step ? "done" : ""}
                onClick={() => i < p.step && !p.approved && update({ step: i })}>
              <span>{i + 1}</span>{name}
            </li>
          ))}
        </ol>
        <button className="ghost" onClick={() => { setP(START); setError(""); }}>New plan</button>
      </div>

      {p.step > 0 && busy && <div className="busy"><span className="spinner" />{busy}</div>}
      {p.step > 0 && error && <div className="error">{error}</div>}

      {/* ---------------- 0 · project: a chat with the planning assistant */}
      {p.step === 0 && (() => {
        const empty = p.chat.length === 0 && !busy && !error;
        const waiting = p.questions.length > 0 && p.qIndex < p.questions.length;   // a question is open
        const allAnswered = p.questions.length > 0 && p.qIndex >= p.questions.length; // e.g. after a reload or an error
        return (
          <div className={`chat ${empty ? "empty" : "talking"}`}>
            {empty ? (
              <div className="hello">
                <h2>What are we planning today, {user.name.split(" ")[0]}?</h2>
                <p className="muted">Tell me what the project does, who will use it and the features it must have.</p>
              </div>
            ) : (
              <div className="chat-log" ref={logRef}>
                <div className="bubble ai">Tell me what the project does, who will use it and the features it must have.</div>
                {p.chat.map((m, i) => (
                  <div key={i} className={`bubble ${m.from}`}>
                    {m.text}
                    {m.list && <ol>{m.list.map((q) => <li key={q}>{q}</li>)}</ol>}
                  </div>
                ))}
                {busy && (
                  <div className="bubble ai typing">
                    {busy !== "thinking" && <em>{busy}</em>}<span /><span /><span />
                  </div>
                )}
                {error && <div className="bubble ai error-note">{error}</div>}
              </div>
            )}
            <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
              <textarea rows={1} value={p.draft} disabled={Boolean(busy) || allAnswered}
                placeholder={waiting ? "Your answer…" : allAnswered ? "All answers sent" : "Describe your project"}
                onChange={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 220)}px`;   // grows with the text
                  update({ draft: e.target.value });
                }}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
              <button className="send" title="Send" disabled={!p.draft.trim() || !members.length || Boolean(busy)}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"
                     strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5" /><path d="M5 12l7-7 7 7" /></svg>
              </button>
            </form>
            <div className="composer-meta">
              {waiting && !busy
                ? <button className="link" onClick={() => send(true)}>Skip the rest and plan with assumptions</button>
                : allAnswered && !busy
                ? <button className="link" onClick={() => buildBacklog()}>Create the backlog</button>
                : <input className="project-name" value={p.board} onChange={(e) => update({ board: e.target.value })}
                         placeholder="Project / Trello board name" title="Project / Trello board name" />}
              <span className="muted">
                Team: {members.length === 0 ? "nobody yet, add teammates on the Team page" : team.map((u) => u.name).join(", ")}
              </span>
              <label className="file-btn">Load a saved backlog
                <input type="file" accept=".json" onChange={(e) => loadFile(e.target.files[0])} />
              </label>
            </div>
          </div>
        );
      })()}

      {/* ---------------- 1 · backlog: everything is editable (epics, stories, descriptions, points) */}
      {p.step === 1 && p.scope && (() => {
        const titles = p.scope.epics.flatMap((e) => e.stories.map((x) => x.title.trim().toLowerCase()));
        const bad = (t) => !t.trim() || titles.filter((x) => x === t.trim().toLowerCase()).length > 1;
        const stories = p.scope.epics.reduce((n, e) => n + e.stories.length, 0);
        const points = p.scope.epics.reduce((n, e) => n + e.stories.reduce((m, x) => m + (x.estimate_points || 3), 0), 0);
        const problem = titles.some((t) => !t) ? "Every story needs a title."
          : new Set(titles).size < titles.length ? "Two stories have the same title: rename one."
          : p.scope.epics.some((e) => !e.title.trim()) ? "Every epic needs a title."
          : stories === 0 ? "Add at least one story." : "";
        return (
          <>
            {p.scope.epics.map((epic, ei) => (
              <div key={ei} className="card epic">
                <div className="epic-head">
                  <input className={`epic-title ${epic.title.trim() ? "" : "invalid"}`} value={epic.title}
                         placeholder="Epic title" onChange={(e) => editEpic(ei, { title: e.target.value })} />
                  <button className="link quiet-link" onClick={() => removeEpic(ei)}>Remove epic</button>
                </div>
                {epic.stories.map((st, si) => (
                  <div key={si} className="story-row">
                    <div className="grow">
                      <input className={`story-title-input ${bad(st.title) ? "invalid" : ""}`} value={st.title}
                             placeholder="Story title" onChange={(e) => editStory(ei, si, { title: e.target.value })} />
                      <textarea className="quiet" rows={2} value={st.description || ""} placeholder="What it does, and how we know it is done"
                                onChange={(e) => editStory(ei, si, { description: e.target.value })} />
                    </div>
                    <div className="story-side">
                      <select value={st.estimate_points || 3} onChange={(e) => editStory(ei, si, { estimate_points: Number(e.target.value) })}>
                        {POINTS.map((n) => <option key={n} value={n}>{n} pts</option>)}
                      </select>
                      <button className="link quiet-link" onClick={() => removeStory(ei, si)}>Remove</button>
                    </div>
                  </div>
                ))}
                <button className="ghost add" onClick={() => addStory(ei)}>+ Add story</button>
              </div>
            ))}
            <button className="ghost add wide-add" onClick={addEpic}>+ Add epic</button>
            <div className="row end">
              <span className="muted small">{p.scope.epics.length} epics · {stories} stories · {points} points</span>
              {problem && <span className="inline-error">{problem}</span>}
              <button onClick={() => update({ step: 0 })}>← Back</button>
              <button className="primary" disabled={Boolean(busy) || Boolean(problem)} onClick={proposeSprints}>Propose sprints →</button>
            </div>
          </>
        );
      })()}

      {/* ---------------- 2 · sprints */}
      {p.step === 2 && (
        <>
          <div className="card soft">
            {sprints.map((n) => {
              const pts = p.tickets.filter((t) => t.sprint === n).reduce((a, t) => a + t.points, 0);
              const pct = Math.round((100 * pts) / capacity);
              return <Bar key={n} label={`Sprint ${n} · ${pts}/${capacity} pts · ${pct}%`} pct={pct} />;
            })}
          </div>
          <div className="columns">
            {sprints.map((n) => (
              <div key={n} className="column">
                <h3>Sprint {n}</h3>
                {p.tickets.filter((t) => t.sprint === n).map((t) => (
                  <div key={t.title} className="ticket">
                    <div className="story-title">{t.title}</div>
                    <div className="muted small">{t.epic}</div>
                    <div className="ticket-controls">
                      <select value={t.points} onChange={(e) => setTicket(t.title, { points: Number(e.target.value) })}>
                        {POINTS.map((x) => <option key={x} value={x}>{x} pts</option>)}
                      </select>
                      <select value={t.sprint} onChange={(e) => setTicket(t.title, { sprint: Number(e.target.value) })}>
                        {[...sprints, sprints.length ? Math.max(...sprints) + 1 : 1].map((x) => (
                          <option key={x} value={x}>{sprints.includes(x) ? `Sprint ${x}` : `New sprint ${x}`}</option>
                        ))}
                      </select>
                    </div>
                    {t.proposed_sprint && t.proposed_sprint !== t.sprint && <div className="tag">moved by you</div>}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <Risks result={p.result} />
          <div className="row end">
            <button onClick={() => update({ step: 1 })}>← Back</button>
            <button className="primary" disabled={Boolean(busy)} onClick={() => assign(3)}>Assign the team →</button>
          </div>
        </>
      )}

      {/* ---------------- 3 · team */}
      {p.step === 3 && (
        <>
          <p className="muted">Assigned by <b>{p.modelVersion}</b>. Pick someone else and that story is locked:
            <b> Re-assign</b> keeps it and spreads the rest around it.</p>
          <div className="columns">
            {sprints.map((n) => {
              const inSprint = p.tickets.filter((t) => t.sprint === n);
              const total = inSprint.reduce((a, t) => a + t.points, 0) || 1;
              const free = inSprint.filter((t) => !t.assignee).reduce((a, t) => a + t.points, 0);
              return (
                <div key={n} className="column">
                  <h3>Sprint {n} <span className="muted small">· {total} pts</span></h3>
                  <div className="load">
                    {members.map((m) => {
                      const pts = inSprint.filter((t) => t.assignee === m.name).reduce((a, t) => a + t.points, 0);
                      const use = Math.round((100 * (pts + m.current_load)) / m.capacity_points);
                      return <Bar key={m.name} pct={use}
                                  label={`${m.name} · ${pts} pts · ${use}% of capacity · ${Math.round((100 * pts) / total)}% of sprint`} />;
                    })}
                    {free > 0 && <Bar tone="var(--grey)" pct={(100 * free) / total} label={`Unassigned · ${free} pts`} />}
                  </div>
                  {inSprint.map((t) => (
                    <div key={t.title} className="ticket">
                      <div className="story-title">{t.title} <span className="muted small">· {t.points} pts</span></div>
                      <div className="ticket-controls">
                        <select value={t.assignee || ""} onChange={(e) => {
                          const who = e.target.value || null;
                          setTicket(t.title, { assignee: who, locked: Boolean(who), confidence: 1 });
                        }}>
                          <option value="">— nobody —</option>
                          {members.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
                        </select>
                        <button className={`lock ${t.locked ? "on" : ""}`} title="Locked stories keep their person"
                                onClick={() => setTicket(t.title, { locked: !t.locked })}>{t.locked ? "Locked" : "Lock"}</button>
                        <select value={t.sprint} onChange={(e) => setTicket(t.title, { sprint: Number(e.target.value) })}>
                          {sprints.map((x) => <option key={x} value={x}>Sprint {x}</option>)}
                        </select>
                      </div>
                      <div className="muted small">
                        {t.assignee
                          ? <>match {Math.round((t.confidence || 0) * 100)}%</>
                          : (t.confidence || 0) >= 0.15 ? "open: whoever fits is already full"
                            : "open: pick someone"}
                        {t.proposed_assignee && t.proposed_assignee !== t.assignee && <> · AI suggested {t.proposed_assignee}</>}
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          {p.judge && (
            <div className="card soft">
              <h3>AI judge <span className="muted small">· {p.judge.model}</span></h3>
              {p.judge.judged === 0 ? <p className="muted small">Nothing assigned to grade yet.</p> : (
                <>
                  <div className="forecasts">
                    <div className="stat"><b>{p.judge.mean_score.toFixed(2)}</b><span>mean score / 5</span></div>
                    <div className="stat"><b>{Math.round(p.judge.agreement_rate * 100)}%</b><span>judge agrees</span></div>
                    <div className="stat"><b>{Math.round(p.judge.good_rate * 100)}%</b><span>good fits (4–5)</span></div>
                  </div>
                  <ul className="risks">
                    {p.judge.verdicts.filter((v) => !v.agrees).map((v) => (
                      <li key={v.title}><b>{v.title}</b> · {v.assignee} → judge prefers <b>{v.best_member}</b>: {v.reason}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}
          <Risks result={p.result} />
          <div className="row end">
            <button onClick={() => update({ step: 2 })}>← Back</button>
            <button disabled={Boolean(busy)} onClick={() => assign(3)}>Re-assign around locked</button>
            <button disabled={Boolean(busy)} onClick={askJudge}>Ask the AI judge</button>
            <button className="primary" onClick={() => update({ step: 4 })}>Review & approve →</button>
          </div>
        </>
      )}

      {/* ---------------- 4 · approve */}
      {p.step === 4 && !p.approved && (() => {
        const moved = p.tickets.filter((t) => t.proposed_sprint && t.sprint !== t.proposed_sprint).length;
        const changed = p.tickets.filter((t) => t.proposed_assignee && t.assignee !== t.proposed_assignee).length;
        return (
          <>
            <div className="forecasts card">
              <div className="stat"><b>{p.tickets.length}</b><span>stories</span></div>
              <div className="stat"><b>{sprints.length}</b><span>sprints</span></div>
              <div className="stat"><b>{moved}</b><span>moved by you</span></div>
              <div className="stat"><b>{changed}</b><span>re-assigned by you</span></div>
            </div>
            <div className="columns">
              {sprints.map((n) => (
                <div key={n} className="column">
                  <h3>Sprint {n}</h3>
                  {p.tickets.filter((t) => t.sprint === n).map((t) => (
                    <div key={t.title} className="mini-person"><b>{t.title}</b> <span className="muted small">{t.points} pts · {t.assignee || "nobody"}</span></div>
                  ))}
                </div>
              ))}
            </div>
            <div className="row end">
              <button onClick={() => update({ step: 3 })}>← Back</button>
              <button className="primary" disabled={Boolean(busy)} onClick={approve}>Approve & create Trello cards on “{p.board}”</button>
            </div>
          </>
        );
      })()}

      {p.step === 4 && p.approved && (() => {
        const ops = p.approved.ops_result;
        const review = ops.review || {};
        return (
          <div className="card">
            <div className="success big">Plan approved by {user.name} and written to Trello.</div>
            {ops.board_url && <p><a className="primary-link" href={ops.board_url} target="_blank" rel="noreferrer">Open the Trello board</a></p>}
            <div className="forecasts">
              <div className="stat"><b>{review.tickets ?? ops.tickets.length}</b><span>stories</span></div>
              <div className="stat"><b>{Math.round((review.sprint_acceptance_rate ?? 1) * 100)}%</b><span>AI sprints kept</span></div>
              <div className="stat"><b>{Math.round((review.assignee_acceptance_rate ?? 1) * 100)}%</b><span>AI assignees kept</span></div>
            </div>
            <ul className="risks">
              {ops.tickets.map((t) => (
                <li key={t.key}>Sprint {t.sprint} · <b>{t.title}</b> · {t.assignee || "nobody"} ·{" "}
                  {t.url ? <a href={t.url} target="_blank" rel="noreferrer">open card</a> : t.status}</li>
              ))}
            </ul>
            <p className="muted small">Logged with model version: {ops.model_version}</p>
          </div>
        );
      })()}
    </section>
  );
}
