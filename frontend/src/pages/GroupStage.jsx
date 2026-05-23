import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { getGroup, predictMatch } from "../api";
import Flag from "../components/Flag";
import styles from "./GroupStage.module.css";

const LETTERS = "ABCDEFGHIJKL".split("");

function pct(v) { return v != null ? v.toFixed(0) + "%" : "—"; }

function Heatmap({ teams }) {
  const [matrix, setMatrix] = useState({});

  useEffect(() => {
    if (teams.length < 2) return;
    const pairs = [];
    for (let i = 0; i < teams.length; i++)
      for (let j = i + 1; j < teams.length; j++)
        pairs.push([teams[i].team, teams[j].team]);

    Promise.all(
      pairs.map(([h, a]) =>
        predictMatch(h, a, true).then(r => ({ h, a, probs: r.probabilities })).catch(() => null)
      )
    ).then(results => {
      const m = {};
      results.forEach(r => {
        if (!r) return;
        m[`${r.h}|${r.a}`] = r.probs;
      });
      setMatrix(m);
    });
  }, [teams]);

  if (Object.keys(matrix).length === 0) return <p className={styles.heatmapLoading}>Loading head-to-head…</p>;

  return (
    <div className={styles.heatmap}>
      <table className={styles.heatmapTable}>
        <thead>
          <tr>
            <th />
            {teams.map(t => <th key={t.team}><Flag code={t.fifa_code} size={18} /></th>)}
          </tr>
        </thead>
        <tbody>
          {teams.map((rowTeam, ri) => (
            <tr key={rowTeam.team}>
              <td className={styles.heatmapRowLabel}>
                <Flag code={rowTeam.fifa_code} size={16} />
                <span>{rowTeam.team.split(" ")[0]}</span>
              </td>
              {teams.map((colTeam, ci) => {
                if (ri === ci) return <td key={colTeam.team} className={styles.heatmapDiag} />;
                const key = ri < ci ? `${rowTeam.team}|${colTeam.team}` : `${colTeam.team}|${rowTeam.team}`;
                const probs = matrix[key];
                if (!probs) return <td key={colTeam.team} className={styles.heatmapEmpty}>—</td>;
                const winPct = ri < ci ? probs.home_win : probs.away_win;
                const intensity = Math.round((winPct / 100) * 255);
                const bg = `rgba(79,158,255,${(winPct / 100) * 0.6})`;
                return (
                  <td key={colTeam.team} className={styles.heatmapCell} style={{ background: bg }}>
                    {winPct?.toFixed(0)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className={styles.heatmapNote}>Win probability of row team vs column team</p>
    </div>
  );
}

function GroupCard({ letter }) {
  const [data, setData] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => { getGroup(letter).then(setData).catch(() => {}); }, [letter]);

  if (!data) return (
    <div className={styles.card}>
      <h3 className={styles.groupLabel}>Group {letter}</h3>
      <p className={styles.loading}>Loading…</p>
    </div>
  );

  const sorted = [...data.teams].sort((a, b) => (b.qualify_pct || 0) - (a.qualify_pct || 0));

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <h3 className={styles.groupLabel}>Group {letter}</h3>
        <button className={styles.expandBtn} onClick={() => setExpanded(e => !e)}>
          {expanded ? "Hide H2H ↑" : "Show H2H ↓"}
        </button>
      </div>

      {sorted.map((t, i) => (
        <Link key={t.team} to={`/team/${encodeURIComponent(t.team)}`} className={styles.teamRow}>
          <span className={styles.rowRank}>{i + 1}</span>
          <Flag code={t.fifa_code} size={20} />
          <span className={styles.rowName}>{t.team}</span>
          <div className={styles.rowBars}>
            <div className={styles.miniBarWrap}>
              <div className={styles.miniBar} style={{ width: `${t.qualify_pct || 0}%`, background: "var(--c-qualify)" }} />
            </div>
            <span className={styles.rowPct} style={{ color: "var(--c-qualify)" }}>{pct(t.qualify_pct)}</span>
            <span className={styles.rowWin} style={{ color: "var(--c-winner)" }}>{pct(t.winner_pct)}</span>
          </div>
        </Link>
      ))}

      {expanded && <Heatmap teams={data.teams} />}
    </div>
  );
}

export default function GroupStage() {
  return (
    <div>
      <h1 className={styles.title}>Group Stage</h1>
      <p className={styles.sub}>Qualify percentage. Click "Show H2H" to see head-to-head win probabilities.</p>
      <div className={styles.grid}>
        {LETTERS.map(l => <GroupCard key={l} letter={l} />)}
      </div>
    </div>
  );
}
