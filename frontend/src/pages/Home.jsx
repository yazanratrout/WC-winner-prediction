import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { getTeams } from "../api";
import Flag from "../components/Flag";
import AnimatedBar from "../components/AnimatedBar";
import useAutoRefresh from "../hooks/useAutoRefresh";
import styles from "./Home.module.css";

const STAGE_KEYS = ["qualify_pct", "r16_pct", "qf_pct", "sf_pct", "final_pct", "winner_pct"];
const STAGE_LABELS = ["Qualify", "Round 16", "Quarter Final", "Semi Final", "Final", "Win"];
const STAGE_COLORS = ["var(--c-qualify)", "var(--c-r16)", "var(--c-qf)", "var(--c-sf)", "var(--c-final)", "var(--c-winner)"];

// cycle: null → "desc" → "asc" → null
const NEXT = { null: "desc", desc: "asc", asc: null };
const ARROW = { desc: " ↓", asc: " ↑", null: "" };

function pct(v) { return v != null ? v.toFixed(1) + "%" : "—"; }

function PodiumCard({ team, rank }) {
  const medals = ["🥇", "🥈", "🥉"];
  const glows = ["rgba(255,215,0,0.25)", "rgba(192,192,192,0.18)", "rgba(205,127,50,0.18)"];
  const borders = ["rgba(255,215,0,0.4)", "rgba(192,192,192,0.3)", "rgba(205,127,50,0.3)"];

  return (
    <Link to={`/team/${encodeURIComponent(team.team)}`} className={styles.podiumCard} style={{ boxShadow: `0 0 40px ${glows[rank]}`, borderColor: borders[rank] }}>
      <span className={styles.medal}>{medals[rank]}</span>
      <Flag code={team.fifa_code} size={52} />
      <span className={styles.podiumName}>{team.team}</span>
      <span className={styles.podiumGroup}>Group {team.group_letter}</span>
      <div className={styles.podiumPct}>
        <span className={styles.podiumBig} style={{ color: "var(--c-winner)" }}>{pct(team.winner_pct)}</span>
        <span className={styles.podiumSub}>to win</span>
      </div>
      <div className={styles.podiumMini}>
        <span style={{ color: "var(--c-sf)" }}>Semi Final {pct(team.sf_pct)}</span>
        <span style={{ color: "var(--c-final)" }}>Final {pct(team.final_pct)}</span>
      </div>
    </Link>
  );
}

export default function Home() {
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState({ key: null, dir: null }); // dir: "asc" | "desc" | null
  const [lastRefresh, setLastRefresh] = useState(null);

  const load = useCallback(() => {
    getTeams().then(data => { setTeams(data); setLastRefresh(new Date()); setLoading(false); });
  }, []);

  useEffect(() => { load(); }, [load]);
  useAutoRefresh(load, 60_000);

  function handleSort(key) {
    setSort(prev => {
      if (prev.key !== key) return { key, dir: "desc" };
      const nextDir = NEXT[prev.dir];
      return nextDir ? { key, dir: nextDir } : { key: null, dir: null };
    });
  }

  const base = teams.filter(t =>
    t.team.toLowerCase().includes(query.toLowerCase()) ||
    (t.group_letter || "").toLowerCase().includes(query.toLowerCase())
  );

  const filtered = sort.key && sort.dir
    ? [...base].sort((a, b) =>
        sort.dir === "desc"
          ? (b[sort.key] || 0) - (a[sort.key] || 0)
          : (a[sort.key] || 0) - (b[sort.key] || 0)
      )
    : base;

  if (loading) return (
    <div className={styles.spinner}>
      <div className={styles.spinnerDot} />
      <span>Loading teams…</span>
    </div>
  );

  const top3 = teams.slice(0, 3);

  return (
    <div>
      {/* Podium */}
      <div className={styles.podiumRow}>
        <PodiumCard team={top3[1]} rank={1} />
        <PodiumCard team={top3[0]} rank={0} />
        <PodiumCard team={top3[2]} rank={2} />
      </div>

      {/* Header + controls */}
      <div className={styles.tableHeader}>
        <h2 className={styles.tableTitle}>All 48 Teams</h2>
        <div className={styles.controls}>
          <input
            className={styles.search}
            placeholder="Search team or group…"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>
        {lastRefresh && <span className={styles.refresh}>↻ {lastRefresh.toLocaleTimeString()}</span>}
      </div>

      {/* Table */}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>Group</th>
              {STAGE_KEYS.map((k, i) => {
                const isActive = sort.key === k;
                const dir = isActive ? sort.dir : null;
                return (
                  <th key={k}>
                    <button
                      className={isActive ? styles.sortBtnActive : styles.sortBtn}
                      style={{ color: isActive ? STAGE_COLORS[i] : undefined }}
                      onClick={() => handleSort(k)}
                      title={`Sort by ${STAGE_LABELS[i]}`}
                    >
                      <span style={{ color: STAGE_COLORS[i] }}>{STAGE_LABELS[i]}</span>
                      <span className={styles.sortArrow}>{ARROW[dir] || " ↕"}</span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => (
              <tr key={t.team} className={i < 3 && !query && !sort.key ? styles.topRow : ""}>
                <td className={styles.rankCell}>{i + 1}</td>
                <td>
                  <Link to={`/team/${encodeURIComponent(t.team)}`} className={styles.teamCell}>
                    <Flag code={t.fifa_code} size={22} />
                    <span>{t.team}</span>
                  </Link>
                </td>
                <td className={styles.groupCell}>{t.group_letter}</td>
                {STAGE_KEYS.slice(0, 5).map((k, ki) => (
                  <td key={k} style={{ color: STAGE_COLORS[ki] }}>{pct(t[k])}</td>
                ))}
                <td>
                  <AnimatedBar value={t.winner_pct || 0} max={teams[0]?.winner_pct || 20} color="var(--c-winner)" height={7} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && <p className={styles.noResults}>No teams found for "{query}"</p>}
      </div>
    </div>
  );
}
