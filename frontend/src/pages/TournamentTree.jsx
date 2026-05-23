import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { getTeams } from "../api";
import Flag from "../components/Flag";
import AnimatedBar from "../components/AnimatedBar";
import styles from "./TournamentTree.module.css";

const STAGES = [
  { key: "r16_pct", label: "Round 16", color: "var(--c-r16)" },
  { key: "qf_pct", label: "Quarter Final", color: "var(--c-qf)" },
  { key: "sf_pct", label: "Semi Final", color: "var(--c-sf)" },
  { key: "final_pct", label: "Final", color: "var(--c-final)" },
  { key: "winner_pct", label: "Win", color: "var(--c-winner)" },
];

const VIEWS = [
  { label: "Top 8", count: 8 },
  { label: "Top 16", count: 16 },
  { label: "All 48", count: 48 },
];

function pct(v) { return v != null ? v.toFixed(1) + "%" : "—"; }

function TeamCard({ team, rank }) {
  return (
    <Link to={`/team/${encodeURIComponent(team.team)}`} className={styles.teamCard}>
      <div className={styles.cardTop}>
        <span className={styles.cardRank}>{rank}</span>
        <Flag code={team.fifa_code} size={40} />
        <div className={styles.cardInfo}>
          <span className={styles.cardName}>{team.team}</span>
          <span className={styles.cardGroup}>Group {team.group_letter}</span>
        </div>
        <div className={styles.cardWinBlock}>
          <span className={styles.cardWinPct} style={{ color: "var(--c-winner)" }}>{pct(team.winner_pct)}</span>
          <span className={styles.cardWinLabel}>to win</span>
        </div>
      </div>

      <div className={styles.cardDivider} />

      <div className={styles.cardBars}>
        {STAGES.map(s => (
          <div key={s.key} className={styles.cardBarRow}>
            <span className={styles.cardBarLabel} style={{ color: s.color }}>{s.label}</span>
            <div className={styles.cardBarTrack}>
              <AnimatedBar value={team[s.key] || 0} max={100} color={s.color} height={8} />
            </div>
          </div>
        ))}
      </div>
    </Link>
  );
}

export default function TournamentTree() {
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(true);
  const [count, setCount] = useState(8);

  useEffect(() => {
    getTeams().then(data => { setTeams(data); setLoading(false); });
  }, []);

  if (loading) return (
    <div className={styles.loading}>
      <div className={styles.spinner} />
      <span>Loading…</span>
    </div>
  );

  const sorted = [...teams].sort((a, b) => (b.winner_pct || 0) - (a.winner_pct || 0));
  const visible = sorted.slice(0, count);

  return (
    <div>
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.title}>Knockout Probabilities</h1>
          <p className={styles.sub}>Stage reach % from 10,000 simulations, sorted by win probability.</p>
        </div>
        <div className={styles.viewToggle}>
          {VIEWS.map(v => (
            <button
              key={v.count}
              className={count === v.count ? styles.viewBtnActive : styles.viewBtn}
              onClick={() => setCount(v.count)}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.legend}>
        {STAGES.map(s => (
          <span key={s.key} className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: s.color }} />{s.label}
          </span>
        ))}
      </div>

      <div className={styles.visualGrid}>
        {visible.map((t, i) => (
          <TeamCard key={t.team} team={t} rank={i + 1} />
        ))}
      </div>
    </div>
  );
}
