import { useState, useEffect, useCallback } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { getTeam, getHistory } from "../api";
import Flag from "../components/Flag";
import useAutoRefresh from "../hooks/useAutoRefresh";
import styles from "./TeamDetail.module.css";

const STAGES = ["qualify", "r16", "qf", "sf", "final", "winner"];
const STAGE_META = {
  qualify: { label: "Qualify",       color: "var(--c-qualify)" },
  r16:     { label: "Round 16",      color: "var(--c-r16)" },
  qf:      { label: "Quarter Final", color: "var(--c-qf)" },
  sf:      { label: "Semi Final",    color: "var(--c-sf)" },
  final:   { label: "Final",         color: "var(--c-final)" },
  winner:  { label: "Win",           color: "var(--c-winner)" },
};

function pct(v) { return v != null ? v.toFixed(1) + "%" : "—"; }

export default function TeamDetail() {
  const { name } = useParams();
  const navigate = useNavigate();
  const [team, setTeam] = useState(null);
  const [teamMap, setTeamMap] = useState({});
  const [history, setHistory] = useState({});
  const [activeStage, setActiveStage] = useState("winner");
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    Promise.all([
      getTeam(name),
      ...STAGES.map(s => getHistory(name, s)),
    ]).then(([teamData, ...hists]) => {
      setTeam(teamData);
      const h = {};
      STAGES.forEach((s, i) => { h[s] = hists[i].history; });
      setHistory(h);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [name]);

  useEffect(() => { load(); }, [load]);
  useAutoRefresh(load, 60_000);

  if (loading) return <div className={styles.spinner}><div className={styles.spinnerDot} /></div>;
  if (!team) return <p className={styles.notFound}>Team not found.</p>;

  const p = team.probabilities || {};
  const chartData = (history[activeStage] || []).map(h => ({
    time: h.timestamp.slice(0, 16).replace("T", " "),
    value: parseFloat((h.probability || 0).toFixed(2)),
  }));
  const { color: activeColor } = STAGE_META[activeStage];

  return (
    <div className={styles.page}>
      <button onClick={() => navigate(-1)} className={styles.back}>← Back</button>

      {/* Hero */}
      <div className={styles.hero}>
        <Flag code={team.fifa_code} size={80} className={styles.heroFlag} />
        <div className={styles.heroInfo}>
          <h1 className={styles.heroName}>{team.team}</h1>
          <div className={styles.heroMeta}>
            <span>Group <strong style={{ color: "var(--accent)" }}>{team.group}</strong></span>
          </div>
          {team.opponents?.length > 0 && (
            <div className={styles.heroOpponents}>
              <span style={{ marginRight: 6 }}>Group opponents:</span>
              {team.opponents.map(o => (
                <Link key={o} to={`/team/${encodeURIComponent(o)}`} className={styles.oppChip}>
                  {o}
                </Link>
              ))}
            </div>
          )}
        </div>
        <div className={styles.heroBigPct}>
          <span style={{ color: "var(--c-winner)" }}>{pct(p.winner_pct)}</span>
          <span className={styles.heroPctLabel}>to win</span>
        </div>
      </div>

      {/* Stage cards */}
      <div className={styles.stageCards}>
        {STAGES.map(s => {
          const { label, color } = STAGE_META[s];
          const val = p[`${s}_pct`];
          return (
            <button
              key={s}
              className={activeStage === s ? styles.stageCardActive : styles.stageCard}
              style={activeStage === s ? { borderColor: color } : {}}
              onClick={() => setActiveStage(s)}
            >
              <span className={styles.stageLabel}>{label}</span>
              <span className={styles.stageVal} style={{ color }}>{pct(val)}</span>
              <div className={styles.stageMiniBar}>
                <div style={{ width: `${val || 0}%`, background: color, height: "100%", borderRadius: 99 }} />
              </div>
            </button>
          );
        })}
      </div>

      {/* Evolution chart */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3>Probability evolution — <span style={{ color: activeColor }}>{STAGE_META[activeStage].label}</span></h3>
        </div>
        {chartData.length > 1 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="time" tick={{ fill: "var(--muted)", fontSize: 10 }} />
              <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} domain={[0, "auto"]} unit="%" />
              <Tooltip
                formatter={v => [v + "%", STAGE_META[activeStage].label]}
                contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border-strong)", borderRadius: 8 }}
              />
              <Line type="monotone" dataKey="value" stroke={activeColor} strokeWidth={2.5} dot={{ fill: activeColor, r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className={styles.noData}>Only one snapshot so far — this chart updates as matches are played.</p>
        )}
      </div>
    </div>
  );
}
