import { useState, useEffect } from "react";
import { getTeams, predictMatch } from "../api";
import Flag from "../components/Flag";
import styles from "./MatchPredictor.module.css";

function ProbBar({ label, value, color, flip = false }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(value), 80);
    return () => clearTimeout(t);
  }, [value]);

  return (
    <div className={styles.probBarRow} style={{ flexDirection: flip ? "row-reverse" : "row" }}>
      <span className={styles.probLabel} style={{ color, textAlign: flip ? "right" : "left" }}>{label}</span>
      <div className={styles.probTrack}>
        <div
          className={styles.probFill}
          style={{ width: `${w}%`, background: color, marginLeft: flip ? "auto" : 0 }}
        />
      </div>
      <span className={styles.probPct} style={{ color }}>{value?.toFixed(1)}%</span>
    </div>
  );
}

export default function MatchPredictor() {
  const [teams, setTeams] = useState([]);
  const [teamMap, setTeamMap] = useState({});
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [neutral, setNeutral] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getTeams().then(data => {
      setTeams(data);
      const m = {};
      data.forEach(t => { m[t.team] = t; });
      setTeamMap(m);
    });
  }, []);

  const teamNames = [...teams].sort((a, b) => a.team.localeCompare(b.team)).map(t => t.team);

  function predict() {
    if (!home || !away || home === away) return;
    setLoading(true);
    predictMatch(home, away, neutral)
      .then(data => { setResult(data); setLoading(false); })
      .catch(() => setLoading(false));
  }

  const homeTeam = teamMap[home];
  const awayTeam = teamMap[away];

  return (
    <div className={styles.page}>
      <h1 className={styles.title}>Match Predictor</h1>
      <p className={styles.sub}>Select any two WC 2026 teams to predict the match outcome</p>

      {/* Selector card */}
      <div className={styles.selectorCard}>
        <div className={styles.teamPicker}>
          <div className={styles.pickerSide}>
            {homeTeam && <Flag code={homeTeam.fifa_code} size={48} />}
            <select className={styles.select} value={home} onChange={e => { setHome(e.target.value); setResult(null); }}>
              <option value="">— Team A —</option>
              {teamNames.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>

          <div className={styles.vsCircle}>VS</div>

          <div className={styles.pickerSide}>
            {awayTeam && <Flag code={awayTeam.fifa_code} size={48} />}
            <select className={styles.select} value={away} onChange={e => { setAway(e.target.value); setResult(null); }}>
              <option value="">— Team B —</option>
              {teamNames.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
        </div>

        <div className={styles.footer}>
          <label className={styles.neutralLabel}>
            <input type="checkbox" checked={neutral} onChange={e => setNeutral(e.target.checked)} />
            Neutral venue
          </label>
          <button
            className={styles.predictBtn}
            onClick={predict}
            disabled={!home || !away || home === away || loading}
          >
            {loading ? <span className={styles.btnSpinner} /> : null}
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>
      </div>

      {/* Result */}
      {result && (
        <div className={styles.resultCard}>
          {/* H2H header */}
          <div className={styles.h2hHeader}>
            <div className={styles.h2hTeam}>
              <Flag code={homeTeam?.fifa_code} size={44} />
              <span className={styles.h2hName}>{result.home_team}</span>
            </div>

            <div className={styles.h2hCenter}>
              <span className={styles.h2hVs}>vs</span>
            </div>

            <div className={styles.h2hTeam}>
              <Flag code={awayTeam?.fifa_code} size={44} />
              <span className={styles.h2hName}>{result.away_team}</span>
            </div>
          </div>

          {/* Probability bars */}
          <div className={styles.probSection}>
            <ProbBar label={`${result.home_team} win`} value={result.probabilities.home_win} color="var(--green)" />
            <ProbBar label="Draw" value={result.probabilities.draw} color="var(--yellow)" />
            <ProbBar label={`${result.away_team} win`} value={result.probabilities.away_win} color="var(--red)" flip />
          </div>

          {/* Big numbers */}
          <div className={styles.bigNums}>
            <div className={styles.bigNum} style={{ color: "var(--green)" }}>
              <span>{result.probabilities.home_win?.toFixed(1)}%</span>
              <span className={styles.bigLabel}>{result.home_team} wins</span>
            </div>
            <div className={styles.bigNum} style={{ color: "var(--yellow)" }}>
              <span>{result.probabilities.draw?.toFixed(1)}%</span>
              <span className={styles.bigLabel}>Draw</span>
            </div>
            <div className={styles.bigNum} style={{ color: "var(--red)" }}>
              <span>{result.probabilities.away_win?.toFixed(1)}%</span>
              <span className={styles.bigLabel}>{result.away_team} wins</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
