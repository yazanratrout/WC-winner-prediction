import { useEffect, useState } from "react";
import styles from "./AnimatedBar.module.css";

export default function AnimatedBar({ value, max = 100, color, label, showPct = true, height = 8 }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(Math.round((value / max) * 100)), 60);
    return () => clearTimeout(t);
  }, [value, max]);

  return (
    <div className={styles.wrap}>
      <div className={styles.track} style={{ height }}>
        <div
          className={styles.fill}
          style={{ width: `${width}%`, background: color, height }}
        />
      </div>
      {showPct && <span className={styles.label} style={{ color }}>{value?.toFixed(1)}%</span>}
    </div>
  );
}
