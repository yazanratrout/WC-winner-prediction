import { flagUrl } from "../flags";
import styles from "./Flag.module.css";

export default function Flag({ code, size = 28, className = "" }) {
  const url = flagUrl(code, size <= 20 ? 20 : 40);
  if (!url) return null;
  return (
    <img
      src={url}
      width={size}
      height={Math.round(size * 0.67)}
      className={`${styles.flag} ${className}`}
      alt={code}
      loading="lazy"
    />
  );
}
