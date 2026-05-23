import { useEffect, useRef } from "react";

export default function useAutoRefresh(fn, intervalMs = 60_000) {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    const id = setInterval(() => fnRef.current(), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
}
