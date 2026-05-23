// FIFA alpha-3 → ISO alpha-2 for flagcdn.com
const MAP = {
  ESP: "es", ARG: "ar", FRA: "fr", MAR: "ma", BRA: "br", SEN: "sn",
  JPN: "jp", ECU: "ec", COL: "co", NED: "nl", POR: "pt", GER: "de",
  ALG: "dz", IRN: "ir", ENG: "gb-eng", MEX: "mx", CRO: "hr", KOR: "kr",
  CAN: "ca", AUS: "au", URU: "uy", CIV: "ci", TUR: "tr", SUI: "ch",
  EGY: "eg", BEL: "be", UZB: "uz", AUT: "at", USA: "us", PAR: "py",
  PAN: "pa", CZE: "cz", HAI: "ht", SWE: "se", TUN: "tn", CPV: "cv",
  KSA: "sa", IRQ: "iq", NOR: "no", JOR: "jo", COD: "cd", RSA: "za",
  BIH: "ba", QAT: "qa", SCO: "gb-sct", CUR: "cw", NZL: "nz", GHA: "gh",
};

export function flagUrl(fifaCode, size = 40) {
  const alpha2 = MAP[fifaCode];
  if (!alpha2) return null;
  return `https://flagcdn.com/w${size}/${alpha2}.png`;
}
