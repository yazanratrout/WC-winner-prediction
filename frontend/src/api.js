import axios from "axios";

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export const api = axios.create({ baseURL: BASE });

export const getTeams = () => api.get("/teams").then(r => r.data);
export const getTeam = name => api.get(`/team/${encodeURIComponent(name)}`).then(r => r.data);
export const getGroup = letter => api.get(`/group/${letter}`).then(r => r.data);
export const predictMatch = (home_team, away_team, neutral = true) =>
  api.post("/match/predict", { home_team, away_team, neutral }).then(r => r.data);
export const getHistory = (team, stage = "winner") =>
  api.get(`/history/${encodeURIComponent(team)}`, { params: { stage } }).then(r => r.data);
