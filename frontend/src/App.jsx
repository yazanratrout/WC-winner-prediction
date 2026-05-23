import { Routes, Route, NavLink } from "react-router-dom";
import Home from "./pages/Home";
import TeamDetail from "./pages/TeamDetail";
import MatchPredictor from "./pages/MatchPredictor";
import GroupStage from "./pages/GroupStage";
import TournamentTree from "./pages/TournamentTree";
import styles from "./App.module.css";

export default function App() {
  return (
    <div className={styles.shell}>
      <nav className={styles.nav}>
        <span className={styles.brand}>WC 2026</span>
        <div className={styles.navLinks}>
          <NavLink to="/" end className={({ isActive }) => isActive ? styles.active : ""}>Overview</NavLink>
          <NavLink to="/groups" className={({ isActive }) => isActive ? styles.active : ""}>Groups</NavLink>
          <NavLink to="/bracket" className={({ isActive }) => isActive ? styles.active : ""}>Bracket</NavLink>
          <NavLink to="/predict" className={({ isActive }) => isActive ? styles.active : ""}>Predict</NavLink>
        </div>
      </nav>
      <main className={styles.main}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/team/:name" element={<TeamDetail />} />
          <Route path="/groups" element={<GroupStage />} />
          <Route path="/bracket" element={<TournamentTree />} />
          <Route path="/predict" element={<MatchPredictor />} />
        </Routes>
      </main>
    </div>
  );
}
