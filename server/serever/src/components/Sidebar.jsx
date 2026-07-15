import { NavLink } from 'react-router-dom';
import { Activity, Beaker, BookOpen, HardHat } from 'lucide-react';
import './Sidebar.css';

export default function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-header">
        <div className="logo-container">
          <div className="logo-icon">
            <HardHat size={28} weight="fill" />
          </div>
          <h2 className="system-title">水熊虫不动脑</h2>
        </div>
      </div>

      <nav className="nav-menu">
        <NavLink
          to="/"
          className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
          end
        >
          <Activity size={20} />
          <span>巡检监控</span>
        </NavLink>

        <NavLink
          to="/training"
          className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
        >
          <Beaker size={20} />
          <span>模型训练</span>
        </NavLink>

        <NavLink
          to="/knowledge"
          className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
        >
          <BookOpen size={20} />
          <span>专用知识库</span>
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        <div className="status-badge">
          <div className="status-dot"></div>
          <span>System Online</span>
        </div>
      </div>
    </aside>
  );
}
