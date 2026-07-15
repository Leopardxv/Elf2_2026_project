import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useEffect, useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Monitor from './pages/Monitor';
import Training from './pages/Training';
import Knowledge from './pages/Knowledge';
import Login from './pages/Login';
import './App.css';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [boardIp, setBoardIp] = useState(() => localStorage.getItem('elf2-board-ip') || '100.120.94.115');

  useEffect(() => {
    localStorage.setItem('elf2-board-ip', boardIp);
  }, [boardIp]);

  if (!isLoggedIn) {
    return <Login onLogin={() => setIsLoggedIn(true)} />;
  }

  return (
    <Router>
      <div className="app-container">
        <div className="app-window fade-in-up">
          <Sidebar />
          <main className="main-content">
            <Header boardIp={boardIp} onBoardIpChange={setBoardIp} />
            <div className="page-content">
              <Routes>
                <Route path="/" element={<Monitor />} />
                <Route path="/training" element={<Training boardIp={boardIp} />} />
                <Route path="/knowledge" element={<Knowledge boardIp={boardIp} />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
