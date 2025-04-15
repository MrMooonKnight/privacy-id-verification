import React, { createContext, useState, useEffect, useContext } from 'react';

// Authentication context
const AuthContext = createContext();

// Custom hook to use auth context
export const useAuth = () => useContext(AuthContext);

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(localStorage.getItem('token') || null);

  // Initialize authentication state on component mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Check if token exists
        if (token) {
          // Optional: Validate token with backend
          // For now, we'll assume the token is valid
          const storedUser = JSON.parse(localStorage.getItem('user') || '{}');
          setUser(storedUser);
          setIsAuthenticated(true);
        }
      } catch (error) {
        console.error('Authentication error:', error);
        logout();
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, [token]);

  // Login function
  const login = (token, user_id) => {
    const userData = { userId: user_id };
    localStorage.setItem('token', token);
    localStorage.setItem('user', JSON.stringify(userData));
    setToken(token);
    setUser(userData);
    setIsAuthenticated(true);
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  // Update user data
  const updateUser = (userData) => {
    localStorage.setItem('user', JSON.stringify(userData));
    setUser(userData);
  };

  // Context value
  const value = {
    isAuthenticated,
    user,
    userInfo: user,
    loading,
    token,
    login,
    logout,
    updateUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext; 