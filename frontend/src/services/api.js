import axios from 'axios';

// Create an axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    // Get token from localStorage
    const token = localStorage.getItem('token');
    
    // If token exists, add it to request headers
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// API functions for authentication
export const authAPI = {
  // Register a new user
  register: async (formData) => {
    return api.post('/register', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Verify user identity
  verify: async (formData) => {
    return api.post('/verify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Get user profile
  getProfile: async () => {
    return api.get('/profile');
  },
};

// API functions for blockchain access control
export const accessAPI = {
  // Grant access to a third party
  grantAccess: async (recipientAddress, expirationTime) => {
    return api.post('/access/grant', {
      recipient_address: recipientAddress,
      expiration_time: expirationTime,
    });
  },
  
  // Revoke access from a third party
  revokeAccess: async (recipientAddress) => {
    return api.post('/access/revoke', {
      recipient_address: recipientAddress,
    });
  },
  
  // Check if a third party has access
  checkAccess: async (recipientAddress) => {
    return api.get('/access/check', {
      params: {
        recipient_address: recipientAddress,
      },
    });
  },
};

// Health check
export const healthCheck = async () => {
  return axios.get(
    `${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/health`
  );
};

export default api; 