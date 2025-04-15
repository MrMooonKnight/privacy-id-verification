import React from 'react';
import { Container, Typography, Paper, Box } from '@mui/material';

const ProfilePage = () => {
  return (
    <Container component="main" maxWidth="md" sx={{ my: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Profile Page
        </Typography>
        <Typography variant="body1">
          This is a placeholder for the user profile page. Here users will be able to view and manage their profile information.
        </Typography>
        <Box sx={{ bgcolor: '#f5f5f5', p: 2, mt: 2, borderRadius: 1 }}>
          <Typography variant="body2" color="textSecondary">
            Coming soon: Profile management functionality
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default ProfilePage; 