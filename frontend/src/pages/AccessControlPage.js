import React from 'react';
import { Container, Typography, Paper, Box } from '@mui/material';

const AccessControlPage = () => {
  return (
    <Container component="main" maxWidth="md" sx={{ my: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Access Control
        </Typography>
        <Typography variant="body1">
          This is a placeholder for the access control management page. Here users will be able to manage who has access to their identity information.
        </Typography>
        <Box sx={{ bgcolor: '#f5f5f5', p: 2, mt: 2, borderRadius: 1 }}>
          <Typography variant="body2" color="textSecondary">
            Coming soon: Blockchain-based access control management
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default AccessControlPage; 