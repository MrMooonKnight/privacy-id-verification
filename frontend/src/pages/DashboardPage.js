import React from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button
} from '@mui/material';
import { useAuth } from '../context/AuthContext';
import { Link as RouterLink } from 'react-router-dom';

const DashboardPage = () => {
  const { userInfo } = useAuth();

  return (
    <Container component="main" maxWidth="lg" sx={{ my: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard
      </Typography>
      
      <Typography variant="body1" paragraph>
        Welcome to your secure identity dashboard. Here you can manage your identity verification settings and access controls.
      </Typography>
      
      <Grid container spacing={3}>
        {/* Identity Status */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h5" gutterBottom>
              Identity Status
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Verification Status" 
                  secondary="Verified" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="User ID" 
                  secondary={userInfo?.userId || "Not available"} 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Last Verification" 
                  secondary="2023-10-15 14:30:22" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Identity Hash" 
                  secondary="0x8f7e9a2c3b4d5e6f..." 
                />
              </ListItem>
            </List>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                component={RouterLink} 
                to="/verify" 
                variant="contained"
              >
                Re-verify Identity
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Access Control */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h5" gutterBottom>
              Access Control
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Active Authorizations" 
                  secondary="2 Services" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Data Access Requests" 
                  secondary="1 Pending" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Last Access" 
                  secondary="2023-10-14 09:15:43" 
                />
              </ListItem>
            </List>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                component={RouterLink} 
                to="/access-control" 
                variant="contained"
              >
                Manage Access
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Recent Activity */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>
              Recent Activity
            </Typography>
            <List>
              <ListItem>
                <ListItemText 
                  primary="Identity Verification" 
                  secondary="2023-10-15 14:30:22" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Access Granted to Financial Services Inc." 
                  secondary="2023-10-10 11:45:36" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Profile Information Updated" 
                  secondary="2023-10-05 16:20:15" 
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemText 
                  primary="Initial Registration" 
                  secondary="2023-10-01 09:30:00" 
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
        
        {/* Quick Actions */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Button 
              component={RouterLink} 
              to="/profile" 
              variant="outlined"
            >
              View Profile
            </Button>
            <Button 
              component={RouterLink} 
              to="/access-control" 
              variant="outlined"
            >
              Manage Access
            </Button>
            <Button 
              component={RouterLink} 
              to="/verify" 
              variant="outlined"
            >
              Verify Identity
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage; 