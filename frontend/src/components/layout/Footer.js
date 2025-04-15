import React from 'react';
import { Box, Container, Typography, Link, Grid } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { Security, Lock, Fingerprint } from '@mui/icons-material';

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: (theme) => theme.palette.grey[100],
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={3}>
          <Grid item xs={12} sm={4}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Security sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h6" color="text.primary">
                Secure Identity Verification
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Blockchain-based AI for privacy-preserving identity verification.
              Secure, private, and user-controlled identity management.
            </Typography>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              Quick Links
            </Typography>
            <Box>
              <Link component={RouterLink} to="/" color="inherit" sx={{ display: 'block', mb: 1 }}>
                Home
              </Link>
              <Link component={RouterLink} to="/register" color="inherit" sx={{ display: 'block', mb: 1 }}>
                Register
              </Link>
              <Link component={RouterLink} to="/verify" color="inherit" sx={{ display: 'block', mb: 1 }}>
                Verify Identity
              </Link>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Typography variant="h6" color="text.primary" gutterBottom>
              <Lock sx={{ mr: 1, verticalAlign: 'middle' }} />
              Privacy Features
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              <Fingerprint sx={{ mr: 1, verticalAlign: 'middle', fontSize: 'small' }} />
              Biometric Verification
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              <Lock sx={{ mr: 1, verticalAlign: 'middle', fontSize: 'small' }} />
              Zero-Knowledge Proofs
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              <Security sx={{ mr: 1, verticalAlign: 'middle', fontSize: 'small' }} />
              Blockchain Security
            </Typography>
          </Grid>
        </Grid>
        
        <Box mt={3} pt={3} borderTop={1} borderColor="divider">
          <Typography variant="body2" color="text.secondary" align="center">
            &copy; {new Date().getFullYear()} Secure Identity Verification. All rights reserved.
          </Typography>
          <Typography variant="body2" color="text.secondary" align="center">
            This system complies with GDPR and CCPA regulations.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer; 