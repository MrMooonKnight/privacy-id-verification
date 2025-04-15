import React, { useState } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  useMediaQuery,
  useTheme,
  Container,
  Tooltip,
  Fade,
} from '@mui/material';
import {
  Menu as MenuIcon,
  AccountCircle,
  Dashboard,
  VpnKey,
  ExitToApp,
  Home,
  LockOpen,
  HowToReg,
  Security,
  VerifiedUser,
} from '@mui/icons-material';
import { useAuth } from '../../context/AuthContext';

const Header = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // State for user menu
  const [anchorEl, setAnchorEl] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Handle user menu open
  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  // Handle user menu close
  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  // Handle mobile drawer toggle
  const toggleDrawer = (open) => (event) => {
    if (
      event.type === 'keydown' &&
      (event.key === 'Tab' || event.key === 'Shift')
    ) {
      return;
    }
    setDrawerOpen(open);
  };

  // Handle logout
  const handleLogout = () => {
    logout();
    handleMenuClose();
    navigate('/');
  };

  // Mobile drawer content
  const drawerContent = (
    <Box
      sx={{ width: 280 }}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <Box sx={{ 
        p: 3, 
        display: 'flex', 
        alignItems: 'center', 
        background: 'linear-gradient(120deg, #0B3954 0%, #1D7874 100%)',
        color: 'white'
      }}>
        <VerifiedUser sx={{ mr: 1.5, fontSize: '2rem' }} />
        <Typography variant="h6" component="div">
          Identity Verification
        </Typography>
      </Box>
      <Divider />
      <List>
        <ListItem button component={RouterLink} to="/" sx={{ py: 1.5 }}>
          <ListItemIcon>
            <Home color="primary" />
          </ListItemIcon>
          <ListItemText primary="Home" />
        </ListItem>

        {isAuthenticated ? (
          <>
            <ListItem button component={RouterLink} to="/dashboard" sx={{ py: 1.5 }}>
              <ListItemIcon>
                <Dashboard color="primary" />
              </ListItemIcon>
              <ListItemText primary="Dashboard" />
            </ListItem>
            <ListItem button component={RouterLink} to="/profile" sx={{ py: 1.5 }}>
              <ListItemIcon>
                <AccountCircle color="primary" />
              </ListItemIcon>
              <ListItemText primary="Profile" />
            </ListItem>
            <ListItem button component={RouterLink} to="/access-control" sx={{ py: 1.5 }}>
              <ListItemIcon>
                <VpnKey color="primary" />
              </ListItemIcon>
              <ListItemText primary="Access Control" />
            </ListItem>
            <Divider sx={{ my: 2 }} />
            <ListItem button onClick={handleLogout} sx={{ py: 1.5 }}>
              <ListItemIcon>
                <ExitToApp color="error" />
              </ListItemIcon>
              <ListItemText primary="Logout" primaryTypographyProps={{ color: 'error' }} />
            </ListItem>
          </>
        ) : (
          <>
            <ListItem button component={RouterLink} to="/register" sx={{ py: 1.5 }}>
              <ListItemIcon>
                <HowToReg color="primary" />
              </ListItemIcon>
              <ListItemText primary="Register" />
            </ListItem>
            <ListItem button component={RouterLink} to="/verify" sx={{ py: 1.5 }}>
              <ListItemIcon>
                <LockOpen color="primary" />
              </ListItemIcon>
              <ListItemText primary="Verify Identity" />
            </ListItem>
          </>
        )}
      </List>
    </Box>
  );

  return (
    <AppBar 
      position="sticky" 
      elevation={0}
      sx={{ 
        background: 'linear-gradient(90deg, #0B3954 0%, #1D7874 100%)',
        borderBottom: '1px solid rgba(255,255,255,0.1)'
      }}
    >
      <Container maxWidth="lg">
        <Toolbar sx={{ py: 1 }}>
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleDrawer(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}

          <Typography
            variant="h6"
            component={RouterLink}
            to="/"
            sx={{
              color: 'white',
              textDecoration: 'none',
              flexGrow: 1,
              display: 'flex',
              alignItems: 'center',
              fontWeight: 500,
              letterSpacing: '0.5px',
            }}
          >
            <VerifiedUser sx={{ mr: 1.5, fontSize: '2rem' }} />
            Secure Identity Verification
          </Typography>

          {/* Desktop navigation */}
          {!isMobile && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {isAuthenticated ? (
                <>
                  <Button
                    color="inherit"
                    component={RouterLink}
                    to="/dashboard"
                    sx={{ 
                      mx: 1, 
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: 'rgba(255,255,255,0.1)'
                      }
                    }}
                    startIcon={<Dashboard />}
                  >
                    Dashboard
                  </Button>
                  <Button
                    color="inherit"
                    component={RouterLink}
                    to="/access-control"
                    sx={{ 
                      mx: 1, 
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: 'rgba(255,255,255,0.1)'
                      }
                    }}
                    startIcon={<VpnKey />}
                  >
                    Access Control
                  </Button>
                  <Tooltip title="Account settings" TransitionComponent={Fade} arrow>
                    <IconButton
                      color="inherit"
                      onClick={handleMenuOpen}
                      aria-label="account"
                      aria-controls="menu-appbar"
                      aria-haspopup="true"
                      sx={{ ml: 2 }}
                    >
                      <Avatar
                        sx={{ 
                          width: 40, 
                          height: 40, 
                          bgcolor: 'secondary.main',
                          border: '2px solid white',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                        }}
                      >
                        {user?.userId?.charAt(0) || 'U'}
                      </Avatar>
                    </IconButton>
                  </Tooltip>
                  <Menu
                    id="menu-appbar"
                    anchorEl={anchorEl}
                    anchorOrigin={{
                      vertical: 'bottom',
                      horizontal: 'right',
                    }}
                    keepMounted
                    transformOrigin={{
                      vertical: 'top',
                      horizontal: 'right',
                    }}
                    open={Boolean(anchorEl)}
                    onClose={handleMenuClose}
                    sx={{
                      '& .MuiPaper-root': {
                        borderRadius: 2,
                        minWidth: 180,
                        boxShadow: '0px 5px 15px rgba(0,0,0,0.15)',
                        mt: 1
                      }
                    }}
                  >
                    <MenuItem
                      component={RouterLink}
                      to="/profile"
                      onClick={handleMenuClose}
                      sx={{ py: 1.5 }}
                    >
                      <ListItemIcon>
                        <AccountCircle fontSize="small" color="primary" />
                      </ListItemIcon>
                      <Typography variant="body1">Profile</Typography>
                    </MenuItem>
                    <Divider />
                    <MenuItem onClick={handleLogout} sx={{ py: 1.5 }}>
                      <ListItemIcon>
                        <ExitToApp fontSize="small" color="error" />
                      </ListItemIcon>
                      <Typography variant="body1" color="error">Logout</Typography>
                    </MenuItem>
                  </Menu>
                </>
              ) : (
                <>
                  <Button
                    color="inherit"
                    component={RouterLink}
                    to="/register"
                    sx={{ 
                      mx: 1,
                      fontWeight: 500,
                      borderRadius: 2,
                      '&:hover': {
                        backgroundColor: 'rgba(255,255,255,0.1)'
                      }
                    }}
                    startIcon={<HowToReg />}
                  >
                    Register
                  </Button>
                  <Button
                    component={RouterLink}
                    to="/verify"
                    variant="contained"
                    color="secondary"
                    sx={{ 
                      ml: 1,
                      fontWeight: 500,
                      borderRadius: 2,
                      boxShadow: '0 4px 10px rgba(29, 120, 116, 0.3)',
                      '&:hover': {
                        boxShadow: '0 6px 15px rgba(29, 120, 116, 0.4)',
                      }
                    }}
                    startIcon={<LockOpen />}
                  >
                    Verify Identity
                  </Button>
                </>
              )}
            </Box>
          )}
        </Toolbar>
      </Container>
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={toggleDrawer(false)}
      >
        {drawerContent}
      </Drawer>
    </AppBar>
  );
};

export default Header; 