import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Container, 
  Grid, 
  Card, 
  CardContent, 
  CardActions,
  CardMedia,
  Paper,
  Divider,
  useTheme,
  alpha,
  Chip
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { 
  Security, 
  Fingerprint, 
  VerifiedUser, 
  Lock, 
  Speed, 
  AccountBalance,
  GppGood,
  DataUsage,
  CheckCircle,
  ArrowForward,
  LockOutlined,
  SupervisorAccount,
  Visibility,
  DocumentScanner,
  FaceRetouchingNatural,
  HowToReg
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';

const featureCards = [
  {
    title: 'Biometric Verification',
    description: 'Verify your identity using facial recognition and other biometric data with industry-leading accuracy.',
    icon: <FaceRetouchingNatural fontSize="large" />,
    color: '#0B3954',
    link: '/register'
  },
  {
    title: 'Document Verification',
    description: 'Verify government-issued IDs with advanced AI detection to prevent fraud and tampering.',
    icon: <DocumentScanner fontSize="large" />,
    color: '#1D7874',
    link: '/register'
  },
  {
    title: 'Privacy Preservation',
    description: 'Zero-knowledge proofs allow verification without exposing your sensitive personal data.',
    icon: <LockOutlined fontSize="large" />,
    color: '#125550',
    link: '/register'
  },
  {
    title: 'Blockchain Security',
    description: 'Your identity data is secured and controlled through immutable blockchain technology.',
    icon: <Security fontSize="large" />,
    color: '#196087',
    link: '/register'
  },
  {
    title: 'Real-time Verification',
    description: 'Near-instant identity verification with AI-driven fraud detection and risk assessment.',
    icon: <Speed fontSize="large" />,
    color: '#26a69a',
    link: '/verify'
  },
  {
    title: 'User-Controlled Access',
    description: 'Grant and revoke access to your identity data with blockchain-based permissions system.',
    icon: <SupervisorAccount fontSize="large" />,
    color: '#00838f',
    link: '/access-control'
  }
];

const HomePage = () => {
  const { isAuthenticated } = useAuth();
  const theme = useTheme();
  
  return (
    <Box>
      {/* Hero Section */}
      <Box 
        sx={{ 
          py: { xs: 8, md: 12 },
          background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 40%, ${theme.palette.secondary.main} 100%)`,
          color: '#ffffff',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {/* Decorative graphic elements */}
        <Box sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          opacity: 0.1,
          zIndex: 0,
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.3) 2px, transparent 2px)',
          backgroundSize: '30px 30px',
        }} />
        
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={7}>
              <Chip 
                label="Blockchain + AI Technology" 
                color="secondary" 
                sx={{ 
                  mb: 3, 
                  fontWeight: 500, 
                  fontSize: '0.875rem',
                  py: 1.5,
                  px: 2,
                  '& .MuiChip-label': {
                    px: 1
                  }
                }} 
              />
              <Typography 
                variant="h2" 
                component="h1" 
                gutterBottom
                sx={{ 
                  fontWeight: 800,
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  lineHeight: 1.2,
                  background: 'linear-gradient(90deg, #ffffff 30%, #e0f7fa 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  color: 'transparent',
                  mb: 3
                }}
              >
                Secure Identity Verification
              </Typography>
              <Typography 
                variant="h5" 
                sx={{ 
                  fontWeight: 400, 
                  opacity: 0.9,
                  mb: 3
                }}
              >
                Privacy-Preserving. User-Controlled. Decentralized.
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  fontSize: '1.1rem', 
                  opacity: 0.8,
                  maxWidth: '90%',
                  mb: 4 
                }}
              >
                Our system combines AI, blockchain technology, and advanced cryptography to provide 
                secure, private, and user-controlled identity verification that complies with GDPR and CCPA.
              </Typography>
              
              <Box sx={{ mt: 5, display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                {isAuthenticated ? (
                  <Button 
                    variant="contained" 
                    color="secondary" 
                    size="large"
                    component={RouterLink}
                    to="/dashboard"
                    startIcon={<GppGood />}
                    sx={{ 
                      py: 1.5, 
                      px: 4, 
                      fontSize: '1rem',
                      boxShadow: '0 8px 20px rgba(29, 120, 116, 0.3)',
                      '&:hover': {
                        boxShadow: '0 12px 25px rgba(29, 120, 116, 0.4)',
                      }
                    }}
                  >
                    Go to Dashboard
                  </Button>
                ) : (
                  <>
                    <Button 
                      variant="contained" 
                      color="secondary" 
                      size="large"
                      component={RouterLink}
                      to="/register"
                      startIcon={<Security />}
                      sx={{ 
                        py: 1.5, 
                        px: 4, 
                        fontSize: '1rem',
                        boxShadow: '0 8px 20px rgba(29, 120, 116, 0.3)',
                        '&:hover': {
                          boxShadow: '0 12px 25px rgba(29, 120, 116, 0.4)',
                        }
                      }}
                    >
                      Register Now
                    </Button>
                    <Button 
                      variant="outlined" 
                      size="large" 
                      component={RouterLink}
                      to="/verify"
                      startIcon={<VerifiedUser />}
                      sx={{ 
                        py: 1.5, 
                        px: 4, 
                        fontSize: '1rem',
                        borderColor: 'rgba(255,255,255,0.7)',
                        color: '#ffffff',
                        borderWidth: 2,
                        '&:hover': {
                          borderColor: '#ffffff',
                          backgroundColor: 'rgba(255,255,255,0.1)',
                        }
                      }}
                    >
                      Verify Identity
                    </Button>
                  </>
                )}
              </Box>
            </Grid>
            <Grid 
              item 
              xs={12} 
              md={5} 
              sx={{ 
                display: { xs: 'none', md: 'flex' },
                justifyContent: 'center'
              }}
            >
              <Box sx={{ 
                position: 'relative', 
                height: 420, 
                width: 420, 
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center'
              }}>
                {/* Animated background rings */}
                <Box sx={{
                  position: 'absolute',
                  width: 420,
                  height: 420,
                  borderRadius: '50%',
                  border: '2px solid rgba(255,255,255,0.1)',
                  animation: 'pulse 3s infinite',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(0.95)', opacity: 0.7 },
                    '50%': { transform: 'scale(1.05)', opacity: 0.3 },
                    '100%': { transform: 'scale(0.95)', opacity: 0.7 },
                  }
                }} />
                <Box sx={{
                  position: 'absolute',
                  width: 320,
                  height: 320,
                  borderRadius: '50%',
                  border: '4px solid rgba(255,255,255,0.15)',
                  animation: 'pulse 3s infinite 0.3s',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(0.95)', opacity: 0.7 },
                    '50%': { transform: 'scale(1.05)', opacity: 0.3 },
                    '100%': { transform: 'scale(0.95)', opacity: 0.7 },
                  }
                }} />
                <Box sx={{
                  position: 'absolute',
                  width: 220,
                  height: 220,
                  borderRadius: '50%',
                  border: '6px solid rgba(255,255,255,0.2)',
                  animation: 'pulse 3s infinite 0.6s',
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(0.95)', opacity: 0.7 },
                    '50%': { transform: 'scale(1.05)', opacity: 0.3 },
                    '100%': { transform: 'scale(0.95)', opacity: 0.7 },
                  }
                }} />
                
                {/* Central icon */}
                <VerifiedUser sx={{ 
                  fontSize: 120, 
                  opacity: 0.9,
                  color: '#ffffff',
                  filter: 'drop-shadow(0 0 20px rgba(255,255,255,0.3))'
                }} />
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Key Features Section */}
      <Container maxWidth="lg" sx={{ py: 10 }}>
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography 
            variant="h3" 
            component="h2" 
            sx={{ 
              fontWeight: 700, 
              mb: 3,
              color: theme.palette.primary.main
            }}
          >
            Key Features
          </Typography>
          <Typography 
            variant="h6" 
            sx={{ 
              maxWidth: 700, 
              mx: 'auto', 
              color: theme.palette.text.secondary,
              fontWeight: 400,
              lineHeight: 1.6
            }}
          >
            Our blockchain-based identity verification system combines cutting-edge technology with privacy-first design
          </Typography>
        </Box>
        
        <Grid container spacing={4}>
          {featureCards.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card 
                elevation={1}
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  borderRadius: 3,
                  transition: 'all 0.3s ease-in-out',
                  overflow: 'hidden',
                  '&:hover': {
                    transform: 'translateY(-12px)',
                    boxShadow: '0 12px 20px rgba(0,0,0,0.1)',
                    '& .feature-icon-wrapper': {
                      transform: 'scale(1.1)'
                    }
                  }
                }}
              >
                <Box sx={{ 
                  pt: 5, 
                  px: 2, 
                  display: 'flex', 
                  justifyContent: 'center' 
                }}>
                  <Box 
                    className="feature-icon-wrapper"
                    sx={{ 
                      width: 80, 
                      height: 80, 
                      borderRadius: '50%', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      background: alpha(feature.color, 0.12),
                      color: feature.color,
                      transition: 'transform 0.3s ease-in-out'
                    }}
                  >
                    {feature.icon}
                  </Box>
                </Box>
                <CardContent sx={{ flexGrow: 1, textAlign: 'center', pt: 3 }}>
                  <Typography 
                    variant="h5" 
                    component="h3" 
                    gutterBottom
                    sx={{ 
                      fontWeight: 600,
                      mb: 2,
                      color: feature.color
                    }}
                  >
                    {feature.title}
                  </Typography>
                  <Typography 
                    variant="body1" 
                    color="text.secondary"
                    sx={{ 
                      fontSize: '0.95rem',
                      px: 1,
                      minHeight: '4.5rem'
                    }}
                  >
                    {feature.description}
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: 'center', pb: 4 }}>
                  <Button 
                    component={RouterLink} 
                    to={feature.link} 
                    variant="text"
                    endIcon={<ArrowForward />}
                    sx={{ 
                      color: feature.color,
                      '&:hover': {
                        backgroundColor: alpha(feature.color, 0.08)
                      }
                    }}
                  >
                    Learn More
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* How It Works Section */}
      <Box sx={{ 
        py: 10, 
        background: alpha(theme.palette.primary.main, 0.04)
      }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography 
              variant="h3" 
              component="h2" 
              sx={{ 
                fontWeight: 700, 
                mb: 3,
                color: theme.palette.primary.main
              }}
            >
              How It Works
            </Typography>
            <Typography 
              variant="h6" 
              sx={{ 
                maxWidth: 700, 
                mx: 'auto', 
                color: theme.palette.text.secondary,
                fontWeight: 400,
                lineHeight: 1.6
              }}
            >
              Three simple steps to secure your identity with blockchain and AI technology
            </Typography>
          </Box>

          <Grid container spacing={5} alignItems="center" sx={{ mb: 4 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ 
                  display: 'inline-flex', 
                  p: 2, 
                  borderRadius: '50%', 
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  mb: 3
                }}>
                  <HowToReg sx={{ fontSize: 60, color: theme.palette.primary.main }} />
                </Box>
                <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                  1. Register
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Create your account by providing basic information, a face scan, and a government-issued ID document
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ 
                  display: 'inline-flex', 
                  p: 2, 
                  borderRadius: '50%', 
                  bgcolor: alpha(theme.palette.secondary.main, 0.1),
                  mb: 3
                }}>
                  <LockOutlined sx={{ fontSize: 60, color: theme.palette.secondary.main }} />
                </Box>
                <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                  2. Secure Storage
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Your identity is securely stored with encryption and only identity hashes are recorded on the blockchain
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Box sx={{ 
                  display: 'inline-flex', 
                  p: 2, 
                  borderRadius: '50%', 
                  bgcolor: alpha(theme.palette.success.main, 0.1),
                  mb: 3
                }}>
                  <Visibility sx={{ fontSize: 60, color: theme.palette.success.main }} />
                </Box>
                <Typography variant="h5" component="h3" gutterBottom fontWeight={600}>
                  3. Verify
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Verify your identity anytime using facial recognition, with complete control over who can access your data
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Call to Action */}
      <Box sx={{ 
        py: 10, 
        background: `linear-gradient(to right, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
        color: '#ffffff'
      }}>
        <Container maxWidth="md" sx={{ textAlign: 'center' }}>
          <Typography variant="h3" component="h2" gutterBottom fontWeight={700}>
            Ready to Get Started?
          </Typography>
          <Typography variant="h6" paragraph sx={{ opacity: 0.9, mb: 5 }}>
            Join thousands of users who trust our secure identity verification system
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3 }}>
            <Button 
              variant="contained" 
              size="large"
              component={RouterLink}
              to="/register"
              color="secondary"
              sx={{ 
                backgroundColor: '#ffffff', 
                color: theme.palette.primary.main,
                px: 4,
                py: 1.5,
                fontWeight: 600,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.9)'
                }
              }}
            >
              Register Now
            </Button>
            <Button 
              variant="outlined" 
              size="large"
              component={RouterLink}
              to="/verify"
              sx={{ 
                borderColor: '#ffffff', 
                color: '#ffffff',
                px: 4,
                py: 1.5,
                borderWidth: 2,
                fontWeight: 600,
                '&:hover': {
                  borderColor: '#ffffff',
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              Learn More
            </Button>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default HomePage; 