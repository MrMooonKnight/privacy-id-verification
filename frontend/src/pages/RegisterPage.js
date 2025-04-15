import React, { useState } from 'react';
import { 
  Typography, 
  Box, 
  Paper, 
  Stepper, 
  Step, 
  StepLabel,
  Button,
  Container,
  TextField,
  Grid,
  Alert,
  CircularProgress
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { styled } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';

// Styled components for file upload
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const UploadPreview = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
  height: 200,
  border: '1px solid #ccc',
  borderRadius: theme.shape.borderRadius,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  overflow: 'hidden',
  '& img': {
    maxHeight: '100%',
    maxWidth: '100%',
  }
}));

// Registration steps
const steps = ['Personal Information', 'Face Image', 'ID Document', 'Review'];

const RegisterPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phoneNumber: '',
    dateOfBirth: '',
    address: ''
  });
  const [faceImage, setFaceImage] = useState(null);
  const [facePreview, setFacePreview] = useState('');
  const [idDocument, setIdDocument] = useState(null);
  const [idPreview, setIdPreview] = useState('');
  const [error, setError] = useState('');
  const [warnings, setWarnings] = useState([]);
  const [loading, setLoading] = useState(false);

  // Handle form field changes
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // Handle face image upload
  const handleFaceUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setFaceImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setFacePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle ID document upload
  const handleIdUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setIdDocument(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setIdPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Move to next step
  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      handleSubmit();
    } else {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  // Move to previous step
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  // Submit registration form
  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    setWarnings([]);

    try {
      // Create form data for multipart/form-data
      const data = new FormData();
      
      // Add form fields
      Object.keys(formData).forEach(key => {
        data.append(key, formData[key]);
      });
      
      // Add files
      data.append('face_image', faceImage);
      data.append('id_document', idDocument);

      // Submit to API
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/register`, 
        data,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      // Check if there are any warnings
      if (response.data.warnings && response.data.warnings.length > 0) {
        setWarnings(response.data.warnings);
      }

      // Log in the user with the token
      if (response.data.token && response.data.user_id) {
        login(response.data.token, response.data.user_id);
        navigate('/dashboard');
      } else {
        throw new Error('Registration successful but no token received');
      }
    } catch (err) {
      console.error('Registration error:', err);
      
      // Check if the error response contains warnings but registration can proceed
      if (err.response?.data?.warnings) {
        setWarnings(err.response.data.warnings);
        
        // If there's also a token and user_id despite the warning, proceed with login
        if (err.response.data.token && err.response.data.user_id) {
          login(err.response.data.token, err.response.data.user_id);
          navigate('/dashboard');
          return;
        }
      }
      
      setError(err.response?.data?.error || 'Failed to register. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Render step content based on active step
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="First Name"
                name="firstName"
                value={formData.firstName}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Last Name"
                name="lastName"
                value={formData.lastName}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email Address"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Phone Number"
                name="phoneNumber"
                value={formData.phoneNumber}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Date of Birth"
                name="dateOfBirth"
                type="date"
                value={formData.dateOfBirth}
                onChange={handleChange}
                InputLabelProps={{ shrink: true }}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Address"
                name="address"
                value={formData.address}
                onChange={handleChange}
                multiline
                rows={3}
                required
              />
            </Grid>
          </Grid>
        );
      case 1:
        return (
          <Box>
            <Typography variant="body1" gutterBottom>
              Please upload a clear photo of your face. This will be used for biometric verification.
            </Typography>
            <Button
              component="label"
              variant="contained"
              startIcon={<CloudUploadIcon />}
            >
              Upload Face Image
              <VisuallyHiddenInput type="file" accept="image/*" onChange={handleFaceUpload} />
            </Button>
            <UploadPreview>
              {facePreview ? (
                <img src={facePreview} alt="Face preview" />
              ) : (
                <Typography variant="body2" color="textSecondary">No image selected</Typography>
              )}
            </UploadPreview>
          </Box>
        );
      case 2:
        return (
          <Box>
            <Typography variant="body1" gutterBottom>
              Please upload a clear photo of your government-issued ID document.
            </Typography>
            <Button
              component="label"
              variant="contained"
              startIcon={<CloudUploadIcon />}
            >
              Upload ID Document
              <VisuallyHiddenInput type="file" accept="image/*" onChange={handleIdUpload} />
            </Button>
            <UploadPreview>
              {idPreview ? (
                <img src={idPreview} alt="ID document preview" />
              ) : (
                <Typography variant="body2" color="textSecondary">No document selected</Typography>
              )}
            </UploadPreview>
          </Box>
        );
      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>Review Your Information</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle1">Name:</Typography>
                <Typography variant="body1">{`${formData.firstName} ${formData.lastName}`}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle1">Email:</Typography>
                <Typography variant="body1">{formData.email}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle1">Phone:</Typography>
                <Typography variant="body1">{formData.phoneNumber}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle1">Date of Birth:</Typography>
                <Typography variant="body1">{formData.dateOfBirth}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle1">Address:</Typography>
                <Typography variant="body1">{formData.address}</Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle1">Face Image:</Typography>
                <Box sx={{ width: 100, height: 100, overflow: 'hidden', border: '1px solid #ccc' }}>
                  {facePreview && <img src={facePreview} alt="Face preview" style={{ maxWidth: '100%', maxHeight: '100%' }} />}
                </Box>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle1">ID Document:</Typography>
                <Box sx={{ width: 100, height: 100, overflow: 'hidden', border: '1px solid #ccc' }}>
                  {idPreview && <img src={idPreview} alt="ID preview" style={{ maxWidth: '100%', maxHeight: '100%' }} />}
                </Box>
              </Grid>
            </Grid>
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  // Check if can proceed to next step
  const canProceed = () => {
    switch (activeStep) {
      case 0:
        return formData.firstName && formData.lastName && formData.email && 
               formData.phoneNumber && formData.dateOfBirth && formData.address;
      case 1:
        return faceImage !== null;
      case 2:
        return idDocument !== null;
      case 3:
        return true;
      default:
        return false;
    }
  };

  return (
    <Container component="main" maxWidth="md" sx={{ mb: 4 }}>
      <Paper variant="outlined" sx={{ my: { xs: 3, md: 6 }, p: { xs: 2, md: 3 } }}>
        <Typography component="h1" variant="h4" align="center" gutterBottom>
          Identity Verification Registration
        </Typography>
        
        <Stepper activeStep={activeStep} sx={{ pt: 3, pb: 5 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        {warnings.length > 0 && warnings.map((warning, index) => (
          <Alert severity="warning" sx={{ mb: 2 }} key={index}>
            {warning}
          </Alert>
        ))}
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {getStepContent(activeStep)}
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
              <Button
                disabled={activeStep === 0}
                onClick={handleBack}
              >
                Back
              </Button>
              <Button
                variant="contained"
                onClick={handleNext}
                disabled={!canProceed()}
              >
                {activeStep === steps.length - 1 ? 'Submit Registration' : 'Next'}
              </Button>
            </Box>
          </>
        )}
      </Paper>
    </Container>
  );
};

export default RegisterPage; 