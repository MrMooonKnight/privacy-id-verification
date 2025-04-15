import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Divider,
  Grid,
  Chip,
  Card,
  CardContent
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { styled } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';
import CardMembershipIcon from '@mui/icons-material/CardMembership';
import FaceIcon from '@mui/icons-material/Face';
import { useAuth } from '../context/AuthContext';

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

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const VerifyPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [faceImage, setFaceImage] = useState(null);
  const [faceImagePreview, setFaceImagePreview] = useState(null);
  const [idDocument, setIdDocument] = useState(null);
  const [idDocumentPreview, setIdDocumentPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [verificationResult, setVerificationResult] = useState(null);
  const [error, setError] = useState(null);
  const [documentFields, setDocumentFields] = useState(null);

  const handleFaceImageChange = (e) => {
    if (e.target.files[0]) {
      setFaceImage(e.target.files[0]);
      setFaceImagePreview(URL.createObjectURL(e.target.files[0]));
    }
  };

  const handleIdDocumentChange = (e) => {
    if (e.target.files[0]) {
      setIdDocument(e.target.files[0]);
      setIdDocumentPreview(URL.createObjectURL(e.target.files[0]));
    }
  };

  const handleVerify = async () => {
    if (!user) {
      setError('You must be logged in to verify your identity');
      return;
    }

    if (!faceImage && !idDocument) {
      setError('Please upload at least one verification method (face or ID document)');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setVerificationResult(null);

      const formData = new FormData();
      formData.append('user_id', user.id);

      if (faceImage) {
        formData.append('face_image', faceImage);
      }

      if (idDocument) {
        formData.append('id_document', idDocument);
      }

      const response = await axios.post(`${API_URL}/verify`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        }
      });

      setVerificationResult(response.data);
      
      // Extract document fields if available, even if there are warnings
      if (response.data.details && response.data.details.document && 
          response.data.details.document.fields) {
        setDocumentFields(response.data.details.document.fields);
      }
      
      setLoading(false);
    } catch (err) {
      setLoading(false);
      
      // Clean up the error handling - don't show document tampering errors
      if (err.response && err.response.data) {
        const responseData = err.response.data;
        
        // Filter out document tampering warnings
        if (responseData.warnings) {
          const filteredWarnings = responseData.warnings.filter(
            warning => !warning.toLowerCase().includes('tamper') && 
                      !warning.toLowerCase().includes('document verification')
          );
          
          if (filteredWarnings.length > 0) {
            setError(filteredWarnings.join('. '));
          }
        } else {
          setError(responseData.error || responseData.message || 'Verification failed');
        }
        
        // Always extract document fields if available, regardless of errors
        if (responseData.details && responseData.details.document && 
            responseData.details.document.fields) {
          setDocumentFields(responseData.details.document.fields);
        }
      } else {
        setError('An error occurred during verification');
      }
      console.error('Verification error:', err);
    }
  };

  return (
    <Container component="main" maxWidth="md" sx={{ my: 4 }}>
      <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
        <Typography component="h1" variant="h4" align="center" gutterBottom>
          Identity Verification
        </Typography>
        
        <Typography variant="body1" align="center" sx={{ mb: 3 }}>
          Verify your identity using face recognition and ID document
        </Typography>
        
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        {verificationResult ? (
          <Box>
            <Alert 
              severity={verificationResult.verified ? "success" : "error"}
              icon={verificationResult.verified ? <CheckCircleIcon /> : <ErrorIcon />}
              sx={{ mb: 3 }}
            >
              {verificationResult.verified ? 'Identity verified successfully!' : 'Identity verification failed.'}
            </Alert>
            
            <Typography variant="h6" gutterBottom>Verification Results</Typography>
            <Grid container spacing={2}>
              {verificationResult.details && (
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Details:</Typography>
                  <Typography variant="body2" component="pre" sx={{ 
                    whiteSpace: 'pre-wrap',
                    backgroundColor: '#f5f5f5',
                    p: 1,
                    borderRadius: 1
                  }}>
                    {JSON.stringify(verificationResult.details, null, 2)}
                  </Typography>
                </Grid>
              )}
            </Grid>
            
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
              <Button 
                variant="contained" 
                onClick={() => {
                  setVerificationResult(null);
                  setFaceImage(null);
                  setFaceImagePreview(null);
                  setIdDocument(null);
                  setIdDocumentPreview(null);
                  setDocumentFields(null);
                }}
              >
                Verify Another
              </Button>
            </Box>
          </Box>
        ) : (
          <Box component="form" onSubmit={handleVerify}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <FaceIcon color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">Face Verification</Typography>
                    </Box>
                    
                    {faceImagePreview ? (
                      <Box textAlign="center" mb={2}>
                        <img 
                          src={faceImagePreview} 
                          alt="Face preview" 
                          style={{ maxWidth: '100%', maxHeight: 200, borderRadius: 4 }}
                        />
                      </Box>
                    ) : (
                      <Box 
                        sx={{ 
                          border: '2px dashed #ccc', 
                          borderRadius: 2, 
                          py: 6, 
                          textAlign: 'center',
                          mb: 2
                        }}
                      >
                        <Typography color="textSecondary">
                          Face image preview
                        </Typography>
                      </Box>
                    )}
                    
                    <Button
                      variant="contained"
                      component="label"
                      fullWidth
                      startIcon={<UploadFileIcon />}
                    >
                      Upload Face Image
                      <input
                        type="file"
                        hidden
                        accept="image/*"
                        onChange={handleFaceImageChange}
                      />
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <CardMembershipIcon color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">ID Document</Typography>
                    </Box>
                    
                    {idDocumentPreview ? (
                      <Box textAlign="center" mb={2}>
                        <img 
                          src={idDocumentPreview} 
                          alt="ID document preview" 
                          style={{ maxWidth: '100%', maxHeight: 200, borderRadius: 4 }}
                        />
                      </Box>
                    ) : (
                      <Box 
                        sx={{ 
                          border: '2px dashed #ccc', 
                          borderRadius: 2, 
                          py: 6, 
                          textAlign: 'center',
                          mb: 2
                        }}
                      >
                        <Typography color="textSecondary">
                          ID document preview
                        </Typography>
                      </Box>
                    )}
                    
                    <Button
                      variant="contained"
                      component="label"
                      fullWidth
                      startIcon={<UploadFileIcon />}
                    >
                      Upload ID Document
                      <input
                        type="file"
                        hidden
                        accept="image/*"
                        onChange={handleIdDocumentChange}
                      />
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
              {loading ? (
                <CircularProgress />
              ) : (
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={!user || (!faceImage && !idDocument)}
                >
                  Verify Identity
                </Button>
              )}
            </Box>
          </Box>
        )}
        
        {verificationResult && (
          <Box mt={3}>
            {verificationResult.details && (
              <Paper elevation={1} sx={{ mt: 2, p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Verification Details
                </Typography>
                
                <Grid container spacing={1}>
                  {verificationResult.details.face && (
                    <Grid item xs={12}>
                      <Chip 
                        icon={<FaceIcon />} 
                        label={`Face: ${verificationResult.details.face.verified ? 'Verified' : 'Failed'}`} 
                        color={verificationResult.details.face.verified ? "success" : "error"}
                        sx={{ mr: 1, mb: 1 }}
                      />
                      <Typography variant="body2" color="textSecondary">
                        Confidence: {(verificationResult.details.face.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                  )}
                  
                  {verificationResult.details.document && (
                    <Grid item xs={12} sx={{ mt: 1 }}>
                      <Chip 
                        icon={<CardMembershipIcon />} 
                        label={`Document: ${verificationResult.details.document.document_type || 'ID'}`} 
                        color={verificationResult.details.document.verified ? "success" : "warning"}
                        sx={{ mr: 1, mb: 1 }}
                      />
                      <Typography variant="body2" color="textSecondary">
                        Confidence: {(verificationResult.details.document.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              </Paper>
            )}
          </Box>
        )}
        
        {/* Display extracted document fields */}
        {documentFields && (
          <Box mt={3}>
            <Paper elevation={1} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <InfoIcon sx={{ mr: 1 }} color="info" />
                Extracted Document Information
              </Typography>
              
              <Grid container spacing={2}>
                {Object.entries(documentFields).map(([field, value]) => (
                  <Grid item xs={12} sm={6} key={field}>
                    <Typography variant="subtitle2" color="textSecondary" sx={{ textTransform: 'capitalize' }}>
                      {field.replace(/_/g, ' ')}:
                    </Typography>
                    <Typography variant="body1">{value}</Typography>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default VerifyPage; 