import axios from 'axios'

// Use environment variable or default to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

console.log('🔗 Connecting to API:', API_BASE_URL)

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased to 60 seconds for image processing
})

export const assessDamage = async (pickupImages, returnImages) => {
  const formData = new FormData()

  // Add pickup images
  pickupImages.forEach(image => {
    formData.append('pickup_images', image)
  })

  // Add return images
  returnImages.forEach(image => {
    formData.append('return_images', image)
  })

  try {
    const response = await api.post('/api/assess-damage', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw new Error(error.response?.data?.detail || 'Failed to assess damage. Please try again.')
  }
}

export const healthCheck = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw new Error('Backend service is unavailable')
  }
}

export const analyzeSingleImage = async (image) => {
  const formData = new FormData()
  formData.append('image', image)

  try {
    const response = await api.post('/api/analyze-single', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    console.error('Single image analysis error:', error)
    throw new Error(error.response?.data?.detail || 'Failed to analyze image')
  }
}