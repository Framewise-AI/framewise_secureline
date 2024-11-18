class SecureLineError(Exception):
    """Base exception class for SecureLine errors"""
    pass

class APIError(SecureLineError):
    """Raised when API returns an error response"""
    pass

class ValidationError(SecureLineError):
    """Raised when input validation fails"""
    pass

class TimeoutError(SecureLineError):
    """Raised when API request times out"""
    pass

class ValidationError(SecureLineError):
    """Raised if the parameters are not in range"""
    pass