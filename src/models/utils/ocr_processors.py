from pathlib import Path
import platform
from enum import Enum

class OCRBackend(Enum):
    """Available OCR backends."""
    MACOS_LIVETEXT = "macos_livetext"
    TESSERACT = "tesseract"


class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass


class OCRProcessor:
    """Handles OCR operations with platform-specific implementations."""
    
    def __init__(self):
        self.backend = self._detect_backend()
        self._initialize_backend()
    
    @staticmethod
    def _detect_backend() -> OCRBackend:
        """Detect appropriate OCR backend for current platform."""
        return (
            OCRBackend.MACOS_LIVETEXT
            if platform.system() == "Darwin"
            else OCRBackend.TESSERACT
        )
    
    def _initialize_backend(self) -> None:
        """Initialize the OCR backend dependencies."""
        if self.backend == OCRBackend.MACOS_LIVETEXT:
            try:
                from ocrmac import ocrmac
                self._ocrmac = ocrmac
            except ImportError as e:
                raise OCRError(
                    "ocrmac not installed. Install with: pip install ocrmac"
                ) from e
        else:
            try:
                import pytesseract
                from PIL import Image
                self._pytesseract = pytesseract
                self._Image = Image
            except ImportError as e:
                raise OCRError(
                    "pytesseract or PIL not installed. "
                    "Install with: pip install pytesseract Pillow"
                ) from e
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using the appropriate OCR backend.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
            
        Raises:
            OCRError: If OCR processing fails
            FileNotFoundError: If image file doesn't exist
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            if self.backend == OCRBackend.MACOS_LIVETEXT:
                return self._extract_macos(image_path)
            else:
                return self._extract_tesseract(image_path)
        except Exception as e:
            raise OCRError(f"OCR failed for {image_path}: {e}") from e
    
    def _extract_macos(self, image_path: str) -> str:
        """Extract text using macOS Live Text."""
        annotations = self._ocrmac.OCR(
            image_path, framework="livetext"
        ).recognize()
        
        lines = [text for text, conf, bb in annotations]
        return " ".join(lines)
    
    def _extract_tesseract(self, image_path: str) -> str:
        """Extract text using Tesseract OCR with English and Ukrainian support."""
        img = self._Image.open(image_path)
        text = self._pytesseract.image_to_string(img, lang="eng+ukr")
        return (text or "").strip()