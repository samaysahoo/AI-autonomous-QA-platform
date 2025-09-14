"""Vision-based test healing using computer vision to resolve UI element locators."""

import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import base64
from PIL import Image
import io
from dataclasses import dataclass

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Represents a UI element found by vision analysis."""
    element_type: str  # button, text, input, etc.
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class HealingResult:
    """Result of a test healing attempt."""
    success: bool
    new_locator: Optional[str] = None
    confidence: float = 0.0
    screenshot_path: Optional[str] = None
    detected_elements: List[UIElement] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.detected_elements is None:
            self.detected_elements = []


class VisionHealer:
    """Vision-based test healing for resolving UI element locators."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cascade_classifiers = self._load_cascade_classifiers()
        self.template_matcher = TemplateMatcher()
    
    def _load_cascade_classifiers(self) -> Dict[str, cv2.CascadeClassifier]:
        """Load OpenCV cascade classifiers for different UI elements."""
        
        classifiers = {}
        
        try:
            # Load standard OpenCV classifiers
            classifiers['button'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            classifiers['text'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            logger.info("Loaded cascade classifiers for UI element detection")
            
        except Exception as e:
            logger.error(f"Error loading cascade classifiers: {e}")
        
        return classifiers
    
    async def heal_test(self, failed_locator: str, screenshot_path: str, 
                       context: Dict[str, Any]) -> HealingResult:
        """Attempt to heal a failed test by finding alternative locators."""
        
        try:
            # Load and analyze screenshot
            screenshot = self._load_screenshot(screenshot_path)
            if screenshot is None:
                return HealingResult(
                    success=False,
                    error_message="Failed to load screenshot"
                )
            
            # Detect UI elements in the screenshot
            detected_elements = await self._detect_ui_elements(screenshot)
            
            # Try to find alternative locators for the failed element
            healing_result = await self._find_alternative_locator(
                failed_locator, detected_elements, context
            )
            
            healing_result.detected_elements = detected_elements
            healing_result.screenshot_path = screenshot_path
            
            return healing_result
            
        except Exception as e:
            logger.error(f"Error during test healing: {e}")
            return HealingResult(
                success=False,
                error_message=str(e)
            )
    
    def _load_screenshot(self, screenshot_path: str) -> Optional[np.ndarray]:
        """Load screenshot image for analysis."""
        
        try:
            if screenshot_path.startswith('data:image'):
                # Handle base64 encoded image
                image_data = base64.b64decode(screenshot_path.split(',')[1])
                image = Image.open(io.BytesIO(image_data))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                # Load from file
                return cv2.imread(screenshot_path)
                
        except Exception as e:
            logger.error(f"Error loading screenshot: {e}")
            return None
    
    async def _detect_ui_elements(self, screenshot: np.ndarray) -> List[UIElement]:
        """Detect UI elements in the screenshot using computer vision."""
        
        elements = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Detect buttons using template matching
        buttons = await self._detect_buttons(gray)
        elements.extend(buttons)
        
        # Detect text elements using OCR
        text_elements = await self._detect_text_elements(gray)
        elements.extend(text_elements)
        
        # Detect input fields
        input_fields = await self._detect_input_fields(gray)
        elements.extend(input_fields)
        
        logger.info(f"Detected {len(elements)} UI elements")
        return elements
    
    async def _detect_buttons(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect button elements using template matching."""
        
        buttons = []
        
        try:
            # Use template matching to find button-like shapes
            # This is a simplified implementation - in practice, you'd use
            # more sophisticated computer vision techniques
            
            # Detect rectangular shapes that could be buttons
            contours, _ = cv2.findContours(
                cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size to find button-like elements
                if 50 < w < 300 and 30 < h < 100:
                    button = UIElement(
                        element_type="button",
                        confidence=0.7,
                        bounding_box=(x, y, w, h),
                        attributes={"area": w * h, "aspect_ratio": w / h}
                    )
                    buttons.append(button)
            
        except Exception as e:
            logger.error(f"Error detecting buttons: {e}")
        
        return buttons
    
    async def _detect_text_elements(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect text elements using OCR."""
        
        text_elements = []
        
        try:
            # Use pytesseract for OCR
            import pytesseract
            
            # Get text regions
            data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        text_element = UIElement(
                            element_type="text",
                            confidence=data['conf'][i] / 100.0,
                            bounding_box=(x, y, w, h),
                            text=text,
                            attributes={"confidence": data['conf'][i]}
                        )
                        text_elements.append(text_element)
            
        except Exception as e:
            logger.error(f"Error detecting text elements: {e}")
        
        return text_elements
    
    async def _detect_input_fields(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect input field elements."""
        
        input_fields = []
        
        try:
            # Detect rectangular shapes that could be input fields
            # Input fields typically have specific characteristics
            
            # Use edge detection to find rectangular shapes
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio for input fields
                if 100 < w < 500 and 20 < h < 60 and w / h > 2:
                    input_field = UIElement(
                        element_type="input",
                        confidence=0.6,
                        bounding_box=(x, y, w, h),
                        attributes={"area": w * h, "aspect_ratio": w / h}
                    )
                    input_fields.append(input_field)
            
        except Exception as e:
            logger.error(f"Error detecting input fields: {e}")
        
        return input_fields
    
    async def _find_alternative_locator(self, failed_locator: str, 
                                      detected_elements: List[UIElement],
                                      context: Dict[str, Any]) -> HealingResult:
        """Find alternative locators for a failed element."""
        
        try:
            # Parse the failed locator to understand what we're looking for
            target_element = self._parse_locator(failed_locator)
            
            # Find similar elements using various matching strategies
            candidates = []
            
            # Strategy 1: Text-based matching
            if target_element.get('text'):
                text_candidates = await self._find_by_text(
                    target_element['text'], detected_elements
                )
                candidates.extend(text_candidates)
            
            # Strategy 2: Element type matching
            if target_element.get('type'):
                type_candidates = await self._find_by_type(
                    target_element['type'], detected_elements
                )
                candidates.extend(type_candidates)
            
            # Strategy 3: Position-based matching
            if target_element.get('position'):
                position_candidates = await self._find_by_position(
                    target_element['position'], detected_elements
                )
                candidates.extend(position_candidates)
            
            # Select the best candidate
            if candidates:
                best_candidate = max(candidates, key=lambda x: x['confidence'])
                
                # Generate new locator
                new_locator = await self._generate_locator(best_candidate['element'])
                
                return HealingResult(
                    success=True,
                    new_locator=new_locator,
                    confidence=best_candidate['confidence']
                )
            else:
                return HealingResult(
                    success=False,
                    error_message="No alternative locators found"
                )
                
        except Exception as e:
            logger.error(f"Error finding alternative locator: {e}")
            return HealingResult(
                success=False,
                error_message=str(e)
            )
    
    def _parse_locator(self, locator: str) -> Dict[str, Any]:
        """Parse a locator string to extract element properties."""
        
        parsed = {}
        
        try:
            if 'text=' in locator:
                # Extract text content
                start = locator.find('text=') + 5
                end = locator.find('"', start)
                if end == -1:
                    end = locator.find("'", start)
                if end > start:
                    parsed['text'] = locator[start:end]
            
            if 'class=' in locator:
                # Extract class information
                start = locator.find('class=') + 6
                end = locator.find('"', start)
                if end == -1:
                    end = locator.find("'", start)
                if end > start:
                    parsed['class'] = locator[start:end]
            
            if 'id=' in locator:
                # Extract ID information
                start = locator.find('id=') + 3
                end = locator.find('"', start)
                if end == -1:
                    end = locator.find("'", start)
                if end > start:
                    parsed['id'] = locator[start:end]
            
            # Determine element type from locator
            if 'button' in locator.lower():
                parsed['type'] = 'button'
            elif 'input' in locator.lower():
                parsed['type'] = 'input'
            elif 'text' in locator.lower():
                parsed['type'] = 'text'
            
        except Exception as e:
            logger.error(f"Error parsing locator: {e}")
        
        return parsed
    
    async def _find_by_text(self, target_text: str, 
                          elements: List[UIElement]) -> List[Dict[str, Any]]:
        """Find elements by text similarity."""
        
        candidates = []
        
        for element in elements:
            if element.text:
                # Calculate text similarity
                similarity = self._calculate_text_similarity(target_text, element.text)
                
                if similarity > 0.6:  # Similarity threshold
                    candidates.append({
                        'element': element,
                        'confidence': similarity,
                        'strategy': 'text_match'
                    })
        
        return candidates
    
    async def _find_by_type(self, target_type: str, 
                          elements: List[UIElement]) -> List[Dict[str, Any]]:
        """Find elements by type matching."""
        
        candidates = []
        
        for element in elements:
            if element.element_type == target_type:
                candidates.append({
                    'element': element,
                    'confidence': element.confidence,
                    'strategy': 'type_match'
                })
        
        return candidates
    
    async def _find_by_position(self, target_position: Dict[str, int], 
                              elements: List[UIElement]) -> List[Dict[str, Any]]:
        """Find elements by position similarity."""
        
        candidates = []
        
        for element in elements:
            x, y, w, h = element.bounding_box
            
            # Calculate position similarity
            position_similarity = self._calculate_position_similarity(
                target_position, {'x': x, 'y': y, 'width': w, 'height': h}
            )
            
            if position_similarity > 0.7:
                candidates.append({
                    'element': element,
                    'confidence': position_similarity,
                    'strategy': 'position_match'
                })
        
        return candidates
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple string matching."""
        
        # Simple implementation - in practice, you'd use more sophisticated
        # text similarity algorithms like Levenshtein distance or embeddings
        
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        if text1_lower == text2_lower:
            return 1.0
        
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.8
        
        # Calculate character overlap
        common_chars = set(text1_lower) & set(text2_lower)
        total_chars = set(text1_lower) | set(text2_lower)
        
        if total_chars:
            return len(common_chars) / len(total_chars)
        
        return 0.0
    
    def _calculate_position_similarity(self, pos1: Dict[str, int], 
                                     pos2: Dict[str, int]) -> float:
        """Calculate position similarity between two elements."""
        
        # Calculate center point distance
        center1 = (pos1['x'] + pos1['width'] / 2, pos1['y'] + pos1['height'] / 2)
        center2 = (pos2['x'] + pos2['width'] / 2, pos2['y'] + pos2['height'] / 2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Normalize distance (assuming screen size of 1000x1000)
        max_distance = np.sqrt(1000**2 + 1000**2)
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, similarity)
    
    async def _generate_locator(self, element: UIElement) -> str:
        """Generate a new locator for the detected element."""
        
        if element.text:
            # Generate text-based locator
            return f'//*[@text="{element.text}"]'
        elif element.element_type == 'button':
            # Generate button locator
            x, y, w, h = element.bounding_box
            return f'//button[contains(@class, "button") and @x="{x}" and @y="{y}"]'
        elif element.element_type == 'input':
            # Generate input locator
            x, y, w, h = element.bounding_box
            return f'//input[@x="{x}" and @y="{y}"]'
        else:
            # Generate generic locator
            x, y, w, h = element.bounding_box
            return f'//*[@x="{x}" and @y="{y}"]'


class TemplateMatcher:
    """Template matching for UI element detection."""
    
    def __init__(self):
        self.templates = {}
    
    async def match_template(self, screenshot: np.ndarray, 
                           template_path: str) -> List[Tuple[int, int, float]]:
        """Match a template against the screenshot."""
        
        try:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                return []
            
            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.8
            locations = np.where(result >= threshold)
            
            matches = []
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                matches.append((pt[0], pt[1], confidence))
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return []
