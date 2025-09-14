"""Code diff analysis for risk assessment and test prioritization."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import difflib

logger = logging.getLogger(__name__)


@dataclass
class CodeChange:
    """Represents a single code change."""
    file_path: str
    change_type: str  # added, modified, deleted
    lines_added: int
    lines_deleted: int
    complexity_change: float
    affected_functions: List[str]
    risk_indicators: List[str]


@dataclass
class DiffAnalysis:
    """Analysis results of a code diff."""
    commit_hash: str
    commit_message: str
    total_changes: int
    files_changed: int
    risk_score: float
    changes: List[CodeChange]
    recommendations: List[str]
    affected_components: List[str]


class CodeDiffAnalyzer:
    """Analyzes code diffs to assess risk and determine test priorities."""
    
    def __init__(self):
        self.risk_patterns = self._load_risk_patterns()
        self.component_mappings = self._load_component_mappings()
    
    def analyze_diff(self, diff_content: str, commit_metadata: Dict[str, Any]) -> DiffAnalysis:
        """Analyze a code diff and return risk assessment."""
        
        try:
            # Parse the diff content
            changes = self._parse_diff(diff_content)
            
            # Calculate risk indicators
            risk_indicators = self._identify_risk_indicators(diff_content, changes)
            
            # Calculate overall risk score
            risk_score = self._calculate_diff_risk_score(changes, risk_indicators)
            
            # Extract affected components
            affected_components = self._extract_affected_components(changes)
            
            # Generate recommendations
            recommendations = self._generate_diff_recommendations(changes, risk_score, risk_indicators)
            
            analysis = DiffAnalysis(
                commit_hash=commit_metadata.get('hash', 'unknown'),
                commit_message=commit_metadata.get('message', ''),
                total_changes=sum(c.lines_added + c.lines_deleted for c in changes),
                files_changed=len(set(c.file_path for c in changes)),
                risk_score=risk_score,
                changes=changes,
                recommendations=recommendations,
                affected_components=affected_components
            )
            
            logger.info(f"Analyzed diff with {len(changes)} changes, risk score: {risk_score:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing diff: {e}")
            return DiffAnalysis(
                commit_hash=commit_metadata.get('hash', 'unknown'),
                commit_message=commit_metadata.get('message', ''),
                total_changes=0,
                files_changed=0,
                risk_score=0.5,
                changes=[],
                recommendations=["Error analyzing diff"],
                affected_components=[]
            )
    
    def _parse_diff(self, diff_content: str) -> List[CodeChange]:
        """Parse diff content into individual code changes."""
        
        changes = []
        current_file = None
        current_change = None
        
        for line in diff_content.split('\n'):
            # File header
            if line.startswith('diff --git'):
                if current_change:
                    changes.append(current_change)
                
                # Extract file path
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3].replace('b/', '')
                    current_file = file_path
                    current_change = CodeChange(
                        file_path=file_path,
                        change_type='modified',
                        lines_added=0,
                        lines_deleted=0,
                        complexity_change=0.0,
                        affected_functions=[],
                        risk_indicators=[]
                    )
            
            # Change statistics
            elif line.startswith('@@'):
                # Parse hunk header for line numbers
                pass
            
            # Added line
            elif line.startswith('+') and not line.startswith('+++'):
                if current_change:
                    current_change.lines_added += 1
                    current_change.change_type = 'added' if current_change.lines_deleted == 0 else 'modified'
            
            # Deleted line
            elif line.startswith('-') and not line.startswith('---'):
                if current_change:
                    current_change.lines_deleted += 1
                    current_change.change_type = 'deleted' if current_change.lines_added == 0 else 'modified'
        
        # Add the last change
        if current_change:
            changes.append(current_change)
        
        return changes
    
    def _identify_risk_indicators(self, diff_content: str, changes: List[CodeChange]) -> List[str]:
        """Identify risk indicators in the diff."""
        
        risk_indicators = []
        
        # Check for high-risk patterns
        for pattern_name, pattern in self.risk_patterns.items():
            if pattern.search(diff_content):
                risk_indicators.append(pattern_name)
        
        # Check individual changes for risk factors
        for change in changes:
            # Large changes
            if change.lines_added + change.lines_deleted > 100:
                risk_indicators.append('large_change')
            
            # Many deletions
            if change.lines_deleted > 50:
                risk_indicators.append('many_deletions')
            
            # Critical files
            if any(critical in change.file_path for critical in ['auth', 'payment', 'security']):
                risk_indicators.append('critical_file')
        
        return list(set(risk_indicators))  # Remove duplicates
    
    def _calculate_diff_risk_score(self, changes: List[CodeChange], 
                                 risk_indicators: List[str]) -> float:
        """Calculate overall risk score for the diff."""
        
        base_score = 0.0
        
        # Risk from change size
        total_lines = sum(c.lines_added + c.lines_deleted for c in changes)
        size_risk = min(total_lines / 200.0, 1.0)  # Normalize to 200 lines
        base_score += size_risk * 0.3
        
        # Risk from number of files
        file_risk = min(len(changes) / 10.0, 1.0)  # Normalize to 10 files
        base_score += file_risk * 0.2
        
        # Risk from indicators
        indicator_weights = {
            'sql_injection': 0.9,
            'xss_vulnerability': 0.9,
            'authentication_bypass': 0.95,
            'unsafe_deserialization': 0.85,
            'path_traversal': 0.8,
            'command_injection': 0.9,
            'large_change': 0.4,
            'many_deletions': 0.5,
            'critical_file': 0.7,
            'complex_logic': 0.6,
            'error_handling': 0.3
        }
        
        for indicator in risk_indicators:
            weight = indicator_weights.get(indicator, 0.3)
            base_score += weight * 0.5  # Each indicator adds up to 0.5
        
        return min(base_score, 1.0)
    
    def _extract_affected_components(self, changes: List[CodeChange]) -> List[str]:
        """Extract affected components from file paths."""
        
        components = set()
        
        for change in changes:
            file_path = change.file_path
            
            # Map file paths to components
            for component, patterns in self.component_mappings.items():
                if any(pattern in file_path for pattern in patterns):
                    components.add(component)
                    break
        
        return list(components)
    
    def _generate_diff_recommendations(self, changes: List[CodeChange], 
                                     risk_score: float,
                                     risk_indicators: List[str]) -> List[str]:
        """Generate recommendations based on diff analysis."""
        
        recommendations = []
        
        # High risk score recommendations
        if risk_score > 0.8:
            recommendations.append("High risk changes detected. Run comprehensive test suite.")
        elif risk_score > 0.6:
            recommendations.append("Medium-high risk changes. Run regression tests.")
        elif risk_score > 0.4:
            recommendations.append("Medium risk changes. Run smoke tests.")
        
        # Specific indicator recommendations
        if 'sql_injection' in risk_indicators:
            recommendations.append("SQL injection risk detected. Add security tests.")
        
        if 'authentication_bypass' in risk_indicators:
            recommendations.append("Authentication changes detected. Test auth flows thoroughly.")
        
        if 'critical_file' in risk_indicators:
            recommendations.append("Critical files modified. High priority testing required.")
        
        if 'large_change' in risk_indicators:
            recommendations.append("Large changes detected. Consider breaking into smaller commits.")
        
        if 'many_deletions' in risk_indicators:
            recommendations.append("Many deletions detected. Ensure no functionality is lost.")
        
        # File-specific recommendations
        critical_files = [c for c in changes if any(critical in c.file_path for critical in ['auth', 'payment', 'security'])]
        if critical_files:
            recommendations.append(f"Critical files modified: {', '.join(c.file_path for c in critical_files)}")
        
        if not recommendations:
            recommendations.append("Low risk changes. Standard testing should suffice.")
        
        return recommendations
    
    def _load_risk_patterns(self) -> Dict[str, re.Pattern]:
        """Load regex patterns for identifying risk indicators."""
        
        patterns = {
            'sql_injection': re.compile(r'(?i)(select|insert|update|delete).*\+.*["\']', re.MULTILINE),
            'xss_vulnerability': re.compile(r'(?i)innerHTML|document\.write|eval\s*\(', re.MULTILINE),
            'authentication_bypass': re.compile(r'(?i)(auth|login|password|token).*==.*true', re.MULTILINE),
            'unsafe_deserialization': re.compile(r'(?i)(pickle|yaml\.load|eval)', re.MULTILINE),
            'path_traversal': re.compile(r'(?i)\.\./|\.\.\\\\', re.MULTILINE),
            'command_injection': re.compile(r'(?i)(os\.system|subprocess|exec|shell)', re.MULTILINE),
            'complex_logic': re.compile(r'(?i)(if.*if.*if|for.*for.*for)', re.MULTILINE),
            'error_handling': re.compile(r'(?i)(try.*except|catch)', re.MULTILINE)
        }
        
        return patterns
    
    def _load_component_mappings(self) -> Dict[str, List[str]]:
        """Load mappings from file patterns to components."""
        
        mappings = {
            'authentication': ['auth', 'login', 'user', 'session', 'token'],
            'payment': ['payment', 'billing', 'transaction', 'stripe', 'paypal'],
            'database': ['model', 'schema', 'migration', 'query'],
            'api': ['api', 'endpoint', 'route', 'controller'],
            'ui': ['component', 'view', 'template', 'css', 'js'],
            'security': ['security', 'encrypt', 'hash', 'permission'],
            'notification': ['email', 'sms', 'push', 'notification'],
            'analytics': ['tracking', 'metric', 'analytics', 'log']
        }
        
        return mappings
    
    def compare_diffs(self, diff1: str, diff2: str) -> Dict[str, Any]:
        """Compare two diffs to find similarities and differences."""
        
        try:
            # Parse both diffs
            changes1 = self._parse_diff(diff1)
            changes2 = self._parse_diff(diff2)
            
            # Find common files
            files1 = set(c.file_path for c in changes1)
            files2 = set(c.file_path for c in changes2)
            common_files = files1 & files2
            
            # Find unique files
            unique_to_1 = files1 - files2
            unique_to_2 = files2 - files1
            
            # Calculate similarity score
            similarity = len(common_files) / max(len(files1), len(files2)) if max(len(files1), len(files2)) > 0 else 0
            
            return {
                'common_files': list(common_files),
                'unique_to_first': list(unique_to_1),
                'unique_to_second': list(unique_to_2),
                'similarity_score': similarity,
                'total_files_1': len(files1),
                'total_files_2': len(files2)
            }
            
        except Exception as e:
            logger.error(f"Error comparing diffs: {e}")
            return {
                'common_files': [],
                'unique_to_first': [],
                'unique_to_second': [],
                'similarity_score': 0.0,
                'total_files_1': 0,
                'total_files_2': 0
            }
