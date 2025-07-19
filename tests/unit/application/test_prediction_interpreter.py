"""
Test PredictionInterpreter - Clinical interpretation of ML predictions.

Following TDD approach to create a service that maps ML probabilities
to clinical diagnoses, risk levels, and treatment recommendations.
"""

import pytest

from big_mood_detector.application.services.prediction_interpreter import (
    PredictionInterpreter,
    ClinicalInterpretation,
)


class TestPredictionInterpreter:
    """Test the prediction interpreter service."""
    
    @pytest.fixture
    def interpreter(self):
        """Create PredictionInterpreter instance."""
        return PredictionInterpreter()
    
    def test_interpret_severe_depression(self, interpreter):
        """Test interpretation of high depression probability."""
        # Arrange
        ml_predictions = {
            "depression": 0.83,
            "mania": 0.05,
            "hypomania": 0.12,
        }
        
        # Act
        result = interpreter.interpret(ml_predictions)
        
        # Assert
        assert isinstance(result, ClinicalInterpretation)
        assert result.primary_diagnosis == "Severe Depressive Episode"
        assert result.risk_level == "high"
        assert result.confidence >= 0.8
        assert "depression" in result.clinical_notes[0].lower()
        assert len(result.recommendations) > 0
        assert any("urgent" in r.lower() or "immediate" in r.lower() 
                  for r in result.recommendations)
    
    def test_interpret_hypomania(self, interpreter):
        """Test interpretation of hypomanic episode."""
        # Arrange
        ml_predictions = {
            "depression": 0.15,
            "mania": 0.25,
            "hypomania": 0.75,
        }
        
        # Act
        result = interpreter.interpret(ml_predictions)
        
        # Assert
        assert result.primary_diagnosis == "Hypomanic Episode"
        assert result.risk_level == "moderate"
        assert result.confidence >= 0.7
        assert "hypomania" in result.clinical_notes[0].lower()
        
    def test_interpret_mixed_state(self, interpreter):
        """Test interpretation of mixed state (high depression + mania)."""
        # Arrange
        ml_predictions = {
            "depression": 0.72,
            "mania": 0.68,
            "hypomania": 0.45,
        }
        
        # Act
        result = interpreter.interpret(ml_predictions)
        
        # Assert
        assert result.primary_diagnosis == "Mixed Episode"
        assert result.risk_level == "critical"  # Mixed states are dangerous
        assert "mixed" in result.clinical_notes[0].lower()
        assert any("crisis" in r.lower() or "emergency" in r.lower() 
                  for r in result.recommendations)
    
    def test_interpret_euthymic_state(self, interpreter):
        """Test interpretation of stable/euthymic state."""
        # Arrange
        ml_predictions = {
            "depression": 0.12,
            "mania": 0.08,
            "hypomania": 0.15,
        }
        
        # Act
        result = interpreter.interpret(ml_predictions)
        
        # Assert
        assert result.primary_diagnosis == "Euthymic (Stable)"
        assert result.risk_level == "low"
        assert result.confidence >= 0.7
        assert any("stable" in note.lower() or "euthymic" in note.lower() 
                  for note in result.clinical_notes)
    
    def test_dsm5_compliance(self, interpreter):
        """Test that interpretations follow DSM-5 criteria."""
        # Test various threshold scenarios
        test_cases = [
            # (depression, mania, hypomania, expected_contains_dsm5)
            (0.75, 0.10, 0.15, True),  # Clear depression
            (0.10, 0.85, 0.20, True),  # Clear mania
            (0.15, 0.20, 0.70, True),  # Clear hypomania
        ]
        
        for dep, mania, hypo, should_have_dsm5 in test_cases:
            ml_predictions = {
                "depression": dep,
                "mania": mania,
                "hypomania": hypo,
            }
            
            result = interpreter.interpret(ml_predictions)
            
            has_dsm5_reference = any(
                "dsm-5" in note.lower() or "dsm5" in note.lower() 
                for note in result.clinical_notes
            )
            
            if should_have_dsm5:
                assert has_dsm5_reference, f"Missing DSM-5 reference for {ml_predictions}"
    
    def test_confidence_calculation(self, interpreter):
        """Test confidence score calculation based on prediction clarity."""
        # High confidence - clear single diagnosis
        clear_result = interpreter.interpret({
            "depression": 0.90,
            "mania": 0.05,
            "hypomania": 0.05,
        })
        assert clear_result.confidence >= 0.85
        
        # Low confidence - ambiguous predictions
        ambiguous_result = interpreter.interpret({
            "depression": 0.45,
            "mania": 0.40,
            "hypomania": 0.35,
        })
        assert ambiguous_result.confidence < 0.6
    
    def test_treatment_recommendations(self, interpreter):
        """Test that appropriate treatment recommendations are provided."""
        # Depression case
        dep_result = interpreter.interpret({
            "depression": 0.80,
            "mania": 0.10,
            "hypomania": 0.10,
        })
        assert len(dep_result.recommendations) >= 3
        assert any("medication" in r.lower() or "therapy" in r.lower() 
                  for r in dep_result.recommendations)
        
        # Mania case
        mania_result = interpreter.interpret({
            "depression": 0.10,
            "mania": 0.85,
            "hypomania": 0.15,
        })
        assert any("hospitalization" in r.lower() or "emergency" in r.lower() 
                  for r in mania_result.recommendations)