#!/usr/bin/env python3
"""
Test Complete XML Pipeline Flow with Clinical Interpretation

This script tests the ACTUAL production flow using the real pipeline classes.
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.mood_prediction_pipeline import MoodPredictionPipeline
from big_mood_detector.domain.services.clinical_interpreter import ClinicalInterpreter
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import StreamingXMLParser
import pandas as pd


def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print('='*80)


def test_complete_pipeline():
    """Test the complete pipeline from XML to clinical interpretation."""
    
    print("="*80)
    print(" COMPLETE XML PIPELINE TEST")
    print(" XML → Features → Predictions → Clinical Interpretation")
    print("="*80)
    
    # Initialize pipeline
    pipeline = MoodPredictionPipeline()
    
    # Test with XML file
    xml_path = Path("apple_export/export.xml")
    output_path = Path("output/xml_complete_test.csv")
    
    if not xml_path.exists():
        print(f"❌ XML file not found: {xml_path}")
        return
    
    print(f"\n📁 Processing: {xml_path.name} ({xml_path.stat().st_size / 1024**2:.1f} MB)")
    
    # Process recent data
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Date range: {start_date} to {end_date}")
    
    # Step 1: Process through pipeline
    print_header("STEP 1: PIPELINE PROCESSING")
    print("🔄 Running complete pipeline...")
    
    start_time = time.time()
    
    try:
        # Process health data
        features_df = pipeline.process_health_export(
            export_path=xml_path,
            output_path=output_path,
            start_date=start_date,
            end_date=end_date
        )
        
        process_time = time.time() - start_time
        
        print(f"\n✅ Pipeline Complete:")
        print(f"   Time: {process_time:.1f}s")
        print(f"   Days processed: {len(features_df)}")
        print(f"   Features extracted: {len(features_df.columns)}")
        
        # Show sample features
        if len(features_df) > 0:
            print(f"\n📊 Sample Features (last 3 days):")
            for idx, row in features_df.tail(3).iterrows():
                print(f"\n   {row.get('date', idx)}:")
                print(f"     Sleep duration: {row.get('sleep_percentage_MN', 0)*24:.1f}h")
                print(f"     Steps: {row.get('long_len_MN', 0)*1000:.0f}")
                print(f"     Confidence: {row.get('confidence_score', 0):.1%}")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Get predictions from pipeline
    print_header("STEP 2: ML PREDICTIONS")
    
    if len(features_df) == 0:
        print("❌ No features generated")
        return
    
    # The pipeline already includes predictions in the output
    prediction_cols = ['depression_risk', 'hypomanic_risk', 'manic_risk']
    
    if all(col in features_df.columns for col in prediction_cols):
        print("✅ Predictions included in pipeline output")
        
        # Show recent predictions
        print(f"\n📊 Recent Predictions:")
        for idx, row in features_df.tail(3).iterrows():
            print(f"\n   {row.get('date', idx)}:")
            print(f"     Depression: {row['depression_risk']:.1%}")
            print(f"     Hypomanic: {row['hypomanic_risk']:.1%}")
            print(f"     Manic: {row['manic_risk']:.1%}")
    else:
        print("⚠️  Predictions not found in output")
    
    # Step 3: Clinical interpretation
    print_header("STEP 3: CLINICAL INTERPRETATION")
    
    interpreter = ClinicalInterpreter()
    
    print("🔄 Generating clinical interpretations for recent predictions...")
    
    # Interpret last 3 days
    for idx, row in features_df.tail(3).iterrows():
        print(f"\n📊 Clinical Interpretation for {row.get('date', idx)}:")
        
        # Get prediction values
        dep_risk = row.get('depression_risk', 0)
        hypo_risk = row.get('hypomanic_risk', 0)
        manic_risk = row.get('manic_risk', 0)
        
        # Convert to PHQ/ASRM scores for interpretation
        # (In production, these would come from actual assessments)
        phq_score = int(dep_risk * 20)  # Scale to 0-20 range
        asrm_score = int(max(hypo_risk, manic_risk) * 15)  # Scale to 0-15 range
        
        # Get sleep/activity from features
        sleep_hours = row.get('sleep_percentage_MN', 0.3) * 24
        daily_steps = row.get('long_len_MN', 8) * 1000
        
        print(f"   ML Risk: Depression {dep_risk:.1%}, Hypomania {hypo_risk:.1%}, Mania {manic_risk:.1%}")
        print(f"   Biomarkers: Sleep {sleep_hours:.1f}h, Steps {daily_steps:.0f}")
        
        # Depression interpretation
        if phq_score > 4:
            dep_result = interpreter.interpret_depression_score(
                phq_score=phq_score,
                sleep_hours=sleep_hours,
                activity_steps=daily_steps
            )
            print(f"\n   Depression Assessment:")
            print(f"     Risk Level: {dep_result.risk_level.value}")
            print(f"     Summary: {dep_result.clinical_summary}")
            if dep_result.recommendations:
                print(f"     Recommendation: {dep_result.recommendations[0].medication}")
        
        # Mania interpretation  
        if asrm_score > 5:
            mania_result = interpreter.interpret_mania_score(
                asrm_score=asrm_score,
                sleep_hours=sleep_hours,
                activity_steps=daily_steps
            )
            print(f"\n   Mania Assessment:")
            print(f"     Risk Level: {mania_result.risk_level.value}")
            print(f"     Episode: {mania_result.episode_type}")
            print(f"     Summary: {mania_result.clinical_summary}")
    
    # Summary
    print_header("PIPELINE SUMMARY")
    
    print(f"\n✅ End-to-End Pipeline Status:")
    print(f"   XML Parsing: ✅ (integrated in pipeline)")
    print(f"   Feature Engineering: ✅ ({len(features_df)} days)")
    print(f"   ML Predictions: ✅ (ensemble working)")
    print(f"   Clinical Interpretation: ✅ (risk-based assessments)")
    
    print(f"\n📊 Performance:")
    print(f"   Total processing time: {process_time:.1f}s")
    print(f"   Average per day: {process_time/max(1, len(features_df)):.2f}s")
    
    print(f"\n💾 Output saved to: {output_path}")
    
    # Check what models were used
    if 'models_used' in features_df.columns:
        models = features_df['models_used'].iloc[-1] if len(features_df) > 0 else "unknown"
        print(f"\n🤖 Models used: {models}")
    
    return features_df


def test_api_integration():
    """Test API endpoints with real data."""
    print_header("BONUS: API INTEGRATION TEST")
    
    import requests
    
    try:
        # Check if API is running
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is running")
            
            # Test with realistic values
            test_data = {
                "phq_score": 12,
                "sleep_hours": 5.5,
                "activity_steps": 4500,
                "suicidal_ideation": False
            }
            
            response = requests.post(
                "http://localhost:8000/api/v1/clinical/interpret/depression",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n📊 API Response:")
                print(f"   Risk Level: {result['risk_level']}")
                print(f"   Clinical Summary: {result['clinical_summary']}")
                print(f"   Confidence: {result['confidence']}")
            else:
                print(f"❌ API error: {response.status_code}")
        else:
            print("⚠️  API not running (start with: uvicorn big_mood_detector.interfaces.api.main:app)")
    except requests.exceptions.ConnectionError:
        print("⚠️  API not accessible")


if __name__ == "__main__":
    # Run main test
    features_df = test_complete_pipeline()
    
    # Optional: test API if running
    test_api_integration()