#!/bin/bash
# Script to help move NHANES files to the correct location

echo "NHANES Data Setup Script"
echo "========================"
echo ""
echo "Please move your downloaded NHANES files to:"
echo "  /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/nhanes/2013-2014/"
echo ""
echo "Expected files:"
echo "  - PAXHD_H.xpt    (Physical Activity Monitor - Header)"
echo "  - PAXMIN_H.xpt   (Physical Activity Monitor - Minute data)"
echo "  - DPQ_H.xpt      (Depression Questionnaire)"
echo "  - RXQ_DRUG.xpt   (Prescription Drug Information)"
echo "  - RXQ_RX_H.xpt   (Prescription Medications)"
echo ""
echo "You can move them with:"
echo "  mv /Users/ray/Downloads/*.xpt /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/nhanes/2013-2014/"
echo ""
echo "Current contents of 2013-2014 directory:"
ls -la /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/nhanes/2013-2014/