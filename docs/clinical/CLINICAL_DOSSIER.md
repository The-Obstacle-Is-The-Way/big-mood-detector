# Clinical Dossier: Bipolar Disorder Digital Biomarkers and Treatment Guidelines

This document serves as the single source of truth for all clinical decision-making thresholds, DSM-5 criteria, and implementation guidelines for the big-mood-detector system.

## Table of Contents

1. [DSM-5 Diagnostic Criteria](#dsm-5-diagnostic-criteria)
2. [Clinical Assessment Scales and Thresholds](#clinical-assessment-scales-and-thresholds)
3. [Risk Stratification Levels](#risk-stratification-levels)
4. [Digital Biomarker Thresholds](#digital-biomarker-thresholds)
5. [Treatment Decision Rules](#treatment-decision-rules)
6. [Early Warning Signs and Intervention Triggers](#early-warning-signs-and-intervention-triggers)
7. [Temporal Requirements](#temporal-requirements)
8. [Implementation Notes](#implementation-notes)

## DSM-5 Diagnostic Criteria

### Manic Episode (DSM-5)
**Duration**: ≥7 days (or any duration if hospitalization required)

**Core Criteria**: Abnormally elevated, expansive, or irritable mood AND increased goal-directed activity or energy

**Symptoms** (≥3 required, or 4 if mood is only irritable):
- Inflated self-esteem or grandiosity
- Decreased need for sleep (e.g., feels rested after 3 hours)
- More talkative than usual or pressure to keep talking
- Flight of ideas or subjective experience of racing thoughts
- Distractibility
- Increase in goal-directed activity or psychomotor agitation
- Excessive involvement in risky activities

**Source**: CANMAT Guidelines, DSM-5-TR

### Hypomanic Episode (DSM-5)
**Duration**: ≥4 consecutive days

**Core Criteria**: Same as manic episode but:
- Not severe enough to cause marked impairment
- No psychotic features
- No hospitalization required

**Source**: CANMAT Guidelines, DSM-5-TR

### Major Depressive Episode (DSM-5)
**Duration**: ≥2 weeks

**Core Criteria**: ≥5 symptoms present during same 2-week period, including at least one of:
1. Depressed mood
2. Markedly diminished interest or pleasure (anhedonia)

**Additional Symptoms**:
- Significant weight loss/gain (>5% body weight) or appetite change
- Insomnia or hypersomnia
- Psychomotor agitation or retardation
- Fatigue or loss of energy
- Feelings of worthlessness or excessive guilt
- Diminished concentration or indecisiveness
- Recurrent thoughts of death or suicidal ideation

**Source**: CANMAT Guidelines, DSM-5-TR

### Mixed Features Specifier (DSM-5)
**Manic/Hypomanic Episode with Mixed Features**: Full criteria for manic/hypomanic episode + ≥3 depressive symptoms

**Depressive Episode with Mixed Features**: Full criteria for major depressive episode + ≥3 manic/hypomanic symptoms

**Note**: Overlapping symptoms (distractibility, irritability, psychomotor agitation) are excluded

**Source**: CANMAT Bipolar Guidelines

## Clinical Assessment Scales and Thresholds

### Depression Scales

#### PHQ-8 (Patient Health Questionnaire-8)
- **Cutoff for probable depression**: ≥10
- **Range**: 0-24
- **Use**: Screening and monitoring in digital phenotyping studies
- **Source**: Fitbit-bipolar-mood study (Lipschitz et al.)

#### PHQ-9 (Patient Health Questionnaire-9)
- **Cutoff for depression**: ≥10
- **Mild depression**: 5-9
- **Moderate depression**: 10-14
- **Moderately severe**: 15-19
- **Severe depression**: 20-27
- **Use**: Standard screening in primary care and digital monitoring
- **Source**: CANMAT MDD Guidelines, PAT study

#### MADRS (Montgomery-Åsberg Depression Rating Scale)
- **Remission**: ≤10
- **Mild depression**: 11-19
- **Moderate depression**: 20-34
- **Severe depression**: ≥35
- **Source**: CANMAT Guidelines, xgboost-mood study

#### HAM-D (Hamilton Depression Rating Scale)
- **Remission**: ≤7
- **Source**: CANMAT MDD Guidelines

### Mania/Hypomania Scales

#### ASRM (Altman Self-Rating Mania Scale)
- **Cutoff for probable manic/hypomanic episode**: ≥6
- **Range**: 0-20
- **Use**: Self-report screening in digital phenotyping
- **Source**: Fitbit-bipolar-mood study (Lipschitz et al.)

#### YMRS (Young Mania Rating Scale)
- **Remission**: <8
- **Mild mania**: 8-14
- **Moderate mania**: 15-25
- **Severe mania**: >25
- **Source**: CANMAT Guidelines, xgboost-mood study

### Circadian and Sleep Assessment

#### Composite Scale of Morningness (CSM)
- **Mean in BD patients**: 29.5 (±7.50)
- **Use**: Assess circadian preference
- **Source**: xgboost-mood study

#### Global Seasonality Score (GSS)
- **Mean in BD patients**: 5.86 (±4.83)
- **Screening cutoff for seasonality**: ≥11
- **Source**: xgboost-mood study

## Risk Stratification Levels

### Depression Risk Levels
Based on digital biomarkers and clinical assessment:

1. **Low Risk**
   - PHQ-8/9 <10
   - Normal sleep patterns (7-9 hours)
   - Regular circadian rhythm
   - Stable activity patterns

2. **Moderate Risk**
   - PHQ-8/9 10-14
   - Sleep duration 5-7 or 9-11 hours
   - Mild circadian disruption
   - Decreased activity levels

3. **High Risk**
   - PHQ-8/9 15-19
   - Sleep duration <5 or >11 hours
   - Significant circadian disruption
   - Markedly reduced activity

4. **Critical Risk**
   - PHQ-8/9 ≥20
   - Severe sleep disruption
   - Suicidal ideation present
   - Requires immediate intervention

### Mania/Hypomania Risk Levels

1. **Low Risk**
   - ASRM <6
   - Normal sleep (6-9 hours)
   - Stable activity patterns

2. **Moderate Risk**
   - ASRM 6-10
   - Sleep 4-6 hours
   - Increased activity/speech patterns

3. **High Risk**
   - ASRM 11-15
   - Sleep <4 hours
   - Significantly elevated activity

4. **Critical Risk**
   - ASRM >15
   - Sleep <3 hours for multiple days
   - Dangerous/risky behaviors
   - Psychotic features

## Digital Biomarker Thresholds

### Sleep Duration
- **Normal range**: 7-9 hours
- **Short sleep (risk factor)**: <6 hours
- **Long sleep (risk factor)**: >9 hours
- **Critical short sleep**: <3 hours (mania indicator)
- **Critical long sleep**: >12 hours (depression indicator)
- **Source**: Fitbit-bipolar-mood study, xgboost-mood study

### Sleep Efficiency
- **Normal**: >85%
- **Poor**: <85%
- **Mean in BD**: 92.27% (±7.13%)
- **Source**: Fitbit-bipolar-mood study

### Sleep Timing
- **Median bedtime in BD**: 11 PM (coded as 5.53 ±1.62 on 6PM-10AM scale)
- **Late bedtime (risk)**: After 2 AM
- **Variable bedtime (risk)**: >2 hour variation
- **Source**: Fitbit-bipolar-mood study

### Sleep Window Merging
- **Threshold**: 3.75 hours (225 minutes)
- **Rationale**: Based on Seoul National Study, sleep episodes within 3.75 hours should be merged as single sleep period
- **Source**: xgboost-mood study, Seoul National Study

### Physical Activity
- **Mean daily steps in BD**: 6,631 (±3,585)
- **Low activity (depression risk)**: <5,000 steps/day
- **High activity (mania risk)**: >15,000 steps/day
- **Very active minutes mean**: 14.92 (±17.50) minutes/day
- **Source**: Fitbit-bipolar-mood study

### Heart Rate
- **Mean daily HR**: 78.42 (±7.62) bpm
- **Mean resting HR**: 69.38 (±7.96) bpm
- **Elevated HR (mania indicator)**: >90 bpm resting
- **Source**: Fitbit-bipolar-mood study

### Circadian Rhythm Metrics

#### Circadian Phase (DLMO - Dim Light Melatonin Onset)
- **Calculation**: CBTmin - 7 hours
- **Delayed phase**: Associated with depressive episodes
- **Advanced phase**: Associated with manic episodes
- **Source**: xgboost-mood study, melatonin-math study

#### Circadian Amplitude
- **Low amplitude**: Associated with mood instability
- **Source**: xgboost-mood study

#### Interdaily Stability (IS)
- **Range**: 0-1 (higher = more stable)
- **Low IS (<0.5)**: Risk factor for mood episodes
- **Source**: PAT study, circadian literature

#### Intradaily Variability (IV)
- **Range**: 0-2 (lower = more stable)
- **High IV (>1)**: Risk factor for mood episodes
- **Source**: PAT study, circadian literature

### Activity Sequence Features (1440 minutes/day)
- **Pattern recognition**: Minute-level activity sequences
- **Use**: Input for PAT (Pretrained Actigraphy Transformer) model
- **Processing**: 18-minute patches for transformer model
- **Source**: PAT study

## Treatment Decision Rules

### Pharmacological Interventions

#### For Acute Mania (First-Line)
- Lithium (if not already on therapeutic dose)
- Divalproex
- Atypical antipsychotics (quetiapine, olanzapine, aripiprazole)
- **Source**: CANMAT Guidelines, VA Guidelines

#### For Acute Depression (First-Line)
- Quetiapine monotherapy
- Lithium + antidepressant (with mood stabilizer coverage)
- Lamotrigine (if not rapid cycling)
- **Caution**: Antidepressant monotherapy contraindicated
- **Source**: CANMAT Guidelines, CDS-bipolar-LLM study

#### For Mixed Features
- **Mania + Mixed Features (Second-Line)**:
  - Asenapine
  - Cariprazine
  - Divalproex
  - Aripiprazole
- **Depression + Mixed Features (Second-Line)**:
  - Cariprazine
  - Lurasidone
- **Note**: No first-line treatments meet evidence threshold
- **Source**: CANMAT Guidelines

### Non-Pharmacological Interventions

#### Cognitive Behavioral Therapy (CBT)
- **Indication**: All phases of BD
- **Delivery**: Can be augmented with AI for real-time support
- **Focus**: Cognitive restructuring, sleep hygiene, activity scheduling
- **Source**: AI-Bipolar-Frontier study

#### Circadian Interventions
- **Bright light therapy**: For depressive episodes with delayed phase
- **Dark therapy**: For manic episodes
- **Sleep-wake scheduling**: Maintain regular bedtime ±30 minutes
- **Source**: CANMAT Guidelines, circadian studies

## Early Warning Signs and Intervention Triggers

### Depression Early Warning Signs
1. **Sleep changes**: Increase >2 hours from baseline
2. **Activity reduction**: >30% decrease in daily steps
3. **Circadian delay**: Bedtime shifting later by >1 hour
4. **Social withdrawal**: Detected via communication patterns
5. **Cognitive changes**: Slowed speech, increased negative language

**Intervention Trigger**: ≥3 warning signs for 3+ consecutive days

### Mania Early Warning Signs
1. **Sleep reduction**: <6 hours for 2+ nights
2. **Activity increase**: >50% increase in daily activity
3. **Circadian advance**: Earlier wake times by >1 hour
4. **Speech changes**: Increased rate, volume, or social media posting
5. **Goal-directed behavior**: Multiple new projects/activities

**Intervention Trigger**: ≥2 warning signs, especially sleep <4 hours

### Digital Monitoring Recommendations
- **Continuous monitoring**: Wearable devices for sleep/activity
- **Daily assessment**: Mood ratings via app (minimum 2x/day)
- **Weekly review**: Clinician dashboard review
- **Alert thresholds**: Automated alerts for critical values

## Temporal Requirements

### Episode Duration Criteria
- **Manic episode**: ≥7 days (or any duration if hospitalized)
- **Hypomanic episode**: ≥4 consecutive days
- **Major depressive episode**: ≥2 weeks
- **Mixed features**: Concurrent with primary episode
- **Rapid cycling**: ≥4 episodes in 12 months
- **Ultra-rapid cycling**: Episodes within days to weeks
- **Ultradian cycling**: Mood shifts within 24 hours

### Pattern Detection Windows
- **Minimum data for individual identification**: 30 days
- **Optimal data for mood prediction**: 60 days
- **Training window for personalized models**: 3-6 months
- **Circadian rhythm assessment**: 7-14 days minimum

### Response Assessment Timelines
- **Early improvement**: 20% symptom reduction by 2-4 weeks
- **Response**: 50% symptom reduction
- **Remission**: 
  - Depression: MADRS ≤10 or PHQ-9 ≤4 for ≥2 months
  - Mania: YMRS <8 for ≥2 months

## Implementation Notes

### Data Quality Requirements
1. **Wearable compliance**: >75% wear time
2. **Missing data threshold**: <20% for reliable analysis
3. **Minimum recording**: 144 minutes/day for GPS (2 min on, 18 min off cycles)
4. **Accelerometer sampling**: Continuous when possible

### Algorithm Selection
1. **For individual identification**: BiMM Forest (Binary Mixed Model)
   - Accounts for longitudinal data structure
   - Handles individual variability
   - AUC: 0.86 for depression, 0.85 for mania

2. **For mood prediction**: XGBoost with 36 features
   - Including circadian phase as top predictor
   - Sleep and activity features
   - AUC: 0.80 for depression, 0.98 for mania

3. **For activity pattern analysis**: PAT (Pretrained Actigraphy Transformer)
   - Processes week-long activity data
   - Uses 18-minute patches
   - Pretrained on 29,307 participants

### Clinical Integration
1. **Decision support**: AI augmentation with guidelines
   - Never replace clinical judgment
   - Provide evidence-based recommendations
   - Include rationale for suggestions

2. **Real-time monitoring**: Dashboard requirements
   - Visual display of trends
   - Alert system for threshold violations
   - Integration with EHR when possible

3. **Patient engagement**: Mobile app features
   - Self-report mood tracking
   - Educational content
   - Crisis resources
   - Medication reminders

### Ethical Considerations
1. **Privacy**: All data encrypted and de-identified
2. **Consent**: Explicit consent for continuous monitoring
3. **Bias monitoring**: Regular audits for demographic disparities
4. **Clinical oversight**: Human clinician remains decision-maker
5. **Transparency**: Patients can access all their data

### Special Populations

#### Bipolar II Considerations
- Often more depressive episodes
- Mixed features common (up to 57%)
- May have longer sleep duration during depression
- Hypomanic episodes may be subtle in digital data

#### Pediatric/Adolescent
- Sleep needs: 8-10 hours (vs 7-9 for adults)
- Higher activity baseline
- School schedule considerations
- Parental involvement in monitoring

#### Geriatric
- Sleep efficiency naturally decreases
- Lower activity thresholds
- Medication sensitivity
- Cognitive assessment integration

## References

1. **Fitbit-bipolar-mood study**: Lipschitz et al. Digital phenotyping in bipolar disorder using Fitbit data. AUC 0.86 for depression, 0.85 for mania detection.

2. **xgboost-mood study**: Lim et al. Accurately predicting mood episodes using sleep and circadian features. 36 features, circadian phase as top predictor.

3. **PAT study**: Pretrained Actigraphy Transformer for mental health. Week-long data processing, 29,307 participants.

4. **CANMAT Guidelines**: Canadian Network for Mood and Anxiety Treatments 2018 guidelines for bipolar disorder management.

5. **Mobile-footprint study**: Individual mobility patterns linked to mood, sleep, and brain connectivity. Minimum 30 days data for identification.

6. **AI-Bipolar-Frontier**: Role of AI in managing bipolar disorder, including real-time monitoring and personalized treatment.

7. **CDS-bipolar-LLM**: Clinical decision support using large language models for bipolar depression treatment selection.

---

*Last Updated: [Current Date]*
*Version: 1.0*
*This document should be reviewed and updated regularly as new evidence emerges.*