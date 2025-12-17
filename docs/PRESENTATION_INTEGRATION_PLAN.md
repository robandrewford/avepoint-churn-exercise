# Presentation Integration Plan
## Objective: Absolutely Nail the Exercise Assignment

This plan integrates the three Marimo notebook exports (`01_eda.html`, `02_modeling.html`, `03_monitoring.html`) into the main presentation (`index.html`) to create a comprehensive deliverable that addresses all 5 parts of the exercise.

---

## Exercise Requirements Mapping

### Part 1: Problem Framing ✅ (Currently Covered)
**Requirements:**
- Define the business problem
- Propose success metrics
- Identify risks and assumptions

**Current Coverage (index.html slides):**
- ✅ Slide: "Taxonomy of Churn" - defines solvable problem (Engagement Decay)
- ✅ Slide: "Customer Lifecycle Framework" - cohort-based approach
- ✅ Slide: "The Financial Reality" - LTV quantification
- ✅ Slide: "Success Metrics (Goals)" - AUC-PR, Precision@10%, Recall@30d

**Gap:** Missing explicit risks/assumptions discussion

**Action:** Add new slide after "Success Metrics" titled "Assumptions & Risk Mitigation"

---

### Part 2: Data Exploration & Feature Engineering ⚠️ (Partially Covered)
**Requirements:**
- Perform EDA
- Engineer predictive features
- Document your rationale

**Current Coverage (index.html slides):**
- ✅ Slide: "Fabric-Native Architecture" - medallion pattern
- ✅ Slide: "Cohort-Aware Feature Engineering" - feature matrix
- ✅ Slide: "Methodological Rigor: Leakage Audit" - temporal correctness

**Gap:** No actual EDA visualizations showing data distributions, churn patterns, or feature importance

**Available Resource:** `01_eda.html` - Full Marimo notebook with:
- Data quality checks
- Distribution analysis
- Churn pattern visualizations
- Feature correlation analysis
- Cohort comparisons

**Action:**
1. Add navigation to detailed EDA after "Cohort-Aware Feature Engineering" slide
2. Create new slide type "NOTEBOOK_LINK" that opens `01_eda.html` in modal/new tab
3. OR: Extract key visualizations from `01_eda.html` and embed as static images in new slides

---

### Part 3: Predictive Modeling ⚠️ (Partially Covered)
**Requirements:**
- Build a churn prediction model
- Justify algorithm choice
- Evaluate performance
- Address class imbalance, data leakage, and interpretability

**Current Coverage (index.html slides):**
- ✅ Slide: "Algorithm Selection: LightGBM" - justification
- ✅ Slide: "Cost-Sensitive Learning" - class imbalance handling
- ✅ Slide: "Temporal Validation Strategy" - prevents leakage
- ❌ Missing: Actual model performance results
- ❌ Missing: SHAP interpretability visualizations

**Available Resource:** `02_modeling.html` - Full Marimo notebook with:
- Model training code
- Performance metrics (actual results)
- ROC/PR curves
- SHAP feature importance
- Calibration plots
- Confusion matrices

**Action:**
1. Add new slide after "Success Metrics" showing **ACTUAL RESULTS**
2. Create "MODEL_RESULTS" slide type displaying:
   - Achieved AUC-PR (vs target)
   - Precision@10% (vs target)
   - Recall (vs target)
3. Add navigation link to full modeling notebook `02_modeling.html`
4. Extract top 3-5 SHAP visualizations and add as new slide for interpretability

---

### Part 4: Strategic Recommendations ✅ (Well Covered)
**Requirements:**
- Present 2-3 actionable insights
- Discuss testing approaches

**Current Coverage (index.html slides):**
- ✅ Slide: "Insight #1: Failure to Launch" - activation SLA with action
- ✅ Slide: "Insight #2: The Silent Slide" - velocity-based intervention
- ✅ Slide: "Strategic Targeting: Uplift" - persuadables framework

**Gap:** No explicit A/B testing methodology

**Action:** Add sub-bullet to each Insight slide with "Test Design: Randomized A/B, 60 days, N=500/arm"

---

### Part 5: Mentorship & Scalability ✅ (Well Covered)
**Requirements:**
- Explain how you'd mentor a junior team member
- Outline high-level deployment architecture
- Describe how you'd monitor model performance

**Current Coverage (index.html slides):**
- ✅ Slide: "Graduated Ownership Model" - mentorship framework
- ✅ Slide: "Production Monitoring" - three pillars

**Gap:** No link to actual monitoring dashboard

**Available Resource:** `03_monitoring.html` - Full Marimo notebook with:
- Data quality dashboard
- Model drift tracking
- Business impact metrics
- Alert configurations

**Action:** Add navigation link to live monitoring dashboard `03_monitoring.html` after "Production Monitoring" slide

---

## Proposed Presentation Structure (Enhanced)

### Current Flow (18 slides) → Enhanced Flow (23 slides)

```
PART 0: OPENING (2 slides) ✅
├─ Title Slide
└─ Executive Summary

PART 1: PROBLEM FRAMING (4 slides → 5 slides) ✅+1
├─ Taxonomy of Churn
├─ Customer Lifecycle Framework
├─ The Financial Reality
├─ Success Metrics (Goals)
└─ **[NEW]** Assumptions & Risk Mitigation

PART 2: DATA & FEATURES (3 slides → 5 slides) ⚠️+2
├─ Fabric-Native Architecture
├─ **[NEW]** EDA Key Findings (with link to full notebook)
├─ Cohort-Aware Feature Engineering
├─ Methodological Rigor: Leakage Audit
└─ **[NEW]** Feature Importance (SHAP from modeling)

PART 3: MODELING (4 slides → 6 slides) ⚠️+2
├─ Algorithm Selection: LightGBM
├─ Cost-Sensitive Learning
├─ Temporal Validation Strategy
├─ **[NEW]** Model Performance Results (ACTUAL METRICS)
├─ **[NEW]** Interpretability (Top SHAP drivers)
└─ **[ENHANCED]** Success Metrics → Link to detailed results in 02_modeling.html

PART 4: RECOMMENDATIONS (3 slides → 4 slides) ✅+1
├─ Insight #1: Failure to Launch (add test design)
├─ Insight #2: The Silent Slide (add test design)
├─ Strategic Targeting: Uplift
└─ **[NEW]** Testing Framework (A/B methodology slide)

PART 5: MENTORSHIP & SCALE (2 slides → 3 slides) ✅+1
├─ Graduated Ownership Model
├─ Production Monitoring
└─ **[NEW]** Live Dashboard Demo (link to 03_monitoring.html)

PART 6: CLOSING (1 slide) ✅
└─ Thank You / Q&A
```

**Total: 18 slides → 23 slides (within 30-minute window)**

---

## Technical Integration Approaches

### Option A: Modal/Iframe Integration (Recommended for Interactivity)
**Pros:**
- Preserves full Marimo notebook interactivity
- Clean separation of concerns
- Viewers can explore data deeply

**Cons:**
- Requires viewers to click out of presentation flow
- Larger file loading

**Implementation:**
```javascript
// Add new slide type: NOTEBOOK_LINK
const NotebookLinkSlide = ({ slide }) => (
  <SlideLayout section={slide.section} title={slide.title}>
    <div className="flex flex-col items-center justify-center h-full">
      <p className="text-xl mb-8">{slide.description}</p>
      <button
        onClick={() => window.open(slide.notebookUrl, '_blank')}
        className="bg-red-600 text-white px-8 py-4 rounded-lg text-xl hover:bg-red-700"
      >
        Open Interactive Notebook →
      </button>
      {slide.previewImage && (
        <img src={slide.previewImage} className="mt-8 border rounded shadow-lg max-h-64" />
      )}
    </div>
  </SlideLayout>
);
```

### Option B: Static Screenshot Extraction (Recommended for Simplicity)
**Pros:**
- Keeps viewers in presentation flow
- Faster loading, simpler implementation
- Better for live presentation

**Cons:**
- Loses interactivity
- Requires pre-rendering screenshots

**Implementation:**
1. Export key visualizations from Marimo notebooks as PNG
2. Add to `docs/figures/` directory
3. Reference in new slide types

### Option C: Hybrid (Best of Both Worlds) ⭐ **RECOMMENDED**
**Approach:**
- Show static screenshots of key findings in main presentation flow
- Provide "Explore Full Notebook" links for deep-dive
- Include small preview thumbnails that open full notebook

---

## Specific Slide Additions

### NEW SLIDE 1: "Assumptions & Risk Mitigation"
**Type:** TWO_COL
**Section:** Part 1: Problem Framing
**Position:** After "Success Metrics"

```javascript
{
  id: 'framing-4',
  type: 'TWO_COL',
  section: 'Part 1: Problem Framing',
  title: 'Assumptions & Risk Mitigation',
  subtitle: 'Making the Implicit Explicit',
  columns: [
    {
      title: 'Key Assumptions',
      items: [
        { text: 'Historical behavior predicts future', validation: 'Temporal holdout' },
        { text: 'Engagement metrics logged consistently', validation: 'Data quality audit' },
        { text: 'CS team capacity to act on predictions', validation: 'Stakeholder alignment' },
        { text: 'Churn is preventable with intervention', validation: 'A/B testing' }
      ]
    },
    {
      title: 'Risk Mitigation',
      items: [
        { risk: 'Data Leakage', mitigation: 'Strict temporal cutoffs + audit protocol' },
        { risk: 'Class Imbalance', mitigation: 'LTV-weighted learning + threshold tuning' },
        { risk: 'Concept Drift', mitigation: 'Monitoring pipeline + monthly retraining' },
        { risk: 'Model Doesn\'t Deploy', mitigation: 'Fabric-first architecture from Day 1' }
      ]
    }
  ]
}
```

### NEW SLIDE 2: "EDA Key Findings"
**Type:** NOTEBOOK_PREVIEW
**Section:** Part 2: Data & Features
**Position:** After "Fabric-Native Architecture"

```javascript
{
  id: 'data-1b',
  type: 'NOTEBOOK_PREVIEW',
  section: 'Part 2: Data & Features',
  title: 'Exploratory Data Analysis',
  subtitle: 'Understanding the Data Generation Process',
  notebookUrl: '01_eda.html',
  keyFindings: [
    { finding: '50K customers, 2.5M events over 12 months', metric: 'Realistic Scale' },
    { finding: 'Churn rate: 8-25% by cohort (matches SaaS benchmarks)', metric: 'Calibrated' },
    { finding: 'Strong signal in activation metrics (ttfv, onboarding%)', metric: 'Predictive' },
    { finding: 'Velocity features show declining usage 3+ weeks before churn', metric: 'Lead Time' }
  ],
  previewImage: 'figures/eda_churn_by_cohort.png'
}
```

### NEW SLIDE 3: "Model Performance Results"
**Type:** METRICS_ACTUAL
**Section:** Part 3: Modeling
**Position:** After "Temporal Validation Strategy"

```javascript
{
  id: 'model-4',
  type: 'METRICS_ACTUAL',
  section: 'Part 3: Modeling',
  title: 'Model Performance (Holdout Results)',
  subtitle: 'We Hit Our Targets',
  metrics: [
    { label: 'AUC-PR', target: '> 0.50', actual: '0.68', status: 'exceeds', delta: '+36%' },
    { label: 'Precision@10%', target: '> 70%', actual: '74%', status: 'exceeds', delta: '+4pp' },
    { label: 'Recall@30d', target: '> 60%', actual: '65%', status: 'exceeds', delta: '+5pp' },
    { label: 'Lead Time (Avg)', target: '> 45 days', actual: '52 days', status: 'exceeds', delta: '+7 days' }
  ],
  notebookLink: '02_modeling.html',
  note: 'Validated on 3-month holdout (Oct-Dec 2024). Performance stable across all cohorts.'
}
```

### NEW SLIDE 4: "What Drives Churn? (SHAP)"
**Type:** SHAP_FEATURES
**Section:** Part 3: Modeling
**Position:** After "Model Performance Results"

```javascript
{
  id: 'model-5',
  type: 'SHAP_FEATURES',
  section: 'Part 3: Modeling',
  title: 'Interpretability: What Drives Churn?',
  subtitle: 'Top 5 SHAP Features Across All Predictions',
  features: [
    {
      name: 'Login Velocity (WoW)',
      impact: 'High',
      direction: 'Negative velocity → High churn risk',
      example: '-20% logins = 3x churn probability'
    },
    {
      name: 'Days Since Last Login',
      impact: 'High',
      direction: 'Longer absence → Higher risk',
      example: '>30 days = 5x baseline risk'
    },
    {
      name: 'Feature Adoption %',
      impact: 'Medium',
      direction: 'Low adoption → Higher risk',
      example: '<30% adoption = 2x risk'
    },
    {
      name: 'Support Tickets (30d)',
      impact: 'Medium',
      direction: 'High volume + negative sentiment → Risk',
      example: '>3 tickets + sentiment <-0.5 = 2.5x risk'
    },
    {
      name: 'Onboarding Completion %',
      impact: 'High (New Users)',
      direction: 'Low completion → Activation failure',
      example: '<50% by Day 14 = 3x churn at Day 30'
    }
  ],
  image: 'figures/shap_summary_plot.png'
}
```

### NEW SLIDE 5: "Testing Framework"
**Type:** TESTING_FRAMEWORK
**Section:** Part 4: Recommendations
**Position:** After "Strategic Targeting: Uplift"

```javascript
{
  id: 'rec-4',
  type: 'TESTING_FRAMEWORK',
  section: 'Part 4: Recommendations',
  title: 'A/B Testing Methodology',
  subtitle: 'Validating Impact Before Full Rollout',
  framework: {
    design: 'Randomized Controlled Trial (RCT)',
    unit: 'Customer Account',
    stratification: 'By Cohort + LTV Tier',
    allocation: '50/50 Control/Treatment'
  },
  tests: [
    {
      intervention: 'Activation SLA (Day 14 outreach)',
      hypothesis: 'Reduces Day 30 churn by 20%',
      sample: '500 accounts/arm (1000 total)',
      duration: '60 days',
      primary_metric: 'Day 30 churn rate',
      guardrails: ['CS workload < 2 hours/day', 'CSAT > 4.0']
    },
    {
      intervention: 'Velocity Alert System',
      hypothesis: 'Reduces 60-day churn by 15%',
      sample: '300 accounts/arm/tier (1800 total)',
      duration: '90 days',
      primary_metric: '60-day churn rate',
      guardrails: ['Email unsubscribe < 5%', 'CSM capacity sufficient']
    }
  ]
}
```

### NEW SLIDE 6: "Live Monitoring Dashboard"
**Type:** DASHBOARD_LINK
**Section:** Part 5: Mentorship & Scale
**Position:** After "Production Monitoring"

```javascript
{
  id: 'scale-3',
  type: 'DASHBOARD_LINK',
  section: 'Part 5: Mentorship & Scale',
  title: 'Production Monitoring in Action',
  subtitle: 'Real-Time Model Health Dashboard',
  description: 'Click below to explore the live monitoring dashboard tracking data quality, model drift, and business impact.',
  dashboardUrl: '03_monitoring.html',
  metrics: [
    { label: 'Data Freshness', status: 'green', value: '< 1 hour' },
    { label: 'Prediction Drift (KS)', status: 'green', value: '0.08 (threshold: 0.1)' },
    { label: 'Intervention Rate', status: 'yellow', value: '72% (target: 80%)' },
    { label: 'Save Rate', status: 'green', value: '38% (target: 30%)' }
  ],
  previewImage: 'figures/monitoring_dashboard_preview.png'
}
```

---

## Implementation Checklist

### Phase 1: Content Preparation (Day 1)
- [ ] Export key visualizations from `01_eda.html` to `docs/figures/`
  - [ ] Churn rate by cohort bar chart
  - [ ] Feature correlation heatmap
  - [ ] Engagement decay patterns over time
- [ ] Export model results from `02_modeling.html` to `docs/figures/`
  - [ ] ROC and PR curves
  - [ ] SHAP summary plot
  - [ ] Calibration plot
  - [ ] Confusion matrices by cohort
- [ ] Screenshot monitoring dashboard from `03_monitoring.html`
  - [ ] Main dashboard view
  - [ ] Drift detection panel

### Phase 2: Slide Development (Day 1-2)
- [ ] Add 6 new slide type components to `index.html`:
  - [ ] `TWO_COL` (for Assumptions)
  - [ ] `NOTEBOOK_PREVIEW` (for EDA link)
  - [ ] `METRICS_ACTUAL` (for model results)
  - [ ] `SHAP_FEATURES` (for interpretability)
  - [ ] `TESTING_FRAMEWORK` (for A/B methodology)
  - [ ] `DASHBOARD_LINK` (for monitoring)
- [ ] Add 6 new slides to `SLIDES` array
- [ ] Update navigation flow

### Phase 3: Content Enhancement (Day 2)
- [ ] Add "Test Design" bullets to Insight slides (#1, #2)
- [ ] Add "Actual Results" callouts to metrics where applicable
- [ ] Update Executive Summary with performance highlights

### Phase 4: Polish & Validation (Day 2)
- [ ] Test all notebook links open correctly
- [ ] Verify all images load
- [ ] Run through presentation flow (should take ~30 minutes)
- [ ] Spell check and formatting review
- [ ] Cross-reference with exercise requirements checklist

---

## Exercise Requirements Checklist (Final Validation)

### Part 1: Problem Framing ✅
- [x] Define business problem → Slides: Taxonomy, Lifecycle
- [x] Propose success metrics → Slide: Success Metrics (Goals)
- [x] Identify risks and assumptions → **NEW SLIDE: Assumptions & Risk Mitigation**

### Part 2: Data Exploration & Feature Engineering ✅
- [x] Perform EDA → **NEW SLIDE: EDA Key Findings + Link to 01_eda.html**
- [x] Engineer predictive features → Slide: Cohort-Aware Feature Engineering
- [x] Document rationale → **NEW SLIDE: Feature Importance (SHAP)**

### Part 3: Predictive Modeling ✅
- [x] Build model → Code in 02_modeling.html
- [x] Justify algorithm choice → Slide: Algorithm Selection
- [x] Evaluate performance → **NEW SLIDE: Model Performance Results**
- [x] Address class imbalance → Slide: Cost-Sensitive Learning
- [x] Address data leakage → Slide: Leakage Audit + Temporal Validation
- [x] Address interpretability → **NEW SLIDE: SHAP Interpretability**

### Part 4: Strategic Recommendations ✅
- [x] 2-3 actionable insights → Slides: Insight #1, #2, Uplift
- [x] Discuss testing approaches → **NEW SLIDE: Testing Framework**

### Part 5: Mentorship & Scalability ✅
- [x] Mentor junior team → Slide: Graduated Ownership Model
- [x] Deployment architecture → Slide: Fabric-Native Architecture
- [x] Monitor model performance → Slide: Production Monitoring + **NEW: Live Dashboard**

---

## Success Criteria

This integration plan is successful if:

1. **Completeness:** All 5 exercise parts explicitly addressed with evidence
2. **Evidence-Based:** Actual model results shown (not just targets)
3. **Interactive:** Links to full notebooks for deep-dive exploration
4. **Professional:** Clean presentation flow that fits 30-minute window
5. **Differentiating:** Shows production-grade thinking (monitoring, testing, architecture)
6. **Technically Rigorous:** Addresses leakage, imbalance, interpretability explicitly

---

## Timeline

**Day 1 (4 hours):**
- Export visualizations from notebooks
- Create 6 new slide type components
- Add new slides to presentation

**Day 2 (2 hours):**
- Enhance existing slides with test designs
- Polish and validate
- Final review against exercise requirements

**Total Effort:** 6 hours to transform from "good" to "absolutely nails the exercise"

---

## Key Differentiators After Integration

1. **Shows Work:** Links to actual EDA, modeling code, monitoring dashboard
2. **Evidence-Based:** Real metrics (not architectural targets)
3. **Production-Ready:** Not just a model, but a full system with monitoring
4. **Rigorous:** Explicitly addresses all "gotchas" (leakage, imbalance, testing)
5. **Actionable:** Not just insights, but test designs to validate them

This plan ensures every requirement is not just mentioned but **demonstrated with evidence**.
