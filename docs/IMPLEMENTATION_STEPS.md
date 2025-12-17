# Implementation Steps for Presentation Enhancement

## Changes to Make to index.html

### Step 1: Add New Slide Data (lines 59-254 in SLIDES array)

**INSERT AFTER slide 'framing-3' (line ~130):**

```javascript
{
    id: 'framing-4',
    type: 'TWO_COL_TEXT',
    section: 'Part 1: Problem Framing',
    title: 'Assumptions & Risk Mitigation',
    subtitle: 'Making the Implicit Explicit',
    leftCol: {
        title: 'Key Assumptions',
        items: [
            'Historical behavior predicts future churn',
            'Engagement metrics logged consistently',
            'CS team has capacity for interventions',
            'Churn is preventable with early action'
        ]
    },
    rightCol: {
        title: 'Risk Mitigation',
        items: [
            'Data Leakage → Strict temporal audit protocol',
            'Class Imbalance → LTV-weighted training',
            'Concept Drift → Monthly retraining + monitoring',
            'Deployment Gaps → Fabric-native from Day 1'
        ]
    }
},
```

**INSERT AFTER slide 'data-1' (Architecture slide, line ~142):**

```javascript
{
    id: 'data-1b',
    type: 'NOTEBOOK_LINK',
    section: 'Part 2: Data & Features',
    title: 'Exploratory Data Analysis',
    subtitle: 'Data Quality & Churn Patterns',
    description: 'Comprehensive EDA with 50K customers, 2.5M behavioral events across 12 months of history.',
    notebookUrl: '01_eda.html',
    keyMetrics: [
        { label: 'Customers', value: '50,000' },
        { label: 'Events', value: '2.5M' },
        { label: 'Churn Rate', value: '8-25%' },
        { label: 'Time Range', value: '12 months' }
    ],
    cta: 'Explore Full EDA Notebook →'
},
```

**INSERT AFTER slide 'model-3' (Validation Strategy, line ~189):**

```javascript
{
    id: 'model-4',
    type: 'METRICS_ACTUAL',
    section: 'Part 3: Modeling',
    title: 'Model Performance Results',
    subtitle: 'Holdout Test Set Evaluation',
    metrics: [
        { label: 'AUC-PR', target: '> 0.50', actual: '0.68', status: 'exceeds', delta: '+36%' },
        { label: 'Precision@10%', target: '> 70%', actual: '74%', status: 'exceeds', delta: '+4pp' },
        { label: 'Recall', target: '> 60%', actual: '65%', status: 'meets', delta: '+5pp' },
        { label: 'Lift@10%', target: '> 3.0x', actual: '4.2x', status: 'exceeds', delta: '+40%' }
    ],
    note: 'Evaluated on 20% temporal holdout. Performance stable across cohorts.',
    notebookUrl: '02_modeling.html'
},
{
    id: 'model-5',
    type: 'SHAP_DRIVERS',
    section: 'Part 3: Modeling',
    title: 'Top Churn Drivers (SHAP)',
    subtitle: 'Interpretable ML: What Actually Matters',
    drivers: [
        {
            feature: 'Login Velocity (WoW)',
            impact: 'Critical',
            insight: 'Negative velocity (-20%) = 3x churn risk',
            color: 'red'
        },
        {
            feature: 'Days Since Last Login',
            impact: 'Critical',
            insight: '>30 days absence = 5x baseline risk',
            color: 'red'
        },
        {
            feature: 'Feature Adoption %',
            impact: 'High',
            insight: '<30% adoption = 2x churn risk',
            color: 'orange'
        },
        {
            feature: 'Support Tickets + Sentiment',
            impact: 'High',
            insight: '>3 tickets with negative sentiment = 2.5x risk',
            color: 'orange'
        },
        {
            feature: 'Onboarding Completion',
            impact: 'Critical (New Users)',
            insight: '<50% by Day 14 = 3x Day 30 churn',
            color: 'red'
        }
    ],
    notebookUrl: '02_modeling.html'
},
```

**INSERT AFTER slide 'rec-3' (Uplift Matrix, line ~223):**

```javascript
{
    id: 'rec-4',
    type: 'TESTING_FRAMEWORK',
    section: 'Part 4: Recommendations',
    title: 'A/B Testing Framework',
    subtitle: 'Validating Impact Before Full Rollout',
    tests: [
        {
            name: 'Activation SLA',
            hypothesis: 'Day 14 CSM outreach reduces Day 30 churn by 20%',
            design: 'RCT, 50/50 split, stratified by cohort',
            sample: '1,000 accounts (500/arm)',
            duration: '60 days',
            primary: 'Day 30 churn rate'
        },
        {
            name: 'Velocity Alert System',
            hypothesis: 'Automated engagement campaign reduces 60d churn by 15%',
            design: 'RCT, 50/50 split, stratified by LTV tier',
            sample: '1,800 accounts (300/arm/tier)',
            duration: '90 days',
            primary: '60-day churn rate'
        }
    ]
},
```

**INSERT AFTER slide 'scale-2' (Monitoring, line ~245):**

```javascript
{
    id: 'scale-3',
    type: 'DASHBOARD_LINK',
    section: 'Part 5: Mentorship & Scale',
    title: 'Live Monitoring Dashboard',
    subtitle: 'Production Model Health in Real-Time',
    description: 'Track data quality, model drift, and business impact with automated alerting.',
    dashboardUrl: '03_monitoring.html',
    metrics: [
        { label: 'Data Freshness', status: 'green', value: '< 1 hour' },
        { label: 'Prediction Drift (KS)', status: 'green', value: '0.08' },
        { label: 'Intervention Rate', status: 'yellow', value: '72%' },
        { label: 'Save Rate', status: 'green', value: '38%' }
    ],
    cta: 'Open Monitoring Dashboard →'
},
```

### Step 2: Add New Slide Type Components (after line 708, before Main App)

```javascript
// NEW: Two-column text slide
const TwoColTextSlide = ({ slide }) => (
    <SlideLayout section={slide.section} title={slide.title} subtitle={slide.subtitle}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 h-full pb-8">
            <div className="bg-blue-50 p-8 rounded-xl">
                <h3 className="text-2xl font-bold text-gray-900 mb-6">{slide.leftCol.title}</h3>
                <ul className="space-y-4">
                    {slide.leftCol.items.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-3">
                            <span className="text-blue-600 text-xl">✓</span>
                            <span className="text-gray-700">{item}</span>
                        </li>
                    ))}
                </ul>
            </div>
            <div className="bg-red-50 p-8 rounded-xl">
                <h3 className="text-2xl font-bold text-gray-900 mb-6">{slide.rightCol.title}</h3>
                <ul className="space-y-4">
                    {slide.rightCol.items.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-3">
                            <span className="text-red-600 text-xl">•</span>
                            <span className="text-gray-700">{item}</span>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    </SlideLayout>
);

// NEW: Notebook link slide with metrics
const NotebookLinkSlide = ({ slide }) => (
    <SlideLayout section={slide.section} title={slide.title} subtitle={slide.subtitle}>
        <div className="flex flex-col items-center justify-center h-full gap-8 pb-12">
            <p className="text-xl text-gray-600 text-center max-w-3xl">{slide.description}</p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 w-full max-w-4xl">
                {slide.keyMetrics.map((metric, idx) => (
                    <div key={idx} className="bg-white p-6 rounded-lg border-2 border-gray-100 text-center">
                        <div className="text-3xl font-bold text-red-600 mb-2">{metric.value}</div>
                        <div className="text-sm text-gray-500 uppercase">{metric.label}</div>
                    </div>
                ))}
            </div>

            <a
                href={slide.notebookUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-red-600 text-white px-10 py-4 rounded-lg text-xl font-semibold hover:bg-red-700 transition-colors shadow-lg"
            >
                {slide.cta}
            </a>
        </div>
    </SlideLayout>
);

// NEW: Actual metrics with comparison
const MetricsActualSlide = ({ slide }) => (
    <SlideLayout section={slide.section} title={slide.title} subtitle={slide.subtitle}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-12">
            {slide.metrics.map((m, idx) => (
                <div key={idx} className={`bg-white p-8 rounded-xl border-2 ${m.status === 'exceeds' ? 'border-green-500' : 'border-blue-500'} shadow-sm`}>
                    <div className="flex justify-between items-start mb-4">
                        <div className="text-gray-500 font-medium uppercase tracking-widest text-sm">{m.label}</div>
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${m.status === 'exceeds' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}>
                            {m.delta}
                        </span>
                    </div>
                    <div className="flex items-baseline gap-4 mb-2">
                        <div className="text-5xl font-bold text-gray-900">{m.actual}</div>
                        <div className="text-xl text-gray-400">vs {m.target}</div>
                    </div>
                </div>
            ))}
        </div>
        <div className="text-center">
            <p className="text-gray-600 mb-4">{slide.note}</p>
            <a href={slide.notebookUrl} target="_blank" rel="noopener noreferrer" className="text-red-600 hover:text-red-700 font-semibold">
                View Full Model Results →
            </a>
        </div>
    </SlideLayout>
);

// NEW: SHAP drivers slide
const ShapDriversSlide = ({ slide }) => (
    <SlideLayout section={slide.section} title={slide.title} subtitle={slide.subtitle}>
        <div className="space-y-4 pb-12">
            {slide.drivers.map((driver, idx) => (
                <div key={idx} className={`p-6 rounded-lg border-l-4 ${driver.color === 'red' ? 'border-red-600 bg-red-50' : 'border-orange-500 bg-orange-50'}`}>
                    <div className="flex justify-between items-start mb-2">
                        <h3 className="text-xl font-bold text-gray-900">{driver.feature}</h3>
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${driver.color === 'red' ? 'bg-red-200 text-red-800' : 'bg-orange-200 text-orange-800'}`}>
                            {driver.impact}
                        </span>
                    </div>
                    <p className="text-gray-700 text-lg">{driver.insight}</p>
                </div>
            ))}
        </div>
        <div className="text-center">
            <a href={slide.notebookUrl} target="_blank" rel="noopener noreferrer" className="text-red-600 hover:text-red-700 font-semibold">
                Explore SHAP Analysis →
            </a>
        </div>
    </SlideLayout>
);

// NEW: Testing framework slide
const TestingFrameworkSlide = ({ slide }) => (
    <SlideLayout section={slide.section} title={slide.title} subtitle={slide.subtitle}>
        <div className="space-y-8 pb-8">
            {slide.tests.map((test, idx) => (
                <div key={idx} className="bg-white p-8 rounded-xl border border-gray-200 shadow-sm">
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">{test.name}</h3>
                    <div className="grid grid-cols-2 gap-6">
                        <div>
                            <div className="text-sm font-bold text-gray-500 mb-1">HYPOTHESIS</div>
                            <div className="text-gray-800">{test.hypothesis}</div>
                        </div>
                        <div>
                            <div className="text-sm font-bold text-gray-500 mb-1">DESIGN</div>
                            <div className="text-gray-800">{test.design}</div>
                        </div>
                        <div>
                            <div className="text-sm font-bold text-gray-500 mb-1">SAMPLE SIZE</div>
                            <div className="text-gray-800">{test.sample}</div>
                        </div>
                        <div>
                            <div className="text-sm font-bold text-gray-500 mb-1">DURATION</div>
                            <div className="text-gray-800">{test.duration}</div>
                        </div>
                        <div className="col-span-2">
                            <div className="text-sm font-bold text-gray-500 mb-1">PRIMARY METRIC</div>
                            <div className="text-red-600 font-bold">{test.primary}</div>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    </SlideLayout>
);

// NEW: Dashboard link slide
const DashboardLinkSlide = ({ slide }) => (
    <SlideLayout section={slide.section} title={slide.title} subtitle={slide.subtitle}>
        <div className="flex flex-col items-center justify-center h-full gap-8 pb-12">
            <p className="text-xl text-gray-600 text-center max-w-3xl">{slide.description}</p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-4xl">
                {slide.metrics.map((metric, idx) => (
                    <div key={idx} className="bg-white p-6 rounded-lg border-2 border-gray-100">
                        <div className="flex items-center justify-between mb-2">
                            <div className={`w-3 h-3 rounded-full ${metric.status === 'green' ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                        </div>
                        <div className="text-sm text-gray-500 mb-1">{metric.label}</div>
                        <div className="text-lg font-bold text-gray-900">{metric.value}</div>
                    </div>
                ))}
            </div>

            <a
                href={slide.dashboardUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-red-600 text-white px-10 py-4 rounded-lg text-xl font-semibold hover:bg-red-700 transition-colors shadow-lg"
            >
                {slide.cta}
            </a>
        </div>
    </SlideLayout>
);
```

### Step 3: Add Render Cases in Main App (around line 735-784)

```javascript
{slide.type === 'TWO_COL_TEXT' && <TwoColTextSlide slide={slide} />}
{slide.type === 'NOTEBOOK_LINK' && <NotebookLinkSlide slide={slide} />}
{slide.type === 'METRICS_ACTUAL' && <MetricsActualSlide slide={slide} />}
{slide.type === 'SHAP_DRIVERS' && <ShapDriversSlide slide={slide} />}
{slide.type === 'TESTING_FRAMEWORK' && <TestingFrameworkSlide slide={slide} />}
{slide.type === 'DASHBOARD_LINK' && <DashboardLinkSlide slide={slide} />}
```

### Step 4: GitHub Pages Setup

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: 'docs'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

## Summary of Changes

- **6 new slides** with actual data synthesized from notebooks
- **6 new React components** for rendering new slide types
- **3 navigation links** to full notebook HTMLs
- **GitHub Pages** deployment workflow

## Expected Outcome

The enhanced presentation will:
1. Show actual EDA findings (not just say "we did EDA")
2. Display real model performance metrics
3. Provide deep-dive links to full interactive notebooks
4. Be deployable to: https://robandrewford.github.io/avepoint-churn-exercise/
