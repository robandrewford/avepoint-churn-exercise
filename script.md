Presentation Deck: Churn: From Prediction To Retention, A Cohort-Aware, LTV-Weighted Approach for Enterprise SaaS
Time: 30 Minutes (20 min presentation, 10 min Q&A)

Slide 1: Title Slide
Visual: Clean white background. AvePoint red accent bar on the left. Large bold title.

Title: Churn Prediction & Retention Strategy

Subtitle: A Cohort-Aware, LTV-Weighted Approach for Enterprise SaaS

Footer: [Your Name] | Principal Applied Scientist Candidate

Talk Track: "Good morning. My goal today is to demonstrate how we can move from simply predicting churn to actively preventing it. I’ve designed a system that addresses the unique complexities of Enterprise SaaS—specifically the long activation times and high value variance between customers—which I believe reflects AvePoint’s reality."

Slide 2: Executive Summary
Visual: Three horizontal panels (Problem, Solution, Impact).

Content:

Problem: One-size-fits-all models fail in Enterprise SaaS.

Solution: Cohort-aware LightGBM model with LTV-weighted learning on Microsoft Fabric.

Impact: Target reduction in churn by 15%; Focus on "Persuadable" high-LTV customers.

Talk Track: "The core thesis of this project is that treating a new SMB user the same as a mature Enterprise account is a mathematical error. My solution introduces 'Cohort-Aware Modeling' to fix this, deployed on a Fabric architecture that scales."

Part 1: Problem Framing (3 Slides)
Slide 3: The Business Problem: Taxonomy of Churn
Visual: A taxonomy tree diagram splitting Churn into "Voluntary" vs. "Involuntary", and then "Engagement Decay" vs. "Silent Churn".

Key Point: Engagement Decay is the primary target (60-90 day window).

Talk Track: "Not all churn is solvable with ML. I’ve scoped this solution to target 'Engagement Decay'—users who are technically active but losing momentum. This gives us a 60-90 day window to intervene, unlike cancellation requests where it’s already too late."

Slide 4: Customer Lifecycle Framework
Visual: A horizontal timeline arrow divided into three colored zones:

New Users (0-30d): "Activation Risk"

Established (30-180d): "Habit Formation"

Mature (180d+): "Renewal Risk"

Content: Text overlays showing different predictors for each stage (e.g., "Time-to-Value" for New Users vs. "Renewal Date" for Mature).

Talk Track: "A critical insight is that 'churn' means different things at different stages. For a new user, churn is a failure to launch. For a mature user, it's a failure to renew. I’ve designed the feature engineering to respect these distinct cohorts."

Slide 5: The Financial Reality (LTV Impact)
Visual: A bar chart comparing "Cost of Churn" for SMB vs. Enterprise.

Enterprise bar is 10x taller than SMB.

Key Metric: 1 Enterprise Churn ≈ 90 SMB Churns.

Talk Track: "Statistically, these are data points. Financially, they are worlds apart. Losing one Enterprise account causes the same revenue damage as losing 90 SMBs. A standard model treats them equally. My approach uses Cost-Sensitive Learning to weight the model’s attention on high-LTV accounts."

Part 2: Data & Features (3 Slides)
Slide 6: Architecture: Fabric-Native Data Flow
Visual: "Medallion" Architecture diagram using Fabric Hexagon Icons.

Bronze (OneLake): Raw Parquet files.

Silver (Synapse): Cleaned & Sessionized data.

Gold (Feature Store): Point-in-Time Correct Features.

Talk Track: "I’ve architected this to map 1:1 with Microsoft Fabric. We move from raw events in OneLake to a 'Gold' feature store. This ensures that the data engineering done here is not just a prototype, but a blueprint for production deployment."

Slide 7: Feature Engineering Strategy
Visual: A matrix table.

Rows: Activation, Engagement, Velocity, Contract.

Columns: New User, Established, Mature.

Checkmarks show which features apply to which cohort.

Key Concept: Velocity Features (Rate of change).

Talk Track: "Static features like 'total logins' are weak predictors. I engineered 'Velocity Features'—the first derivative of usage. Is usage accelerating or decelerating week-over-week? This negative velocity is often the first smoke signal of a fire."

Slide 8: The "Leakage Audit" (Methodological Rigor)
Visual: A timeline showing a "Prediction Date" cut-off.

Red Zone (Future): Data here is strictly forbidden.

Green Zone (History): Data here is aggregated.

Text: "Strict Point-in-Time Correctness."

Talk Track: "The most common failure in churn models is data leakage—using future data to predict the past. I implemented a formal 'Leakage Audit' for every feature, ensuring we only use information available at the exact moment of prediction."

Part 3: Modeling (5 Slides)
Slide 9: Algorithm Selection
Visual: Comparison card: LightGBM vs. Logistic Regression vs. Neural Nets.

Why LightGBM?

Native handling of missing values (sparse data).

Tree-based learning captures non-linear usage patterns.

Native support for sample weights (critical for LTV).

Talk Track: "I chose LightGBM not just because it's state-of-the-art for tabular data, but because it natively handles the LTV-weighting and null values common in early-lifecycle data without complex imputation pipelines."

Slide 10: Cost-Sensitive Learning (The "Math" Slide)
Visual: Simplified equation of the Weighted Loss Function.

Weight = f(LTV_Tier)

Graphic: A see-saw balancing 1 Enterprise Error vs. 10 SMB Errors.

Talk Track: "This is where we align math with business. We modify the loss function so the model is penalized 10x more for misclassifying an Enterprise customer than an SMB. We are optimizing for 'Revenue Saved', not just 'Accuracy'."

Slide 11: Temporal Validation Strategy
Visual: "Rolling Window" Cross-Validation diagram.

Train: Jan-Mar → Test: Apr

Train: Jan-Apr → Test: May

Key Point: Simulates real-world production forecasting.

Talk Track: "Random cross-validation lies about performance in time-series problems. I used a Rolling Origin validation that mimics exactly how the model will be used in production: training on the past to predict the future, strictly respecting the time arrow."

Slide 12: Model Interpretability (SHAP)
Visual: SHAP Force Plot (Red vs. Blue arrows pushing the score).

Example: "Customer A: High Risk due to '-20% Login Velocity'."

Talk Track: "A 'black box' score isn't actionable for a CSM. We use SHAP values to explain why. For this customer, the model isn't just saying 'High Risk'; it's shouting 'Velocity dropped 20%'. That dictates the specific phone call the CSM needs to make."

Slide 13: Target KPIs & Success Goals
Visual: Dashboard-style KPI cards.

AUC-PR: Goal > 0.50 (Baseline for imbalanced data).

Precision@10%: Goal > 70% (CS Team Capacity).

Recall@30d: Goal > 60% (Coverage).

Talk Track: "Since we are using synthetic data, these are our architectural goals. I prioritized Precision at the top 10% because our CS team has finite capacity. We can't flood them with false alarms. We need to be right when we ask them to act."

Part 4: Recommendations (4 Slides)
Slide 14: Insight #1: The "Failure to Launch" (Activation)
Insight: Users who don't reach "First Value" by Day 14 churn at 3x the rate.

Action: "Day 14 SLA" - Automated CSM alert if onboarding < 50%.

Test: A/B Test (Intervention vs. Control) on new signups.

Talk Track: "Our data shows that if a user hasn't configured the product by Day 14, they are effectively already gone. Recommendation 1 is a strict SLA: If onboarding is under 50% at two weeks, a human intervenes immediately."

Slide 15: Insight #2: The "Silent Slide" (Velocity)
Insight: Negative login velocity precedes churn by 3 weeks.

Action: "Momentum Campaign" - Automated email sequence triggered by -15% WoW velocity.

Test: Measure "Return to Baseline" usage.

Talk Track: "For established users, the signal is subtle. A 15% drop in velocity is invisible to the naked eye but obvious to the model. We can automate a 'Momentum' campaign to nudge them back before they enter the danger zone."

Slide 16: Uplift Framework (Strategic Targeting)
Visual: The "Uplift Quadrant" Matrix.

Sure Things: (High retention, Low risk) → Leave Alone

Lost Causes: (High risk, Low save prob) → Do not disturb

Persuadables: (Medium risk, High response) → Target Here

Talk Track: "This is the most strategic pivot. We shouldn't target the highest risk customers—they are often 'Lost Causes'. We should target the 'Persuadables'. By focusing resources here, we maximize the incremental ROI of the CS team's time."

Slide 17: Business Impact Calculation
Visual: ROI Funnel.

Input: 50 Interventions.

Success Rate: 40% Save Rate.

Outcome: $500k Projected Quarterly Revenue Saved.

Talk Track: "Bringing this back to the bottom line: If we target the top 10% of risk using this Uplift model, and achieve a conservative 40% save rate on Persuadables, we project preventing $500k in annualized churn quarterly."

Part 5: Mentorship & Scale (3 Slides)
Slide 18: Mentorship: The "Graduated Ownership" Model
Visual: Staircase diagram.

Step 1: Shadow (Framing).

Step 2: Assist (EDA).

Step 3: Lead (Modeling).

Step 4: Own (Deployment).

Talk Track: "How do we scale this? I don't just hand off tasks. I use a 'Graduated Ownership' model. A junior data scientist starts by shadowing the problem framing, then leads the EDA. By the deployment phase, they own the code, and I review the architecture."

Slide 19: Production Architecture (Fabric)
Visual: System Diagram.

Daily Trigger (Pipeline) → Score (Synapse) → Write to CRM (Dataverse) → Monitor (PowerBI).

Talk Track: "This isn't just a notebook on a laptop. It's designed to run as a Fabric Pipeline, pushing scores directly into Dataverse so CSMs see the 'Churn Risk' field right inside their CRM. It fits seamlessly into the Microsoft ecosystem."

Slide 20: Monitoring: The Three Pillars
Visual: Three pillars icon.

Data Quality: (Is the feed broken?)

Model Drift: (Has customer behavior changed?)

Business Impact: (Are we actually saving customers?)

Talk Track: "Finally, we need to know when to retrain. We monitor three pillars: Data Quality, Model Drift, and most importantly, Business Impact. If the 'Save Rate' drops, we investigate, regardless of what the AUC says."

Part 6: Closing
Slide 21: Closing & Questions
Visual: Summary bullet points of the value prop.

Cohort-Aware Accuracy.

LTV-Weighted ROI.

Fabric-Native Scale.

Talk Track: "To summarize: We've moved from a generic churn model to a specific, financially-weighted system that integrates with your existing stack. I'm happy to take any questions."

Q&A Preparation (Anticipating the Panel)
Q1: Why LightGBM? Why not a Deep Learning / Transformer approach for time-series?

Answer: "Deep Learning is powerful, but for tabular behavioral data with this sample size (~50k), Gradient Boosted Trees consistently outperform Transformers in empirical benchmarks. Furthermore, LightGBM offers native interpretation (SHAP) and is far more cost-effective to inference in production."

Q2: How do you handle the delay in 'Churn' labels (lag)?

Answer: "Great question. That's why I defined the target as 'Engagement Decay' (predicting a drop in usage) rather than just 'Cancellation'. We get the usage data immediately, allowing us to train on fresher signals without waiting 90 days for a contract to officially expire."

Q3: How would you validate the Uplift assumption (that interventions actually work)?

Answer: "We can't know for sure without testing. My recommendation is to run a randomized control trial (RCT) where we hold out a control group of 'High Risk' users who receive no intervention. This allows us to measure the true causal lift of our CS team's outreach."