# Machine Learning Prediction of Surgical Intervention for Small Bowel Obstruction

Authors: Miles Turpin, Joshua Watson, MD, Matthew Engelhard, MD, PhD, Ricardo Henao, PhD, David Thompson, MD, Lawrence Carin, PhD, Allan Kirk, MD, PhD

## ABSTRACT

### Importance
Small bowel obstruction (SBO) results in over 350,000 operations and over $2 billion in health care expenditures each year in the US. Prompt, effective identification of patients at high and/or low risk of requiring surgery could improve survival, lower incidence of local systemic complications, and shorten the average length of hospitalization.

### Objective
To develop a machine learning model that continuously and effectively predicts the risk of requiring surgery among patients admitted for SBO.

### Design, Setting, and Participants
A prediction model was developed in 2018-2019 based on retrospective analysis of SBO-related encounters taking place in the Duke University Health System between 2013 and 2017.

### Main Outcomes and Measures
Performance was assessed in each hour after admission when predicting whether the patient will (a) receive surgery in the next 24 hours, and (b) receive surgery at any time during the encounter. Measures of performance included the area under the receiver operating characteristic curve (AUROC) and concordance index (C-index). Measures of effectiveness for discharging patients included the incorrect discharge rate and average reduction in hospital stay.

### Results
A total 3,910 encounters among 3,374 unique patients were identified and used in model development. Model-based discharge of low-risk patients was projected to reduce the average length of stay among patients not receiving surgery by over 60 hours while maintaining an incorrect discharge rate lower than the observed readmission rate (9.3%). AUROC for the 24-hour prediction task increased from 0.644 to 0.779 at 12 and 72 hours post-admission, respectively. Concordance index increased from 0.639 to 0.729 at 12 and 72 hours post-admission, respectively.

### Conclusions and Relevance
A machine learning model can effectively and continuously stratify SBO patients by their risk of requiring surgery. This approach, which we show quantitatively to reduce the average length of stay, could be used to improve prioritization of operating room resources by discharging patients whose risk is low. Further study is needed to prospectively explore the benefits of model deployment in an inpatient setting.

## KEY POINTS

### Question
Can a machine learning model effectively stratify patients by their risk of requiring surgery for small bowel obstruction?

### Findings
After learning from 3,910 hospital encounters, the model selected low-risk patients for discharge an average of 60 hours before their observed discharge time with a lower rate of incorrect discharge compared to clinicians. It also effectively stratified patients by risk and predicted whether surgery would be required in the next 24 hours (CI=0.729, AUROC=0.779 at 72 hours after admission).

### Meaning
This model will assist in risk stratifying patients admitted for SBOs and help clinical teams identify patients who can safely and accurately be discharged without surgical management.