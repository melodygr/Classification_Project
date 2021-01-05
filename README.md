# Analysis of Terry Stops in Seattle
Flat Iron Data Science Project - Phase 3
<img src= 
"Images/No_Stopping.jpg" 
         alt="Stop Sign Image" 
         align="right"
         width="275" height="275"> 
         
<!---Photo by Kevork Kurdoghlian on Unsplash--->       
<!---<span>Photo by <a href="https://unsplash.com/@pedroplus?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Pedro da Silva</a> on <a href="https://unsplash.com/s/photos/stop-sign?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>--->
Prepared and Presented by:  **_Melody Peterson_**  
[Presentation PDF](https://github.com/melodygr/Classification_Project/blob/main/Terry%20Stop%20Presentation.pdf "Presentation PDF")

### Business Problem    
A Terry stop in the United States allows the police to briefly detain a person based on reasonable suspicion of involvement in criminal activity. Reasonable suspicion is a lower standard than probable cause which is needed for arrest. When police stop and search a pedestrian, this is commonly known as a stop and frisk. When police stop an automobile, this is known as a traffic stop. If the police stop a motor vehicle on minor infringements in order to investigate other suspected criminal activity, this is known as a pretextual stop. - [Wikipedia](https://en.wikipedia.org/wiki/Terry_stop#:~:text=A%20Terry%20stop%20in "Terry Stop Definition")

This classification project attempts to determine the possible demographic variables that determine the arrest outcome of a Terry stop. Modeling is done for inference only as making prediction with the model would be incorporating any possible human bias into the model.

### Data    
This data represents records of police reported stops under Terry v. Ohio, 392 U.S. 1 (1968).  The dataset was created on 04/12/2017 and first published on 05/22/2018 and is provided by the city of Seattle, WA.  There were 45,317 rows and 23 variables.  The classification target is ‘Arrest Flag’.  Initial ‘Arrest Flag’ distribution  ‘N’ - 42585, ‘Y’ - 2732  

### Modeling Process
In the initial data cleaning/scrubbing phase, place holder values and missing values were treated in ways to best retain as much data as possible while keeping the integrity of the data.  Generally, missing values were binned together into 'Unknown' categories as can be seen in the histograms below.  
![Subject Age Group](https://github.com/melodygr/Classification_Project/blob/main/Images/subject_age_group.png "Subject Age Group")
![Precinct](https://github.com/melodygr/Classification_Project/blob/main/Images/precinct.png "Precinct")

Once the data had been cleaned, initial models were run to help determine if there were any confounding variables as suspected in Stop Resolution.  Issues were found with a feature engineered category of Subject ID Unknown where none of the data points in this categry were positive for the target.  Date was also proving to be confounding because none of the positive target records had occured before a certain date.    
![Subject ID Comparison](https://github.com/melodygr/Classification_Project/blob/main/Images/subj_known_comparison.png "Subject ID comparison")  
![Date Dual Plot](https://github.com/melodygr/Classification_Project/blob/main/Images/date_dual_plot.png "Date Dual Plot")  

After creating an initial baseline model, several of the continuous variables were log transformed and scaled to make them more normally distributed and comparable to each other.
![Logged_Histograms](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/logged_histograms.png "Logged Histograms") 

We then iterated through the modeling process, interpreting our results after each model, and making changes and adjustments based on statistical significance of the variables.  For our final model, you can see in this graph how our predictions match up with the actual data on which we trained the model as well as on predicting the test data value for Sale Price.

![Predictions](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/predictions.png "Predictions")

![Predictions_Test](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/predictions_test.png "Predictions Test")

By holding all variables except one constant at their mean, we can visualize the relationship between sale price and any given variable as predicted by our model.

![Single Var Plots](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/single_var_plots.png "Single Var Plots")

### Conclusions  
* Analysis of misclassified data
* 
* 

### Next Steps / Future Work  
1. Further analyze unknown or missing values
1. Update ‘Arrest Flag’ with arrest values from ‘Stop Resolution’
1. Try no SMOTE
1. Tune Support Vector Classification


