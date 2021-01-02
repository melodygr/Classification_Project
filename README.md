# Analysis of Terry Stops in Seattle
Flat Iron Phase 3 Data Science Project

<img src= 
"Images/Stop_Sign.jpg" 
         alt="Stop Sign Image" 
         align="right"
         width="275" height="275"> 
         
<!---<span>Photo by <a href="https://unsplash.com/@pedroplus?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Pedro da Silva</a> on <a href="https://unsplash.com/s/photos/stop-sign?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>--->

Prepared and Presented by:  **_Melody Peterson_**  
[Presentation PDF](https://github.com/melodygr/Housing-Regression-Project/blob/main/presentation.pdf "Presentation PDF")

### Business Problem    
King County Real Estate is a luxury real estate company serving sellers and buyers in the high income earning areas of King County, Washington. The company wants to understand which features translate to higher housing prices in these areas, as well as develop a model to predict price based on housing features.

### Data    
This project uses the King County House Sales dataset, which can be found in kc_house_data.csv in the data folder in this repo. The description of the column names can be found in column_names.md in the same folder. In an effort to narrow the scope of the data to suit our business problem, we also obtained census data of individual income tax returns by zip code for the state of Washington.  An editted version of this data can be found in agi_zip_code.xlsx in the data folder in this repo.  The cleaning and selection of relevent data from this dataset can be seen in the [Additional_Data](https://github.com/swzoeller/Housing-Regression-Project/blob/main/Additional_Data.ipynb "Additional Data Notebook") notebook in the repo.

### Modeling Process
Following the OSEMN (Obtain, Scrub, Explore, Model, Interpret) data science framework, we began with an understanding of our business problem and the acquisition of data.  We then followed an iterative process of cleaning and exploring the data, checking for issues with modeling assumptions, creating and testing a model, interpreting the model, and reevaluating the data.

In the initial data exploration, after subsetting the data to the top zipcodes, we checked the distributions of the independent variables for normal distributions.  Although it is not required for the data to be distributed normally, it can result in better models and predictions.
![Subset Distributions](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/subset_distributions.png "Subset Distributions")

As part of the data cleaning/scrubbing phase, we checked for duplicates, and treated place holder values and missing values in ways to best retain as much data as possible while keeping the integrity of the data.  We also checked for multicollinearity among the independent variables and found several variables with high correlations, including: sqft living/sqft above, sqft living/grade, sqft living 15/sqft living, grade/sqft above, bathrooms/sqft living.
![Data Heatmap](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/heatmap.png "Heat Map")

Once the data had been cleaned we further explored by looking at plots of the data for linear relationships, normal distributions, and skew caused by outliers.  Many of the variables appeared to be skewed by abnormally high outliers. We used IQR to remove price outliers from the dataset before our train test split.
![Price Distribution](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/outlier_comparison.png "Price Distribution")

After creating an initial baseline model, several of the continuous variables were log transformed and scaled to make them more normally distributed and comparable to each other.
![Logged_Histograms](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/logged_histograms.png "Logged Histograms") 

We then iterated through the modeling process, interpreting our results after each model, and making changes and adjustments based on statistical significance of the variables.  For our final model, you can see in this graph how our predictions match up with the actual data on which we trained the model as well as on predicting the test data value for Sale Price.

![Predictions](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/predictions.png "Predictions")

![Predictions_Test](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/predictions_test.png "Predictions Test")

By holding all variables except one constant at their mean, we can visualize the relationship between sale price and any given variable as predicted by our model.

![Single Var Plots](https://github.com/swzoeller/Housing-Regression-Project/blob/main/images/single_var_plots.png "Single Var Plots")

### Conclusions  
* Significant features in  luxury homes include waterfront property, location (zip codes, longitude), and square foot above ground
* Having more floors or bedrooms does not necessarily imply higher sale price
* Bottom Line: location and square footage are the most important features in determining sale price

### Next Steps / Future Work  
1. Refine dataset (expand and cut certain zip codes)
1. Subset model for different price ranges
1. Investigate polynomial relationships and interactions between variables in greater detail

