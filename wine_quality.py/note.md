path for file data: "D:\wine+quality"

# dl
    hết thứ 3 tuần sau xong labwork 
    hết t5 tuần sau làm xong report -> submit

# report
    code LaTeX

# PART 1
    need to compare with red and white wine ???


## determine which feature is discrete or continuous? quantitative or qualitative? numerical or categorical? Explain.
    INPUT FEATURE
        fixed_acidity — continuous, quantitative, numerical
        volatile_acidity — continuous, quantitative, numerical
        citric_acid — continuous, quantitative, numerical
        residual_sugar — continuous, quantitative, numerical
        chlorides — continuous, quantitative, numerical
        free_sulfur_dioxide — continuous, quantitative, numerical
        total_sulfur_dioxide — continuous, quantitative, numerical
        density — continuous, quantitative, numerical
        pH — continuous, quantitative, numerical
        sulphates — continuous, quantitative, numerical
        alcohol — continuous, quantitative, numerical

        Not in INPUT FEATURE, seperated into 2 types, no combination
            color	Other	Categorical	red or white		    no      -> discrete, qualitative, categorial(only red and white)

        Explain: all of this are physical measurement, value across a range  (UNLESS quality and color)
            continous: can take any real value in a range; your CSV shows decimals like 7.4, 0.9978, 3.51, etc.
            quantitative: all of this variable represent amount
            numeical, it is not category

    OUTPUT FEATURE
        quality	Target	Integer	score between 0 and 10		no      -> discrete, quantitative, categorial(can be counted form 0 to 10, deal with number data, 0 to 10 as ranking)
      

## What are the issues lying within data? for example: missing data, incorrect data? Do you need to prepare data?

Issue
    missing data: none
    check data type: normal
    duplicate row: yes
        Duplicate rows(red): 240
        -------------------------------------------------
        Duplicate rows(white): 937
    incorrect/impossible value
        ph: 0 - 14
        chemical value: not negative
        Result
            Rows with impossible pH(red): 0
            Rows with any negative measurement(red): 0
            -----------------------------------------------------------------------------
            Rows with impossible pH(white): 0
            Rows with any negative measurement(white): 0
    
    

Do you need to prepare the data? (Wine Quality answer)
    Yes, mainly for PCA and fair comparison across features:
    Preparation steps
        Read the file correctly (Wine Quality CSV is often semicolon-separated like your snippet).
        Check data types → ensure all 11 features are numeric.
        Check duplicates (optional but good practice).
        Outlier handling (decide one: keep them, winsorize/cap, or use robust scaling — and justify).
        Standardize features before PCA (mean 0, std 1), because PCA is scale-sensitive and sklearn PCA does not scale automatically.
        Decide how to treat quality: ordinal classification (categorical ordered) or regression (discrete numeric), since UCI says it can be viewed as either.
        PCA !!!!!!!!!!

## Is there any label in the datasets? Explain.
label: quality
    it is an output variable form 0(very bad) to 10(very good)
    Because quality is an integer score with a natural order (e.g., 5 < 6 < 7), it can be treated:
        as regression (predicting a numeric score), or
        as classification with ordered classes (ordinal).

## Calculate mean, variance, covariance, correlation of the selected datasets. How do you calculate these measures for categorical features (if any)?
mean
variance
    measure how far a set of random number are spread out of the mean
    var(X) = (standard deviation of x)^2 = sum((mean(x) - x)^2) / n
covariance
    defines between 2 variables at a time
    measure how much 2 random variables change together
correlation
    scaled version of coveriance
    cor(x, y) = cov(x, y) / phi(x) / phi(y)
DIFF: quality
    The physicochemical predictors are numerical, so we compute mean/variance/covariance/correlation directly. The label quality is an ordered score, so we additionally compute correlation between each feature and quality

## Is there any missing data? How do you handle this issue?
NO

# PART 2: PCA
## 