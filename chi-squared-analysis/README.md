# Chi-Squared Goodness-of-Fit Analysis

This project analyses binned particle energy data by comparing observed counts with an expected background model. The aim is to evaluate whether the observed particle distribution is consistent with the theoretical background prediction.

The dataset is stored as a dictionary containing bin edges and the number of observed events in each bin. Expected counts for each bin are calculated analytically from a background model that combines an exponential component and a constant term.

To assess the agreement between the data and the model, a Pearson chi-squared goodness-of-fit test is performed. The chi-squared statistic, reduced chi-squared value, and corresponding p-value are calculated to quantify how well the expected distribution matches the observed data.

The project demonstrates key statistical analysis techniques commonly used in physics and data science, including:

- Loading and inspecting structured data
- Computing expected values from a model
- Visualising binned data and model predictions
- Performing a chi-squared goodness-of-fit test
- Interpreting statistical measures such as the reduced chi-squared and p-value