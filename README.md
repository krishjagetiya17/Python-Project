Flipkart Products Data Analytics – Exploratory Data Analysis

 -> Project Overview
This project performs Exploratory Data Analysis (EDA) on the Flipkart Products dataset from Kaggle using Python, Pandas, NumPy, Matplotlib, and Seaborn.
The main objective is to analyze product categories, brands, pricing, discounts, and customer ratings through data cleaning and visualization.

 -> Dataset
Source: [Flipkart Products – Kaggle (PromptCloud)]( https://www.kaggle.com/datasets/PromptCloudHQ/flipkartproducts )

The dataset contains about 20,000 Flipkart products with around 15 attributes:

* `pid`: Unique product ID.
* `product_name`: Name of the product.
* `product_category_tree`: Hierarchical category path.
* `retail_price`: Original price.
* `discounted_price`: Price after discount.
* `product_rating` / `overall_rating`: Customer ratings ("No rating available" for missing).
* `brand`: Product brand.
* `is_FK_Advantage_product`: Flipkart Advantage flag.
* `description` / `product_specifications`: Product details.

 -> Objectives

1. Clean the data (duplicates, missing values, "No rating available", numeric price conversion).
2. Extract main categories from the category tree.
3. Identify the brands with the most products.
4. Analyze retail and discounted price distributions.
5. Calculate discount percentage and find the categories with the highest average discounts.
6. Study ratings across categories and brands.
7. Compare Flipkart Advantage and non-Advantage products.
8. Create seven visualizations.

 -> Graphs Included

1. Top 10 Product Categories: Bar chart.
2. Retail Price Distribution: Histogram.
3. Retail Price vs Discounted Price: Scatter plot.
4. Top 10 Brands by Number of Products: Bar chart.
5. Discount Percentage Distribution: Histogram / Box plot.
6. Product Rating Distribution: Bar / Pie chart.
7. Flipkart Advantage vs Non-Advantage: Pie / Bar chart.

 -> Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Google Colab / Jupyter Notebook
* Git & GitHub

 -> Analysis and Expected Outcomes
The analysis explores the following questions:

* Which categories have the most listings?
* Which brands have the most products?
* What is the most common price range?
* How large are discounts across price ranges and categories?
* How are ratings distributed, and how many products are unrated?
* How do Flipkart Advantage products compare with others?

The actual findings will be determined after running the analysis on the dataset.

 -> Conclusion
This project demonstrates how Python and its data libraries can be used to turn raw e-commerce data into meaningful insights on categories, brands, pricing, discounts, and ratings.


 -> Dataset Credit
PromptCloud, Flipkart Products, Kaggle.
[ https://www.kaggle.com/datasets/PromptCloudHQ/flipkartproducts ]







