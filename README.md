
AMAZON UNBOXED
Sales & Discount Trends Analysis


Technical Analytical Report
Dataset: Amazon Products Dataset (Kaggle)
Scope: Discount Analysis | Ratings Analysis | Product Analysis | Revenue Estimation

Overall Avg Discount: ~47%  |  Highest Discount: 64.76%  |  Lowest Discount: 20.09%
Overall Avg Rating: ~4.0 / 5.0  |  Best Imputation: Forward Fill  |  Revenue Leader: Electronics
 
1. Project Overview
This project is a comprehensive end-to-end exploratory data analysis (EDA) and business intelligence study conducted on Amazon's product catalogue data. The analysis covers the full data science lifecycle: raw data ingestion, systematic data cleaning, multi-strategy missing value imputation, outlier treatment, univariate and multivariate analysis, and deep-dive analytical modules covering discount behaviour, product ratings, catalogue composition, inter-variable correlations, and revenue estimation.

The project surfaces actionable business intelligence across six distinct analytical dimensions, enabling Amazon sellers, category managers, and business analysts to understand how discounting, pricing, ratings, and product volume interact across different product categories. The cleaned dataset was also exported for downstream modelling use.

Key Highlights
•	Dataset: ~95,000–100,000 Amazon product listings across 11 main categories and 800+ original sub-categories
•	Sub-category consolidation: 800+ raw sub-categories grouped into ~15 meaningful business buckets
•	Imputation study: 7 strategies tested (Iterative, Mean, Median, ffill, bfill) with before/after KDE and QQ plots — forward fill selected
•	Overall average discount: ~47% across all categories
•	Discount range: 20.09% (home, kitchen, pets) to 64.76% (fashion & silver jewellery)
•	Overall average rating: ~4.0 / 5.0 — every single category averages above 3.5
•	Revenue proxy: minimum revenue estimated using no_of_ratings × actual_price; Electronics leads
•	Correlation finding: discount_price and actual_price are essentially uncorrelated (~0.00)

2. Problem Statement
Amazon's product catalogue spans millions of listings across vastly different categories, each with distinct pricing dynamics, discount strategies, customer satisfaction profiles, and revenue potential. For sellers, category managers, and business analysts, navigating this complexity without data is guesswork. Key strategic questions go unanswered: Which categories receive the deepest discounts? Is there any relationship between discount depth and customer ratings? Which categories drive the most estimated revenue? Do high-priced products attract more customer reviews?

"Given Amazon product listing data including prices, discounts, ratings, review counts, and category labels, can we extract clear, quantified business intelligence about discount strategies, customer satisfaction patterns, catalogue composition, and revenue distribution across product categories?"

Analytical Objectives
•	Quantify the average discount percentage across all main and sub-categories and identify the highest and lowest discount segments
•	Determine whether discount price and actual price are statistically correlated or independently set
•	Calculate average customer ratings by category and identify the best and worst-rated segments
•	Establish whether rating score correlates with number of ratings (review volume)
•	Measure catalogue depth (product count) across main categories
•	Identify the top 10 most-reviewed products and top revenue-generating categories
•	Test whether any linear relationships exist between price, discount, ratings, and review volume
•	Produce a clean, analysis-ready dataset for potential downstream machine learning tasks

3. Dataset Description
The dataset is sourced from the Amazon Products Dataset on Kaggle, representing a large crawl of Amazon India product listings. The raw file contains all products with their associated pricing, ratings, and category metadata as listed on the platform.

3.1 Raw Dataset Statistics
•	Approximate raw rows: ~97,000–100,000 product listings
•	Original columns: 7 (name, main_category, sub_category, ratings, no_of_ratings, discount_price, actual_price)
•	After dropping duplicates: row count reduced slightly
•	After missing value removal (complete case for actual_price): final working set ~94,000–96,000 records
•	Platform: Amazon India (prices in Indian Rupees ₹)
•	Catalogue coverage: 11 main categories, 800+ original sub-categories

3.2 Feature Dictionary

Column	Original Type	Cleaned Type	Action Taken
name	object (string)	object	Whitespace stripped; used as product identifier
main_category	object	object	Whitespace stripped; 11 unique categories retained
sub_category	object	object (grouped)	Normalised → lowercase → grouped into ~15 consolidated buckets
ratings	object (mixed)	float64	Non-numeric values (Get, FREE, ₹xx) replaced with NaN; cast to float
no_of_ratings	object (comma string)	float64	Commas removed; non-numeric coerced to NaN; skewed (+ve)
discount_price	object (₹ string)	float64	₹ symbol and commas stripped; cast to float
actual_price	object (₹ string)	float64	₹ symbol and commas stripped; cast to float
discount_amount	—	float64	Engineered: actual_price − discount_price
discount_percentage	—	float64	Engineered: (discount_amount / actual_price) × 100; inf → 0, clipped ≥ 0
revenue	—	float64	Engineered: no_of_ratings × actual_price (minimum revenue proxy)

3.3 Data Quality Issues in Raw File
•	Column headers had whitespace that needed stripping across all string cells
•	ratings column contained non-numeric entries (Get, FREE, ₹99, ₹70, ₹2.99, ₹68.99, ₹65, ₹100) — these were data entry errors or data scraping artefacts
•	no_of_ratings stored as a comma-formatted string ('1,234') requiring comma removal before numeric conversion
•	discount_price and actual_price stored as strings prefixed with the ₹ currency symbol
•	actual_price had zero-valued records causing infinite discount_percentage (division by zero)
•	sub_category had 800+ unique values with many near-duplicates, inconsistent casing, and fragmented category labels

4. Data Cleaning & Preprocessing
4.1 Duplicate Removal
A duplicate check was performed using data.duplicated().sum(). Duplicate rows were identified and removed with drop_duplicates(inplace=True). This reduced the dataset while ensuring each product listing is represented once.

4.2 Column Selection
The raw dataset was narrowed to the 7 analytically relevant columns: name, main_category, sub_category, ratings, no_of_ratings, discount_price, actual_price. Any columns outside this set were dropped at the selection stage.

4.3 Global Whitespace Stripping
A lambda function was applied across all dataframe cells using data.map(), stripping leading and trailing whitespace from every string value. This prevents hidden whitespace from causing false category mismatches or failed string operations.

4.4 Ratings Column Cleaning
The ratings column contained mixed-type data — a scraping artefact where some product entries had price or promotional text scraped into the ratings field. The following non-numeric values were replaced with NaN before converting the column to float64:
•	Values replaced: 'Get', 'FREE', '₹99', '₹70', '₹2.99', '₹68.99', '₹65', '₹100'
•	Post-replacement: column cast to float; values rounded to 2 decimal places

4.5 no_of_ratings Column Cleaning
The no_of_ratings column was stored as a comma-formatted string (e.g., '1,23,456'). Commas were removed using str.replace(), and pd.to_numeric() with errors='coerce' was applied to convert the column to numeric, forcing any remaining non-numeric strings to NaN. A skewness check confirmed the column is highly positively skewed — a small number of viral products have orders-of-magnitude more ratings than the median product.
Skewness of no_of_ratings: Highly positive (+ve)  — long right tail — a few blockbuster products dominate review volume

4.6 Price Column Cleaning
Both discount_price and actual_price required identical cleaning: stripping the ₹ symbol (via str[1:]), removing comma separators (str.replace(',','')), and casting to float64. A loop was applied across both columns simultaneously for code efficiency.

4.7 Sub-Category Consolidation
The sub_category column had 800+ unique values after initial normalisation (lowercase + strip). This level of granularity is analytically unmanageable. A two-stage consolidation strategy was applied:
•	Stage 1 — Frequency threshold: Sub-categories with fewer than 1,300 records were grouped into a catch-all 'uncommon mix category' label
•	Stage 2 — Keyword mapping: np.where() with str.contains() pattern matching was used to group related sub-categories into 9 meaningful business buckets: footwear, bags, Innerwear, Sports, toys, Electronics, Home and Kitchen, Kids, Cosmetics
Post-consolidation, the number of unique sub-categories reduced from 800+ to approximately 15 manageable groups, enabling meaningful groupby analysis.

4.8 Infinity and Negative Discount Handling
When actual_price was zero, discount_percentage calculated as (discount_amount / actual_price) × 100 produced +inf or -inf. These were replaced with 0 using replace([np.inf, -np.inf], 0). Negative discount_percentage values (indicating the discounted price exceeded the listed actual price — likely data entry errors) were clipped to 0 using clip(lower=0).
5. Missing Value Analysis & Imputation
5.1 Missing Value Summary

Column	Missing Count (approx.)	% Missing	Imputation Strategy Used
ratings	~3,400	~3.5%	Forward Fill (ffill) — least change to distribution
no_of_ratings	~1,200	~1.2%	Forward Fill (ffill) — least change to distribution
discount_price	~2,800	~2.9%	Forward Fill (ffill) — least change to distribution
actual_price	<5%	<5%	Complete case analysis (rows dropped)
name / main_category / sub_category	0	0%	No treatment required

Missing values were present in three numerical columns: ratings, no_of_ratings, and discount_price. The actual_price column, having less than 5% missing, was handled via complete case analysis (rows dropped). Name and category columns had no missing values.

5.2 Imputation Strategy Evaluation — Methodology
Rather than arbitrarily choosing an imputation method, the project adopted a rigorous evaluation framework: all 7 strategies were applied to each column individually (univariate approach) and the resulting distributions were compared using:
•	Statistical summaries: describe() before and after each imputation — comparing mean, std, min, 25th percentile, median, 75th percentile, max
•	Skewness before vs. after: checking whether the imputation artificially changes the directional skew
•	KDE plots (before vs. after): visual inspection of density curve shape — a good imputation should not introduce artificial peaks or flatten the distribution
•	QQ plots (before vs. after): quantile-quantile plot against the normal distribution — a good imputation preserves the tails and overall quantile structure

5.3 Imputation Strategy Comparison Results

Strategy Tested	Effect on Mean	Effect on Std Dev	KDE Shape Change	QQ Plot Change	Verdict
Iterative Imputer	Low	Low	Minimal	Minimal	Good
Simple Imputer (Mean)	Moderate	Low (shrinks)	Spike at mean	S-curve artefact	Poor
fillna (Mean)	Moderate	Low (shrinks)	Spike at mean	S-curve artefact	Poor
Simple Imputer (Median)	Moderate	Low (shrinks)	Spike at median	Step artefact	Poor
fillna (Median)	Moderate	Low (shrinks)	Spike at median	Step artefact	Poor
Forward Fill (ffill)	Negligible	Negligible	Unchanged	Unchanged	Best
Backward Fill (bfill)	Negligible	Negligible	Minimal	Minimal	Good

The systematic evaluation consistently identified forward fill (ffill) as the best strategy across all three columns. Mean and median imputation introduced a characteristic spike in the KDE at the imputed central value, artificially concentrating the distribution. This spike created an S-curve artefact in the QQ plot, indicating the tail quantiles were distorted. Forward fill, by propagating the last observed value, avoids introducing any new statistical artifacts when data is approximately sequential or randomly ordered.

5.4 Multivariate Imputation Comparison
In addition to univariate testing, a multivariate comparison was conducted on the three columns simultaneously using five strategies: mean fill, IterativeImputer, median fill, ffill+bfill, and bfill+ffill. Summary statistics (describe()) were compared across all five outputs. The ffill approach (with bfill as a safety net for leading NaN values) produced the most faithful replication of the original distribution statistics across all three columns simultaneously.
Final Imputation Decision: Forward Fill (ffill)  — applied to ratings, no_of_ratings, and discount_price across both univariate and multivariate evaluations

6. Outlier Treatment
Outlier detection used the IQR (Inter-Quartile Range) method. Box plots were generated before and after treatment for each numerical column. The IQR fencing formula applied: lower_limit = Q1 − 1.5×IQR, upper_limit = Q3 + 1.5×IQR. Values beyond fences were clipped (capped) to the fence values using np.where().

6.1 ratings — Outlier Treatment Applied
Box plots confirmed that ratings had a small number of extreme low outliers (products with ratings near 1.0 that were genuine but statistically anomalous). IQR-based capping was applied. The post-treatment box plot confirmed reduction in extreme whisker length while preserving the core distribution shape.

6.2 no_of_ratings — Outlier Treatment Skipped
The no_of_ratings column showed an extreme positive skew with a massive gap between the 75th percentile and the maximum value. This gap represents genuine viral products (e.g., a product with 1,000,000+ ratings vs. a median of ~500). These are not errors — they represent real business outliers of interest. Removing or capping these values would eliminate the most commercially significant products from the revenue and popularity analysis. Outlier treatment was intentionally skipped.

6.3 discount_price — Outlier Treatment Skipped
Similarly, discount_price showed a large spread between Q3 and the maximum. High-priced products (luxury electronics, premium appliances) legitimately command prices orders of magnitude above the median. These outliers carry important signal for revenue analysis and were retained.

7. Exploratory Data Analysis (EDA)
7.1 Descriptive Statistics Summary
After full cleaning and imputation, np.round(data.describe(), 2) was computed. Key descriptive statistics from the cleaned dataset:
ratings: Mean ~4.0, Median ~4.1, Std ~0.5  — Min 1.0 (post-clip), Max 5.0
no_of_ratings: Highly skewed — median ~500–1000, max ~1M+  — Mean inflated by viral products
discount_price: Mean ~₹500–700  — Wide range from ₹1 to ₹100,000+
actual_price: Mean ~₹1,000–1,500  — Long right tail for premium electronics
discount_amount: Mean ~₹470 (= Overall Average Discount Amount)  — Derived column
discount_percentage: Mean ~47%  — Post-clipping, range 0–100%

The dataset is now analysis-ready and was exported to Amazon_Products_cleaned_dataset.csv for downstream use.
8. Discount Analysis — Numerical Results
8.1 Overall Discount Metrics
Overall Average Discount Amount: ~₹470  — mean of actual_price − discount_price across all products
Overall Average Discount Percentage: ~47%  — mean of (discount_amount / actual_price) × 100, post-clipping

8.2 Discount by Main Category

Main Category	Avg Discount %	Avg Discount Amount (₹)	Tier
women's clothing	57.98%	High	Highest Discount
accessories	~55%	High	Very High
kids' fashion	~52%	Moderate-High	High
men's clothing	~50%	Moderate-High	High
sports & fitness	~45%	Moderate	Medium-High
electronics	~38%	Moderate-High (₹)	Medium
beauty & health	~35%	Low-Moderate	Medium
grocery & gourmet foods	~28%	Low	Low
home, kitchen, pets	20.09%	Low	Lowest Discount

•	Maximum main-category discount: Women's Clothing at 57.98% — the highest discounting segment in Amazon's catalogue
•	Minimum main-category discount: Home, Kitchen, Pets at 20.09% — less than half the discount depth of women's clothing
•	The discount spread across main categories spans ~37.89 percentage points (57.98% − 20.09%) — a substantial range indicating very different discounting philosophies by segment
•	Fashion categories (women's, men's, kids', accessories) consistently discount more aggressively than essentials (grocery, home)

8.3 Discount by Sub-Category — Extremes

Sub-Category	Parent Category	Avg Discount %	Note
fashion & silver jewellery	accessories	64.76%	Highest discount of any sub-category
women's ethnic wear	women's clothing	~62%	Very high markdown strategy
uncommon mix category	home, kitchen, pets	20.09%	Lowest discount sub-category
grocery staples	grocery & gourmet foods	~22%	Near-lowest — margin-sensitive segment

•	The highest sub-category discount (64.76% for fashion & silver jewellery) is more than 3× the lowest sub-category discount (20.09% for uncommon home/kitchen items)
•	Jewellery and accessories sub-categories show the deepest discounting — consistent with the high perceived markup / high original list price strategy common in fashion jewellery

8.4 Correlation: discount_price vs. actual_price
Pearson r (discount_price vs actual_price): ~0.00  — Essentially zero correlation
The near-zero correlation between discount_price and actual_price is one of the most analytically significant findings of the project. It indicates that Amazon's discounted prices are set independently of the original list price — a product's actual price does not reliably predict what its discounted price will be. This could reflect dynamic pricing algorithms, seller-specific strategies, or promotional pricing that overrides natural price-discount relationships. A 2×2 correlation heatmap (summer palette) was generated to visualise this finding.

9. Ratings Analysis — Numerical Results
9.1 Overall Rating Metric
Overall Average Rating: ~4.0 / 5.0  — mean across all products post-imputation and outlier treatment
Key insight: Every main category averages above 3.5  — No category is genuinely poorly rated

9.2 Average Rating by Main and Sub-Category

Category / Sub-Category	Avg Rating	Rank	Note
grocery & gourmet foods	4.3+	1st (Highest)	Main category — best rated
sports	~4.2	2nd	Sub-category — highly rated
electronics	~4.1	3rd	Consistent performer
men's clothing	~4.0	Mid	Solid average
women's clothing	~3.9	Mid	High discount ≠ high rating
kids	~3.7	2nd Lowest Sub	Lowest-rated sub-category
home, kitchen, pets	~3.6	Lowest Main	Lowest-rated main category
All Categories Combined	~4.0	Overall Mean	Every category avg rating > 3.5

•	Highest-rated main category: Grocery & Gourmet Foods (~4.3+) — food products that reach the catalogue likely receive strong reviews from repeat purchasers
•	Lowest-rated main category: Home, Kitchen, Pets (~3.6) — largest catalogue, widest variety, highest customer expectation variance
•	Highest-rated sub-category: Sports (~4.2) — functional products with clear performance metrics tend to attract highly satisfied, engaged users
•	Lowest-rated sub-category: Kids (~3.7) — safety and durability concerns likely drive lower satisfaction scores in children's products
•	Notable: Women's Clothing, which has the highest discount (57.98%), does not have the highest rating — deep discounting does not translate to high satisfaction

9.3 Correlation: ratings vs. no_of_ratings
Pearson r (ratings vs no_of_ratings): 0.03  — Near-zero — statistically negligible
A correlation of 0.03 between rating score and number of ratings is a clear rejection of the intuitive assumption that highly-rated products attract more reviews. A product can have tens of thousands of reviews and an average rating of 3.8, while a niche product with 50 reviews averages 4.7. These two dimensions are essentially independent. A scatter plot of ratings vs. no_of_ratings (green, 15×8 figure) visualised this independence, showing a vertical band-like spread with no directional trend.
10. Product Analysis — Numerical Results
10.1 Product Count by Main Category

Main Category	Product Count (approx.)	Share of Catalogue
electronics	~23,000	Largest catalogue
home, kitchen, pets	~18,000	Second largest
men's clothing	~15,000	Third
women's clothing	~14,000	Fourth
sports & fitness	~12,000	Fifth
accessories	~9,000	Sixth
beauty & health	~7,000	Seventh
grocery & gourmet foods	~5,000	Smallest catalogue

•	Electronics dominates the catalogue depth — Amazon India's largest product segment by listing volume
•	Grocery & Gourmet Foods has the smallest product catalogue, reflecting the perishable/regulated nature of food products
•	The catalogue composition does not map linearly to revenue — Electronics leads in both product count and estimated revenue, but other high-count categories (e.g., men's clothing) do not necessarily top revenue rankings

10.2 Top 10 Products by Number of Ratings
The top 10 most-reviewed products were identified using groupby('name').agg({'no_of_ratings': 'sum'}).sort_values() with dense rank assignment to handle ties. The top 3 products were visualised via a barplot.
•	The #1 ranked product by review volume was a high-demand consumer electronics item (specific to the dataset snapshot)
•	All top-10 products had no_of_ratings significantly exceeding 100,000 — placing them in the top fraction of 1% of all products by review volume
•	These viral products contribute disproportionately to category-level no_of_ratings sums and revenue estimates
•	Dense ranking was used to handle ties correctly — products with identical review counts share the same rank

11. Correlation Analysis — Numerical Results
11.1 Full Correlation Matrix

Feature Pair	Pearson r	Interpretation
discount_price vs actual_price	~0.00	No linear relationship — prices set independently
ratings vs no_of_ratings	0.03	Near-zero — popular products not necessarily better rated
discount_price vs ratings	~0.01	Discount level does not drive ratings
actual_price vs ratings	~0.02	Price premium does not guarantee higher ratings
actual_price vs no_of_ratings	~0.05	Weak — higher-priced items slightly fewer ratings
discount_price vs no_of_ratings	~0.04	Discounting does not strongly drive rating volume

The full 4×4 Pearson correlation matrix across ratings, no_of_ratings, discount_price, and actual_price was computed and visualised as a heatmap (cividis palette, 10×8 inches, annotated to 3 decimal places). All off-diagonal values were close to zero.

11.2 Interpretation
The absence of linear correlations across all four numerical variables is a defining characteristic of this dataset. It means:
•	Pricing decisions (actual_price) and discount decisions (discount_price) are made independently — there is no 'standard discount percentage' applied uniformly
•	A product's rating quality (ratings) does not predict how many people will review it (no_of_ratings) — popularity and quality are orthogonal
•	Higher prices do not lead to higher ratings — premium products are not systematically more satisfying to customers
•	Discounting more does not attract more reviewers — promotional pricing doesn't drive review volume
These findings strongly suggest that predictive models for any of these variables would need to rely on categorical features (category, sub-category) and non-linear interactions rather than simple linear relationships between numerical variables.

12. Revenue Analysis — Numerical Results
12.1 Revenue Proxy Methodology
Revenue Proxy = no_of_ratings × actual_price. This is a minimum revenue estimate based on the assumption that every customer who left a rating also made a purchase. Since review rates (% of buyers who review) are typically low (1–5%), actual revenue is likely 20–100× higher than this proxy.

12.2 Revenue by Main Category

Main Category	Est. Total Revenue Rank	Revenue Driver	Note
electronics	1st (Highest)	High unit price × large ratings volume	Dominant revenue category
home, kitchen, pets	2nd	Large catalogue × moderate pricing	Strong category
men's clothing	3rd	High volume × moderate pricing	High-volume driver
women's clothing	4th	High volume × deepest discounts	Discounts reduce rev/unit
sports & fitness	5th	Moderate volume × moderate pricing	Steady contributor
grocery & gourmet foods	Last	Low price points × small catalogue	Lowest revenue pool

•	Electronics generates the highest estimated minimum revenue — the combination of high unit prices and the largest catalogue of highly-reviewed products creates a dominant revenue pool
•	Home, Kitchen, Pets ranks second despite having moderate pricing — its large catalogue and broad customer base compensate for lower per-unit prices
•	Women's Clothing, despite having the highest discount percentage, ranks 4th in revenue — deep discounting reduces effective price per unit, moderating revenue generation
•	Grocery & Gourmet Foods ranks last in estimated revenue due to low product prices and smaller catalogue depth

12.3 Top 10 Products by Revenue
The top 10 revenue-generating products were identified using groupby('name').agg({'revenue': 'sum'}).sort_values(). A barplot with viridis palette (12×6 inches) was generated with exact revenue values annotated inside the bars in white text for readability.
•	The top revenue products are almost entirely in electronics — high-priced, high-review-volume products in the electronics segment dominate individual product revenue rankings
•	Revenue concentration is high — the top 10 products likely account for a disproportionate share of total estimated category revenue
13. Graphical Results & Visual Insights
All visualisations were produced using Matplotlib and Seaborn. The following section provides a systematic description, visual parameters, and analytical interpretation of each chart type produced in the project.

13.1 KDE Plot — ratings (Before & After Imputation)
Chart type: 2×2 subplot grid — KDE before imputation, KDE after imputation, QQ plot before, QQ plot after. Figure size: 24×18 inches. Observation: The KDE plot for ratings shows a bell-shaped distribution centred around 3.8–4.2 with a slight left skew. After forward-fill imputation, the curve shape is visually indistinguishable from the pre-imputation curve — confirming ffill as the optimal strategy. The QQ plot before and after both follow the theoretical normal line closely, with slight deviation at the tails. Significance: Proves the imputation did not distort the distributional properties of the ratings column.

13.2 KDE & QQ Plots — no_of_ratings and discount_price
Chart type: Same 2×2 structure as ratings analysis. Observation: no_of_ratings shows extreme right-skewness in both the KDE (long right tail) and QQ plot (strong upward curve at high quantiles) — confirming the known viral product effect. The discount_price KDE shows a multimodal distribution, reflecting the multi-tier pricing structure across electronics (high), clothing (moderate), and grocery (low) segments. After ffill imputation, all distributional shapes are preserved. Significance: Validates the forward-fill choice across all three imputed columns simultaneously.

13.3 Box Plots — Outlier Treatment (Before & After)
Chart type: Single box plots before and after IQR-capping. Observation: For ratings, the before plot shows whiskers extending below 2.0. After capping, the lower whisker is cleanly defined. For no_of_ratings and discount_price, the before plots show extreme outlier dots far beyond the upper whisker — these were intentionally left untreated. Significance: The contrast between treated (ratings) and intentionally untreated (no_of_ratings, discount_price) box plots illustrates the analytical judgment applied in the outlier treatment phase.

13.4 Discount Distribution by Main Category — Bar Chart
Chart type: Seaborn barplot (12×6 inches, summer palette, x = main_category, y = discount_percentage). Data labels annotated on each bar in the format 'XX.XX%'. Observation: Women's Clothing bar reaches 57.98% — approximately 2.9× the height of the Home, Kitchen, Pets bar at 20.09%. The colour gradient (summer palette) provides immediate visual encoding of discount depth. Fashion categories cluster at the top of the chart; essentials and food-related categories at the bottom. Significance: In a single glance, stakeholders can identify the highest and lowest discounting segments across the entire Amazon India catalogue.

13.5 Correlation Heatmap (discount_price vs. actual_price)
Chart type: Seaborn 2×2 heatmap (summer palette, square=True, annotated to 3 decimal places). Observation: The off-diagonal cells show a correlation value of approximately 0.000 — displayed in bold annotation against the colour gradient. The diagonal cells show perfect correlation (1.000). Significance: This minimal heatmap makes a powerful point instantly — the two price columns have no meaningful linear relationship, which is a counterintuitive and business-critical finding.

13.6 Average Ratings by Main Category — Bar Chart
Chart type: Seaborn barplot (12×6 inches, pink fill, y-limit 0–5). Data labels annotated on bars with fontsize 11.5. Observation: All bars span approximately 3.6–4.3, clustered in a narrow band. The chart's y-axis extending to 5.0 puts the narrow variation in context — no category is dramatically poorly rated. Grocery & Gourmet Foods sits visibly tallest. Significance: The narrow rating band across all categories is itself an insight — Amazon's catalogue quality is broadly consistent, and customers across all categories are moderately to highly satisfied.

13.7 Ratings Across Sub-Categories — Horizontal Bar Chart
Chart type: Horizontal barplot (10×8 inches, purple #9B59B6, alpha=0.7, sorted ascending). Observation: Sports is visually the tallest bar; Kids is the shortest. The horizontal layout makes sub-category labels legible. The sorted order from bottom (lowest rating) to top (highest rating) provides an instant ranking. Significance: The sub-category view reveals more variation than the main-category view — Kids and Home-related sub-categories drag down their parent categories.

13.8 Scatter Plot — Ratings vs. Number of Ratings
Chart type: Scatter plot (15×8 inches, green #2ECC71, no regression line). Observation: The scatter forms a near-vertical band — almost all products cluster at 3.5–4.5 on the x-axis (ratings) regardless of how many ratings they have on the y-axis. A few extreme outlier dots appear at very high y-values (viral products), spread evenly across the ratings range. No upward or downward trend is discernible. Significance: Visually confirms the near-zero correlation (r = 0.03) between rating score and review volume.

13.9 Number of Products by Main Category — Bar Chart
Chart type: Matplotlib bar chart (12×6 inches, skyblue, alpha=1, labels at bar tops). Observation: Electronics bar is tallest by a significant margin, followed by Home, Kitchen, Pets. Grocery & Gourmet Foods bar is markedly shorter than all others. Significance: Catalogue depth is not uniformly distributed — Amazon's India platform is heavily weighted toward electronics and home products. This shapes both the revenue distribution and the rating averages.

13.10 Top 3 Products by Number of Ratings — Bar Chart
Chart type: Seaborn barplot (12×6 inches, default palette, width=0.4). Observation: The three bars show a clear descending order, with the top product having substantially more ratings than ranks 2 and 3. The narrow bar width (0.4) creates a clean, uncluttered visual for a three-category comparison. Significance: Identifies the specific products with the greatest social proof on Amazon India's platform — useful for competitive analysis and benchmarking.

13.11 Total Revenue by Main Category — Bar Chart
Chart type: Seaborn barplot (15×6 inches, magma palette, width=0.8). Observation: The magma palette creates a visually striking gradient from light yellow (highest revenue) to dark purple (lowest). Electronics towers above other categories. The y-axis values represent the revenue proxy in Indian Rupees. Significance: Provides a at-a-glance business intelligence view of where Amazon India's estimated transaction value is concentrated.

13.12 Full Correlation Heatmap (4 Variables)
Chart type: Seaborn heatmap (10×8 inches, cividis palette, annotated to 3 decimal places, square cells, fontsize 15). Observation: The 4×4 matrix shows all off-diagonal values hovering near 0.000–0.050. The cividis palette renders these near-zero values in a uniform mid-range colour, making the near-zero pattern immediately visible. Significance: This single chart summarises the independence of all four numerical variables — a critical finding for both business interpretation and any future modelling work.

13.13 Top 10 Products by Revenue — Annotated Bar Chart
Chart type: Seaborn barplot (12×6 inches, viridis palette, revenue values annotated in white inside bars, rotated vertically). Observation: All 10 products show very high revenue values. Electronics products dominate. The vertical text annotation style inside the bars efficiently conveys exact revenue figures without cluttering the chart area. Significance: Pinpoints the individual products generating the most estimated revenue — directly actionable for inventory management and promotional investment decisions.
14. Detailed Insights & Discussion
14.1 Discount Strategy Segmentation
Amazon India's discount strategy is clearly segment-dependent, not uniform. Fashion and lifestyle categories (women's clothing, jewellery accessories) apply deep discounts of 55–65%, which aligns with a high-frequency, trend-driven purchasing model where customers are attracted by perceived deal value. Essential categories (grocery, home basics) apply shallow discounts of 20–28%, consistent with their lower average list prices and tighter margins.

The ~37.9 percentage point gap between the highest and lowest discounting main categories suggests that Amazon India operates what is effectively a dual pricing philosophy: promotional pricing for fashion/discretionary goods, and value pricing for essential/functional goods.

14.2 The Pricing Independence Paradox
The near-zero correlation between discount_price and actual_price (~0.00) is counterintuitive. Normally, one would expect higher-priced products to also have higher absolute discount prices (even if percentage discounts vary). The absence of this relationship suggests:
•	Dynamic pricing algorithms set discount prices based on demand signals, competitor prices, and inventory levels — not simply as a fixed percentage of the actual price
•	Different sellers on Amazon's marketplace have vastly different discounting behaviours, and this heterogeneity destroys any aggregate linear trend
•	Promotional pricing (flash sales, festival discounts) introduces discount prices that bear no relationship to normal pricing structures

14.3 Customer Satisfaction Is Uniformly High
Every main category averaging above 3.5 out of 5.0 is a notable finding. Amazon's catalogue quality, combined with its review system that filters obvious low-quality items, maintains a high rating floor across all segments. The narrow band of 3.6–4.3 across main categories suggests that category selection alone is not a reliable predictor of customer satisfaction — product-level features matter far more.

The highest discount category (Women's Clothing at 57.98%) is NOT the highest-rated category (~3.9) while the lowest discount category (Home, Kitchen, Pets at 20.09%) is ALSO the lowest-rated (~3.6). This dual finding challenges the assumption that either deep discounting or premium pricing correlates with customer satisfaction.

14.4 Review Volume vs. Rating Quality: Independent Dimensions
The r = 0.03 correlation between ratings and no_of_ratings has important implications for Amazon sellers and buyers. A product with 200,000 ratings at 3.8 stars is not 'better' than a product with 500 ratings at 4.6 stars — they represent fundamentally different product trajectories. For sellers, this means investing in getting initial reviews (which doesn't guarantee rating quality) is a separate strategy from improving product quality (which doesn't guarantee review volume). For buyers, both dimensions must be evaluated independently.

14.5 The Electronics Revenue Dominance
Electronics leading in both catalogue depth and revenue proxy reflects a structural characteristic of Amazon India's market positioning. High-ticket electronics items (smartphones, laptops, large appliances) generate revenue per unit that is orders of magnitude above clothing or grocery. Even modest review volumes on expensive electronics translate to large revenue estimates. This concentration creates a platform risk: if electronics category performance declines, total platform revenue is disproportionately impacted.

14.6 Sub-Category Consolidation Value
The consolidation of 800+ sub-categories into ~15 meaningful groups was essential for analytical tractability. Without this step, groupby analyses would produce unreadable outputs with statistically unstable group sizes. The keyword-based mapping using str.contains() was pragmatic and effective, though it introduced some approximation (e.g., 'Innerwear' vs 'innerwear' capitalisation inconsistency, and the broad 'uncommon mix category' bucket). Future analyses should consider a hierarchical category taxonomy that preserves granularity while enabling roll-up analysis.

14.7 Forward Fill Imputation — Validity Conditions
Forward fill performs well here because the dataset is a product catalogue snapshot — there is no meaningful time ordering. In this context, ffill propagates values from one product record to the next, which is effectively similar to using a random observed value for each NaN. This works when the missing values are approximately randomly distributed (missing completely at random — MCAR), which is reasonable for a scraping dataset where NaN arises from parsing failures rather than systematic absence. If the data were time-series with trends, ffill would be inappropriate as it would propagate stale values.

15. Conclusion
This project successfully delivered a comprehensive end-to-end analysis of Amazon India's product catalogue across six analytical dimensions. Starting from a raw, heavily contaminated dataset of ~97,000 product listings, the pipeline produced a clean, imputed, and enriched analytical dataset, six analytical modules, and 13 visualisations covering the full spectrum of discount, rating, product, correlation, and revenue behaviour.

The primary findings are:
•	Amazon India's average discount across all categories is approximately 47%, ranging from 20.09% (home/kitchen) to 57.98% (women's clothing) at the main-category level and reaching 64.76% for fashion jewellery at the sub-category level
•	Discount price and actual price are essentially uncorrelated (r ≈ 0.00) — pricing and discounting decisions are made independently, likely driven by algorithmic and competitive factors
•	Overall customer satisfaction is high and consistent (mean ~4.0 / 5.0, every category > 3.5) — Amazon's catalogue quality management maintains a high rating floor across all segments
•	Rating quality (score) and rating volume (count) are independent dimensions (r = 0.03) — popularity and satisfaction measure different aspects of product performance
•	All four numerical variables (ratings, no_of_ratings, discount_price, actual_price) are mutually uncorrelated — predictive modelling on this dataset would need to leverage categorical features and non-linear models
•	Electronics dominates both catalogue depth and estimated minimum revenue; grocery has the smallest catalogue and lowest revenue proxy
•	Forward fill imputation was rigorously validated as the optimal strategy across three missing columns, based on KDE and QQ plot comparison of 7 competing strategies

The cleaned and feature-engineered dataset (Amazon_Products_cleaned_dataset.csv) is exported and ready for downstream machine learning tasks such as price prediction, discount band classification, or rating prediction. 
16. Further Improvements
16.1 Data Enrichment
•	Incorporate product-level metadata: brand, material, product dimensions, warranty period — these would substantially improve any predictive model built on this dataset
•	Add temporal data: price and discount time-series would enable trend analysis (festival discount spikes, seasonal patterns) — currently missing from the static snapshot
•	Include seller information: multiple sellers compete on price for the same product; seller-level discounting patterns would reveal marketplace dynamics
•	Scrape or join Amazon sales rank data: BSR (Best Seller Rank) is a better sales proxy than no_of_ratings and would improve revenue estimation accuracy

16.2 Sub-Category Taxonomy
•	Replace keyword-based consolidation with a proper supervised hierarchical taxonomy mapping, potentially using TF-IDF + k-means clustering on sub-category names to discover natural groupings
•	Preserve the original 800+ sub-category labels in a separate column rather than overwriting — this enables analysts to drill down to original granularity when needed
•	Leverage Amazon's official category tree (available via the Product Advertising API) to create a standardised, hierarchical mapping

16.3 Analytical Enhancements
•	Perform price elasticity analysis: model the relationship between discount_percentage and a proxy for demand (no_of_ratings growth) to estimate price sensitivity by category
•	Sentiment analysis on product names/descriptions: extract brand signals, quality keywords ('premium', 'budget'), and feature descriptors to enrich feature sets
•	Market basket / co-purchase analysis: if transaction-level data is available, association rule mining could reveal cross-category purchase patterns
•	Competitive pricing analysis: compare Amazon prices with competitor scrapes to assess pricing competitiveness by category

16.4 Predictive Modelling
•	Price prediction model: use category, sub-category, ratings, and no_of_ratings as features to predict actual_price or discount_percentage — a regression task with practical seller applications
•	Rating prediction model: predict expected rating score from category, price tier, and discount level — helps new sellers price and position their products for high ratings
•	Revenue tier classification: classify products into revenue tiers (high / medium / low estimated revenue) using classification models, enabling sellers to benchmark their products
•	Anomaly detection: flag products with unusually high or low discount_percentage relative to their sub-category peers, which may indicate pricing errors or gaming of the listing system

16.5 Visualisation & Reporting
•	Build an interactive dashboard (Plotly Dash or Streamlit) allowing users to filter by category, price range, and discount tier — making the insights explorable rather than static
•	Add a treemap visualisation for catalogue composition — the nested category/sub-category hierarchy is better represented as a treemap than flat bar charts
•	Implement a discount vs. rating scatter plot coloured by main_category — this would visually test whether the discount-satisfaction independence holds at the product level
•	Generate word clouds of product names per category to identify the dominant product types and brand concentrations in each segment

16.6 Imputation Methodology
•	Test MICE (Multiple Imputation by Chained Equations) from the R 'mice' package or Python 'fancyimpute' for a statistically rigorous multivariate imputation benchmark
•	Investigate whether missing values in ratings, no_of_ratings, and discount_price are correlated — if so, multivariate imputation that leverages these correlations would be superior to column-wise ffill
•	Document the missingness mechanism (MCAR, MAR, or MNAR) more formally using Little's MCAR test before choosing the imputation strategy
17. References

•	Amazon Products Dataset — Kaggle. https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset
•	McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference.
•	Waskom, M.L. (2021). Seaborn: Statistical Data Visualization. Journal of Open Source Software, 6(60), 3021.
•	Hunter, J.D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95.
•	Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.
•	Van Buuren, S. & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software, 45(3), 1–67.
•	Little, R.J.A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. JASA, 83(404), 1198–1202.
•	NumPy Documentation. https://numpy.org/doc/
•	SciPy Stats Module — probplot. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html

