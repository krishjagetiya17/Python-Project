## Flipkart Products 
import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
IMG = HERE / "images"
SEED = 42

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 180)


# --------------------------------------------------
# 1. DATA LOADING AND CLEANING
# --------------------------------------------------

def load_products(path):
    df = pd.read_csv(path)
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        for c in df.columns
    ]

    # Support different CSV column names
    aliases = {
        "product_name": [
            "product_name", "product", "name", "title",
            "product_title", "productname"
        ],
        "price": [
            "price", "product_price", "selling_price",
            "discounted_price", "sale_price"
        ],
        "rating": [
            "rating", "product_rating", "rate", "ratings",
            "overall_rating"
        ],
        "reviews": [
            "reviews", "review", "review_count",
            "num_reviews", "rating_count"
        ],
        "category": [
            "category", "product_category", "main_category",
            "product_type"
        ],
        "description": [
            "description", "summary", "product_description",
            "about_product"
        ]
    }

    rename = {}

    for standard, choices in aliases.items():
        for col in choices:
            if col in df.columns:
                rename[col] = standard
                break

    df = df.rename(columns=rename)

    if "product_name" not in df.columns:
        raise ValueError(
            "CSV must contain a product name column. "
            "Expected: product_name, product, name or title."
        )

    for col in ["price", "rating", "reviews"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["category", "description"]:
        if col not in df.columns:
            df[col] = ""

    # Clean currency symbols and comma-separated numbers
    for col in ["price", "rating", "reviews"]:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[₹$€£]", "", regex=True)
            .str.extract(r"(-?\d+\.?\d*)", expand=False),
            errors="coerce"
        )

    df["product_name"] = df["product_name"].fillna("").str.strip()
    df["category"] = df["category"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    # Remove invalid product names and duplicate products
    df = df[df["product_name"] != ""].copy()
    df = df.drop_duplicates(subset="product_name")

    # Invalid values become missing
    df.loc[df["price"] <= 0, "price"] = np.nan
    df.loc[~df["rating"].between(0, 5), "rating"] = np.nan
    df.loc[df["reviews"] < 0, "reviews"] = np.nan

    # Convert missing text values to useful defaults
    df["category"] = df["category"].replace("", "Unknown")
    df["description"] = df["description"].replace("", "No description")

    # Create text used by the recommender
    df["features"] = (
        df["product_name"] + " " +
        df["category"] + " " +
        df["description"]
    )

    return df.reset_index(drop=True)


def save_clean(df):
    columns = [
        "product_name", "price", "rating",
        "reviews", "category", "description"
    ]

    df[columns].to_csv(HERE / "flipkart_clean.csv", index=False)
    print("Saved flipkart_clean.csv")


# --------------------------------------------------
# 2. VISUALIZATION HELPERS
# --------------------------------------------------

def save_plot(fig, filename):
    IMG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(df):
    # 01. Missing values
    missing = df[
        ["product_name", "price", "rating", "reviews", "category"]
    ].isna().mean().mul(100)

    fig, ax = plt.subplots(figsize=(7, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing values (%)")
    ax.tick_params(axis="x", rotation=30)
    save_plot(fig, "01_missing_values.png")

    # 02. Products by category
    counts = df["category"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Product Categories")
    ax.set_xlabel("Number of products")
    save_plot(fig, "02_category_distribution.png")

    # 03. Price distribution
    prices = df["price"].dropna()
    prices = prices[prices <= prices.quantile(0.99)]

    fig, ax = plt.subplots(figsize=(8, 4))
    if not prices.empty:
        sns.histplot(prices, bins=30, kde=True, ax=ax)
    ax.set_title("Product Price Distribution")
    ax.set_xlabel("Price")
    save_plot(fig, "03_price_distribution.png")

    # 04. Rating distribution
    ratings = df["rating"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    if not ratings.empty:
        sns.histplot(ratings, bins=20, ax=ax)
    ax.set_title("Product Rating Distribution")
    ax.set_xlabel("Rating out of 5")
    save_plot(fig, "04_rating_distribution.png")

    # 05. Average price by category
    category_price = (
        df.dropna(subset=["price"])
        .groupby("category")["price"]
        .mean().nlargest(12).sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    category_price.plot(kind="barh", ax=ax)
    ax.set_title("Highest Average Price by Category")
    ax.set_xlabel("Average price")
    save_plot(fig, "05_average_price_category.png")

    # 06. Average rating by category
    category_rating = (
        df.dropna(subset=["rating"])
        .groupby("category")["rating"]
        .mean().nlargest(12).sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    category_rating.plot(kind="barh", ax=ax)
    ax.set_title("Highest Average Rating by Category")
    ax.set_xlabel("Average rating")
    save_plot(fig, "06_category_ratings.png")

    # 07. Top-rated products
    top_rated = df.dropna(subset=["rating"]).nlargest(10, "rating")

    fig, ax = plt.subplots(figsize=(10, 6))
    if not top_rated.empty:
        ax.barh(
            top_rated["product_name"].str.slice(0, 35).iloc[::-1],
            top_rated["rating"].iloc[::-1]
        )
    ax.set_title("Top 10 Highest-Rated Products")
    ax.set_xlabel("Rating")
    save_plot(fig, "07_top_rated.png")

    # 08. Most reviewed products
    top_reviewed = df.dropna(subset=["reviews"]).nlargest(10, "reviews")

    fig, ax = plt.subplots(figsize=(10, 6))
    if not top_reviewed.empty:
        ax.barh(
            top_reviewed["product_name"].str.slice(0, 35).iloc[::-1],
            top_reviewed["reviews"].iloc[::-1]
        )
    ax.set_title("Top 10 Most-Reviewed Products")
    ax.set_xlabel("Number of reviews")
    save_plot(fig, "08_most_reviewed.png")

    # 09. Price versus rating
    d = df.dropna(subset=["price", "rating"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if not d.empty:
        ax.scatter(d["price"], d["rating"], alpha=0.5)
    ax.set_title("Product Price vs Rating")
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")
    save_plot(fig, "09_price_vs_rating.png")

    # 10. Rating by category
    d = df.dropna(subset=["rating"]).copy()
    categories = d["category"].value_counts().head(10).index
    d = d[d["category"].isin(categories)]

    fig, ax = plt.subplots(figsize=(10, 5))
    if not d.empty:
        sns.boxplot(data=d, x="category", y="rating", ax=ax)
    ax.set_title("Rating Distribution by Category")
    ax.tick_params(axis="x", rotation=35)
    save_plot(fig, "10_rating_by_category.png")

    # 11. Product price ranges
    price_data = df.dropna(subset=["price"]).copy()
    bins = [0, 500, 1000, 5000, 10000, 50000, np.inf]
    labels = [
        "0-500", "501-1000", "1001-5000",
        "5001-10000", "10001-50000", "50000+"
    ]

    price_data["price_range"] = pd.cut(
        price_data["price"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    price_data["price_range"].value_counts(
        sort=False
    ).plot(kind="bar", ax=ax)
    ax.set_title("Products by Price Range")
    ax.set_xlabel("Price range")
    ax.set_ylabel("Number of products")
    ax.tick_params(axis="x", rotation=25)
    save_plot(fig, "11_price_ranges.png")

    # 12. Category and rating correlation
    d = df.dropna(subset=["price", "rating"]).copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(d) >= 2 and d["price"].nunique() > 1:
        sns.heatmap(
            d[["price", "rating"]].corr(),
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
    else:
        ax.text(0.5, 0.5, "Insufficient data for correlation",
                ha="center", va="center")
        ax.axis("off")

    ax.set_title("Price and Rating Correlation")
    save_plot(fig, "12_price_rating_correlation.png")


# --------------------------------------------------
# 3. POPULARITY RECOMMENDER
# --------------------------------------------------

class PopularityRecommender:
    """Rank products using rating and review count."""

    def __init__(self, df):
        self.df = df.copy()

        valid = df.dropna(subset=["rating"]).copy()
        if valid.empty:
            self.table = valid
            return

        valid["reviews"] = valid["reviews"].fillna(0)

        # Bayesian weighted rating
        C = valid["rating"].mean()
        m = max(valid["reviews"].quantile(0.60), 1)

        v = valid["reviews"]
        R = valid["rating"]

        valid["score"] = (
            (v / (v + m)) * R +
            (m / (v + m)) * C
        )

        self.table = valid.sort_values(
            ["score", "reviews"], ascending=False
        )

    def recommend(self, n=10):
        cols = [
            "product_name", "category", "price",
            "rating", "reviews", "score"
        ]
        return self.table[cols].head(n)


# --------------------------------------------------
# 4. CONTENT-BASED RECOMMENDER
# --------------------------------------------------

class ContentRecommender:
    """Find similar products using TF-IDF and cosine similarity."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2)
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["features"].fillna("")
        )

    def recommend(self, product_name, n=10):
        matches = self.df.index[
            self.df["product_name"].str.lower()
            == product_name.lower()
        ].tolist()

        if not matches:
            raise ValueError(
                f"Product not found: {product_name}. "
                "Check the product name in your CSV."
            )

        i = matches[0]

        similarity = cosine_similarity(
            self.matrix[i], self.matrix
        ).ravel()

        similarity[i] = -1
        order = np.argsort(-similarity)[:n]

        result = self.df.iloc[order][
            ["product_name", "category", "price", "rating"]
        ].copy()

        result["similarity"] = similarity[order]
        return result


# --------------------------------------------------
# 5. SIMPLE RECOMMENDATION EVALUATION
# --------------------------------------------------

def evaluate_recommender(df, k=10):
    """
    Compare the top-k products against a simple
    baseline using available rating and review data.
    """

    valid = df.dropna(subset=["rating"]).copy()

    if len(valid) < 2:
        print("Not enough rated products for evaluation.")
        return

    valid["reviews"] = valid["reviews"].fillna(0)

    # Define highly rated products with sufficient reviews
    min_reviews = max(valid["reviews"].quantile(0.50), 1)
    relevant = valid[
        (valid["rating"] >= 4.0) &
        (valid["reviews"] >= min_reviews)
    ]

    if relevant.empty:
        print("No relevant products found for evaluation.")
        return

    # Evaluate using the available dataset labels
    popular = PopularityRecommender(valid).recommend(k)
    baseline = valid.sort_values(
        ["rating", "reviews"], ascending=False
    ).head(k)

    relevant_names = set(relevant["product_name"])
    pop_names = set(popular["product_name"])
    base_names = set(baseline["product_name"])

    pop_precision = len(pop_names & relevant_names) / max(len(pop_names), 1)
    base_precision = len(base_names & relevant_names) / max(len(base_names), 1)

    print("\nEvaluation results")
    print(f"Popularity recommender precision@{k}: {pop_precision:.3f}")
    print(f"Rating baseline precision@{k}:       {base_precision:.3f}")
    print("Note: This is a simple relevance check, not a")
    print("user-personalized offline recommendation test.")


# --------------------------------------------------
# 6. MAIN PROGRAM
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flipkart Product Analysis and Recommendation System"
    )

    parser.add_argument(
        "--data",
        default=str(HERE / "flipkart_products.csv"),
        help="Path to the product CSV file"
    )

    parser.add_argument(
        "--product",
        default=None,
        help="Exact product name for similar-product recommendations"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of products to display"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate recommendation relevance"
    )

    args = parser.parse_args()

    print("[1/5] Loading and cleaning product data...")
    df = load_products(args.data)

    print(f"Products loaded: {len(df):,}")
    print(f"Categories: {df['category'].nunique():,}")
    print(f"Average rating: {df['rating'].mean():.2f}"
          if df["rating"].notna().any()
          else "Average rating: unavailable")

    save_clean(df)

    print("\n[2/5] Generating visualizations...")
    make_plots(df)
    print(f"Saved graphs to: {IMG}")

    print("\n[3/5] Popular product recommendations")
    popular = PopularityRecommender(df)
    print(popular.recommend(args.top).to_string(index=False))

    if args.product:
        print(f"\n[4/5] Products similar to: {args.product}")
        model = ContentRecommender(df)
        print(model.recommend(args.product, args.top).to_string(index=False))
    else:
        print("\n[4/5] Skipped similar-product recommendations.")
        print("Use --product \"Exact Product Name\" to enable.")

    if args.evaluate:
        print("\n[5/5] Evaluating recommendation relevance...")
        evaluate_recommender(df, args.top)
    else:
        print("\n[5/5] Evaluation skipped. Add --evaluate to run it.")


if __name__ == "__main__":
    main()
