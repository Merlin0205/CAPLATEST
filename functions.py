import pandas as pd
import random
from faker import Faker
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os
import streamlit as st
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Initialize Faker
fake = Faker()

# Constants for the clothing and accessories e-shop
CATEGORIES = [
    "Tops", "Bottoms", "Dresses", "Outerwear", 
    "Shoes", "Accessories", "Sportswear"
]

BRANDS = [
    "Nike", "Adidas", "Puma", "Zara", "H&M", 
    "Gucci", "Prada", "Levi's", "Ralph Lauren", 
    "Under Armour", "Calvin Klein", "New Balance", 
    "Tommy Hilfiger", "Versace", "Burberry"
]

ADJECTIVES = ["Classic", "Modern", "Stylish", "Luxury", "Casual", "Comfortable", "Premium"]
COLORS = ["Red", "Blue", "Black", "White", "Green", "Beige", "Pink", "Grey"]

# Initialize OpenAI client
client = None

def get_openai_client():
    """Gets or creates OpenAI client with current API key."""
    global client
    try:
        # Try to get API key from Streamlit secrets
        api_key = st.secrets.get("OPENAI_API_KEY")
        
        if not api_key:
            st.error("OpenAI API key not found in Streamlit secrets. Please add it in your Streamlit Cloud settings.")
            return None
        
        # Initialize client if not exists or if API key changed
        if client is None or client.api_key != api_key:
            client = OpenAI(
                api_key=api_key
            )
        
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

def generate_behavioral_data(num_customers=2000):
    """
    Generates a behavioral dataset with customer purchase information, including deliberate errors.
    Values are set to create clearly separated groups for clustering.
    Returns:
        pandas.DataFrame: Dataset with customer behavioral data
    """
    # Define clearly separated customer groups
    customer_types = np.random.choice(['low', 'medium', 'high', 'premium'], size=num_customers, p=[0.3, 0.4, 0.2, 0.1])
    
    # Initialize empty lists for data
    behavioral_data = {
        "customer_id": [i for i in range(1, num_customers + 1)],
        "email": [fake.email() for _ in range(num_customers)],
        "total_spent": [],
        "total_orders": [],
        "avg_order_value": [],
        "last_purchase_days_ago": [],
        "categories_bought": [],
        "brands_bought": []
    }

    # Generate data based on customer type with more distinct differences
    for ctype in customer_types:
        if ctype == 'low':
            behavioral_data["total_spent"].append(np.random.uniform(10, 100))
            behavioral_data["total_orders"].append(np.random.randint(1, 3))
            behavioral_data["last_purchase_days_ago"].append(np.random.randint(180, 365))
            behavioral_data["categories_bought"].append(np.random.randint(1, 2))
            behavioral_data["brands_bought"].append(np.random.randint(1, 2))
        elif ctype == 'medium':
            behavioral_data["total_spent"].append(np.random.uniform(200, 500))
            behavioral_data["total_orders"].append(np.random.randint(5, 10))
            behavioral_data["last_purchase_days_ago"].append(np.random.randint(60, 180))
            behavioral_data["categories_bought"].append(np.random.randint(2, 4))
            behavioral_data["brands_bought"].append(np.random.randint(2, 4))
        elif ctype == 'high':
            behavioral_data["total_spent"].append(np.random.uniform(800, 1500))
            behavioral_data["total_orders"].append(np.random.randint(15, 25))
            behavioral_data["last_purchase_days_ago"].append(np.random.randint(14, 60))
            behavioral_data["categories_bought"].append(np.random.randint(4, 6))
            behavioral_data["brands_bought"].append(np.random.randint(4, 7))
        else:  # premium
            behavioral_data["total_spent"].append(np.random.uniform(2000, 5000))
            behavioral_data["total_orders"].append(np.random.randint(30, 50))
            behavioral_data["last_purchase_days_ago"].append(np.random.randint(1, 14))
            behavioral_data["categories_bought"].append(np.random.randint(6, 8))
            behavioral_data["brands_bought"].append(np.random.randint(7, 10))

    # Calculate average order value
    behavioral_data["avg_order_value"] = [
        round(spent / orders if orders > 0 else 0, 2)
        for spent, orders in zip(behavioral_data["total_spent"], behavioral_data["total_orders"])
    ]

    # Generate invalid emails (10 random records)
    for _ in range(10):
        idx = random.randint(0, num_customers - 1)
        behavioral_data["email"][idx] = "invalid_email.com" if random.random() < 0.5 else "user@@example.com"

    # Missing average order values (5 random records)
    for _ in range(5):
        idx = random.randint(0, num_customers - 1)
        behavioral_data["avg_order_value"][idx] = None

    # Negative values (2 records)
    for _ in range(2):
        idx = random.randint(0, num_customers - 1)
        behavioral_data["total_spent"][idx] = -random.uniform(100, 500)
        behavioral_data["avg_order_value"][idx] = -random.uniform(50, 200)

    # Empty records (3 random records)
    for _ in range(3):
        idx = random.randint(0, num_customers - 1)
        behavioral_data["total_spent"][idx] = None
        behavioral_data["total_orders"][idx] = 0
        behavioral_data["avg_order_value"][idx] = None
        behavioral_data["categories_bought"][idx] = None
        behavioral_data["brands_bought"][idx] = None
        behavioral_data["last_purchase_days_ago"][idx] = None

    return pd.DataFrame(behavioral_data)

def generate_preference_data(num_customers=2000):
    """
    Generates a preference dataset for customers with well-defined segments.
    Returns:
        pandas.DataFrame: Dataset with customer preferences
    """
    # Define customer types for better segmentation
    customer_types = np.random.choice(['budget', 'casual', 'fashion', 'luxury'], size=num_customers, p=[0.3, 0.4, 0.2, 0.1])
    
    preference_data = {
        "customer_id": [i for i in range(1, num_customers + 1)],
        "top_category": [],
        "top_brand": [],
        "price_preference_range": [],
        "discount_sensitivity": [],
        "luxury_preference_score": []
    }
    
    # Generate data based on customer type
    for ctype in customer_types:
        if ctype == 'budget':
            # Price-sensitive customers
            preference_data["top_category"].append(random.choice(["Tops", "Bottoms", "Sportswear"]))  # basic categories
            preference_data["top_brand"].append(random.choice(["H&M", "Zara", "Puma"]))  # affordable brands
            preference_data["price_preference_range"].append(1)  # low price preference
            preference_data["discount_sensitivity"].append(round(random.uniform(0.8, 1.0), 2))  # high discount sensitivity
            preference_data["luxury_preference_score"].append(random.randint(1, 2))  # low luxury preference
            
        elif ctype == 'casual':
            # Regular customers
            preference_data["top_category"].append(random.choice(["Tops", "Bottoms", "Dresses", "Shoes"]))
            preference_data["top_brand"].append(random.choice(["Nike", "Adidas", "Levi's", "Calvin Klein"]))
            preference_data["price_preference_range"].append(2)
            preference_data["discount_sensitivity"].append(round(random.uniform(0.4, 0.7), 2))
            preference_data["luxury_preference_score"].append(random.randint(2, 3))
            
        elif ctype == 'fashion':
            # Fashion enthusiasts
            preference_data["top_category"].append(random.choice(["Dresses", "Outerwear", "Accessories"]))
            preference_data["top_brand"].append(random.choice(["Ralph Lauren", "Tommy Hilfiger", "Versace"]))
            preference_data["price_preference_range"].append(2)
            preference_data["discount_sensitivity"].append(round(random.uniform(0.2, 0.5), 2))
            preference_data["luxury_preference_score"].append(random.randint(3, 4))
            
        else:  # luxury
            # Luxury customers
            preference_data["top_category"].append(random.choice(["Outerwear", "Accessories", "Shoes"]))
            preference_data["top_brand"].append(random.choice(["Gucci", "Prada", "Burberry"]))
            preference_data["price_preference_range"].append(3)
            preference_data["discount_sensitivity"].append(round(random.uniform(0.0, 0.3), 2))
            preference_data["luxury_preference_score"].append(5)

    return pd.DataFrame(preference_data)

def check_data_consistency(df):
    """
    Checks data consistency and returns a list of found problems.
    Args:
        df (pandas.DataFrame): Input dataset to check
    Returns:
        tuple: (list of problems, list of problem indices)
    """
    problems = []
    problem_indices = set()

    # Check emails
    invalid_emails = df[~df['email'].str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', regex=True)]
    if not invalid_emails.empty:
        problems.append(f"Found {len(invalid_emails)} invalid email addresses")
        problem_indices.update(invalid_emails.index)

    # Check null values
    null_records = df[df.isnull().any(axis=1)]
    if not null_records.empty:
        problems.append(f"Found {len(null_records)} records with missing values")
        problem_indices.update(null_records.index)

    # Check negative values
    negative_values = df[
        (df['total_spent'] < 0) |
        (df['total_orders'] < 0) |
        (df['avg_order_value'] < 0) |
        (df['last_purchase_days_ago'] < 0)
    ]
    if not negative_values.empty:
        problems.append(f"Found {len(negative_values)} records with negative values")
        problem_indices.update(negative_values.index)

    # Check average order value consistency
    mask = (df['total_orders'] > 0) & (df['total_spent'].notna()) & (df['avg_order_value'].notna())
    inconsistent_avg = df[mask].apply(
        lambda row: abs(row['total_spent'] / row['total_orders'] - row['avg_order_value']) > 0.01,
        axis=1
    )
    inconsistent_records = df[mask & inconsistent_avg]
    if not inconsistent_records.empty:
        problems.append(f"Found {len(inconsistent_records)} records with inconsistent average order value")
        problem_indices.update(inconsistent_records.index)

    return problems, list(problem_indices)

def generate_unique_product_name(unique_product_names):
    """Generates a unique product name."""
    while True:
        brand = random.choice(BRANDS)
        category = random.choice(CATEGORIES)
        adjective = random.choice(ADJECTIVES)
        color = random.choice(COLORS)
        product_name = f"{brand} {color} {adjective} {category}"
        if product_name not in unique_product_names:
            unique_product_names.add(product_name)
            return product_name

def generate_inventory_data(num_products=1000):
    """
    Generates an inventory dataset.
    Returns:
        pandas.DataFrame: Dataset with product inventory data
    """
    unique_product_names = set()
    
    inventory_data = {
        "product_id": [i for i in range(1, num_products + 1)],
        "product_name": [generate_unique_product_name(unique_product_names) for _ in range(num_products)],
        "category": [],
        "brand": [],
        "stock_quantity": [random.randint(0, 100) for _ in range(num_products)],
        "cost_price": [round(random.uniform(50, 1000), 2) for _ in range(num_products)],  # First generate cost_price
        "retail_price": [],  # Retail price will be calculated from cost_price and margin
        "profit_margin": []
    }

    # Populate category and brand based on product_name
    for product_name in inventory_data["product_name"]:
        split_name = product_name.split(" ")
        inventory_data["brand"].append(split_name[0])
        inventory_data["category"].append(split_name[-1])

    # Calculate retail_price and profit_margin
    for i in range(num_products):
        cost_price = inventory_data["cost_price"][i]
        profit_margin = round(random.uniform(0.5, 3.0), 2)  # Margin 50% to 300%
        retail_price = round(cost_price * (1 + profit_margin), 2)
        inventory_data["retail_price"].append(retail_price)
        inventory_data["profit_margin"].append(round(profit_margin * 100, 2))

    return pd.DataFrame(inventory_data)

def calculate_clustering_metrics(features, k_range=(1, 11)):
    """
    Calculates inertia (Elbow method) and silhouette scores for different k values.
    Args:
        features: Normalized feature matrix for clustering
        k_range: Tuple of (min_k, max_k) to test
    Returns:
        tuple: (inertia_values, silhouette_scores)
    """
    print(f"Starting clustering metrics calculation for k range {k_range}")
    print(f"Input features shape: {features.shape}")
    
    inertia_values = []
    silhouette_scores = []
    
    # Calculate metrics for each k
    for k in range(k_range[0], k_range[1]):
        start_time = time.time()
        print(f"\nProcessing k={k}...")
        
        # Fit KMeans
        print(f"Fitting KMeans for k={k}")
        kmeans_start = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        kmeans_time = time.time() - kmeans_start
        print(f"KMeans fitting completed in {kmeans_time:.2f} seconds")
        
        # Calculate inertia
        print(f"Calculating inertia for k={k}")
        inertia_start = time.time()
        inertia_values.append(kmeans.inertia_)
        inertia_time = time.time() - inertia_start
        print(f"Inertia calculation completed in {inertia_time:.2f} seconds")
        
        # Calculate silhouette score (only for k >= 2)
        if k >= 2:
            print(f"Calculating silhouette score for k={k}")
            silhouette_start = time.time()
            labels = kmeans.labels_
            sil_score = silhouette_score(features, labels)
            silhouette_scores.append(sil_score)
            silhouette_time = time.time() - silhouette_start
            print(f"Silhouette score calculation completed in {silhouette_time:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total processing time for k={k}: {total_time:.2f} seconds")
    
    print("\nClustering metrics calculation completed")
    print(f"Final inertia values: {inertia_values}")
    print(f"Final silhouette scores: {silhouette_scores}")
    
    return inertia_values, silhouette_scores

def get_optimal_clusters_from_ai(inertia_values, silhouette_scores):
    """
    Uses OpenAI to analyze elbow and silhouette metrics and recommend optimal number of clusters.
    Args:
        inertia_values: List of inertia values for k=1 to k=10
        silhouette_scores: List of silhouette scores for k=2 to k=10
    Returns:
        str: AI analysis and recommendation
    """
    client = get_openai_client()
    if not client:
        return "Failed to initialize OpenAI client. Please check your API key."
        
    prompt = f"""Analyze these clustering metrics and recommend the optimal number of clusters (k):

Inertia values (k=1 to k=10):
{inertia_values}

Silhouette scores (k=2 to k=10):
{silhouette_scores}

Requirements:
1. Consider both the elbow method (inertia) and silhouette scores
2. The minimum number of clusters must be 3
3. Explain the trade-off between number of clusters and model complexity
4. Recommend a specific k or a narrow range (max 2 numbers)

Format your response exactly like this:
ANALYSIS:
[2-3 sentences explaining what the metrics show]

RECOMMENDATION:
Recommended k: [number or range]

JUSTIFICATION:
[2-3 sentences explaining why this k is optimal]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing clustering metrics. Provide clear and concise recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return "Failed to get AI analysis. Please try again later."

def evaluate_clusters_with_ai(features, labels, metrics):
    """
    Uses OpenAI API to evaluate and describe the identified clusters.
    Args:
        features: Feature matrix used for clustering
        labels: Cluster assignments
        metrics: Dictionary of cluster metrics
    Returns:
        str: AI evaluation and description of clusters
    """
    client = get_openai_client()
    if not client:
        return "Failed to initialize OpenAI client. Please check your API key."
    
    # Prepare cluster statistics
    cluster_stats = {}
    for i in range(len(set(labels))):
        cluster_mask = labels == i
        cluster_features = features[cluster_mask]
        cluster_stats[f"Cluster {i}"] = {
            "size": sum(cluster_mask),
            "mean_values": cluster_features.mean(axis=0).tolist(),
            "std_values": cluster_features.std(axis=0).tolist()
        }
    
    # Prepare prompt for OpenAI
    prompt = f"""
    Based on the following cluster statistics:
    
    Cluster Statistics: {cluster_stats}
    Additional Metrics: {metrics}
    
    Please provide a detailed analysis of the clusters:
    1. Describe the characteristics of each cluster
    2. Identify any notable patterns or relationships
    3. Suggest potential customer segments these clusters might represent
    
    Format your response as:
    - Overview: [general analysis]
    - Cluster Descriptions: [cluster-by-cluster analysis]
    - Key Insights: [main takeaways]
    """
    
    try:
        print("Sending request to OpenAI API...")
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data science expert specializing in customer segmentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        print("Successfully received response from OpenAI")
        # Extract and return the analysis
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Unexpected error communicating with OpenAI: {str(e)}"
        print(f"Unexpected Error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return error_msg

def perform_kmeans_clustering(data, n_clusters):
    """
    Performs K-means clustering on the input data.
    Args:
        data: DataFrame with already normalized features
        n_clusters: Number of clusters to create
    Returns:
        numpy.ndarray: Cluster assignments
    """
    # Get all columns for clustering (including already normalized categorical ones)
    numeric_columns = ['price_preference_range', 'discount_sensitivity', 'luxury_preference_score',
                      'total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                      'categories_bought', 'brands_bought', 'top_category', 'top_brand']
    
    # Get data for clustering (already normalized)
    clustering_data = data[numeric_columns]
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(clustering_data)
    
    return kmeans.labels_

def create_clustered_datasets(normalized_data, cluster_labels, original_combined_data, original_preference_data, original_behavioral_data):
    """
    Creates two versions of clustered datasets: normalized and denormalized with original values.
    """
    # Create normalized dataset with clusters
    clustered_data = normalized_data.copy()
    
    # Normalize categorical columns if they haven't been normalized yet
    if clustered_data['top_category'].dtype == 'object' or clustered_data['top_brand'].dtype == 'object':
        # Normalize top_category
        le_category = LabelEncoder()
        clustered_data['top_category'] = le_category.fit_transform(clustered_data['top_category'])
        
        # Normalize top_brand
        le_brand = LabelEncoder()
        clustered_data['top_brand'] = le_brand.fit_transform(clustered_data['top_brand'])
        
        # Store the encoders in session state
        st.session_state.label_encoders = {
            'category': le_category,
            'brand': le_brand
        }
    
    # Normalize all numeric columns
    numeric_columns = ['price_preference_range', 'discount_sensitivity', 'luxury_preference_score',
                      'total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                      'categories_bought', 'brands_bought', 'top_category', 'top_brand']
    
    scaler = StandardScaler()
    clustered_data[numeric_columns] = scaler.fit_transform(clustered_data[numeric_columns])
    
    # Add cluster labels
    clustered_data['cluster'] = cluster_labels
    
    # Reorder columns to show cluster first
    cols = ['cluster'] + [col for col in clustered_data.columns if col != 'cluster']
    clustered_data = clustered_data[cols]
    
    # Create cluster assignments dataframe
    cluster_assignments = pd.DataFrame({
        'customer_id': normalized_data['customer_id'],
        'cluster': cluster_labels
    })
    
    # Create denormalized dataset with clusters (for display and analysis)
    final_clustered_data = pd.merge(
        original_combined_data,
        cluster_assignments,
        on='customer_id',
        how='inner'
    )
    
    # Add original category and brand values
    final_clustered_data = pd.merge(
        final_clustered_data,
        original_preference_data[['customer_id', 'top_category', 'top_brand']],
        on='customer_id',
        how='left',
        suffixes=('_numeric', '')
    )
    
    # Add email from original behavioral data
    final_clustered_data = pd.merge(
        final_clustered_data,
        original_behavioral_data[['customer_id', 'email']],
        on='customer_id',
        how='left'
    )
    
    # Drop numeric category/brand columns and reorder
    final_clustered_data = final_clustered_data.drop(
        columns=['top_category_numeric', 'top_brand_numeric']
    )
    cols = ['cluster'] + [col for col in final_clustered_data.columns if col != 'cluster']
    final_clustered_data = final_clustered_data[cols]
    
    return clustered_data, final_clustered_data

def calculate_validation_metrics(data, labels):
    """
    Calculates inter-cluster distances.
    Args:
        data: DataFrame with features used for clustering
        labels: Cluster assignments
    Returns:
        dict: Dictionary with distance matrix
    """
    # Get numeric columns for validation
    numeric_columns = ['price_preference_range', 'discount_sensitivity', 'luxury_preference_score',
                      'total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                      'categories_bought', 'brands_bought']
    
    features = data[numeric_columns]
    
    # Calculate inter-cluster distances
    centroids = []
    for cluster in range(len(set(labels))):
        centroids.append(features[labels == cluster].mean().values)
    distances = pdist(centroids)
    distance_matrix = squareform(distances)
    
    return {
        'distance_matrix': distance_matrix,
        'features': features,
        'feature_names': numeric_columns
    }

def plot_validation_metrics(distance_matrix, features, labels, feature_names):
    """
    Creates plot for Inter-Cluster Distance Map.
    Args:
        distance_matrix: Matrix of distances between clusters
        features: DataFrame with features (not used)
        labels: Cluster assignments (not used)
        feature_names: List of feature names (not used)
    Returns:
        matplotlib.figure.Figure: Distance matrix heatmap
    """
    fig = plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        distance_matrix,
        annot=True,
        cmap="RdYlBu_r",
        fmt=".2f",
        xticklabels=[f"Segment {i+1}" for i in range(len(distance_matrix))],
        yticklabels=[f"Segment {i+1}" for i in range(len(distance_matrix))],
        cbar_kws={'label': 'Distance between segments'}
    )
    
    plt.title("Customer Segment Distance Map", pad=20, size=14)
    plt.xlabel("Customer Segment", size=12)
    plt.ylabel("Customer Segment", size=12)
    
    plt.tight_layout()
    return fig

def evaluate_cluster_validation(distance_matrix, features, n_clusters):
    """
    Uses AI to evaluate clustering validation metrics and provide insights.
    """
    client = get_openai_client()
    if not client:
        return "Error: OpenAI client not available"

    prompt = f"""
    As a clustering expert, evaluate these customer segmentation results:

    METRICS:
    1. Number of segments: {n_clusters}
    2. Segment separation:
       - Average distance: {np.mean(distance_matrix):.3f}
       - Min distance: {np.min(distance_matrix[distance_matrix > 0]):.3f}
       - Max distance: {np.max(distance_matrix):.3f}
    3. Feature patterns: {features.mean(axis=0)}

    REQUIREMENTS:
    1. First line must be exactly:
    SEGMENTATION QUALITY: [one of: "INSUFFICIENT ⚠️", "USABLE ✓", "OPTIMAL ★"]

    2. On next line:
    - For INSUFFICIENT: Explain why and suggest specific changes
    - For USABLE: Explain why and suggest possible improvements
    - For OPTIMAL: Explain why, no suggestions needed

    EXAMPLE RESPONSES:

    Example 1:
    SEGMENTATION QUALITY: INSUFFICIENT ⚠️
    Segments show significant overlap (avg distance: 0.5), making customer targeting unreliable - try reducing to 3 segments.

    Example 2:
    SEGMENTATION QUALITY: USABLE ✓
    Segments show acceptable separation (avg distance: 1.2) - consider testing with 5 segments for potential improvement.

    Example 3:
    SEGMENTATION QUALITY: OPTIMAL ★
    Strong segment separation (avg distance: 2.1) with clear customer behavior patterns in each group.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a customer segmentation expert. Provide clear, actionable evaluations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during AI evaluation: {str(e)}"

def generate_cluster_naming_prompt(data):
    """Generate a prompt for OpenAI to name clusters based on customer behavior."""
    cluster_summary = []
    clusters = sorted(data["cluster"].unique())
    
    numeric_columns = ['total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                      'categories_bought', 'brands_bought', 'price_preference_range',
                      'discount_sensitivity', 'luxury_preference_score']
    
    for cluster in clusters:
        cluster_data = data[data["cluster"] == cluster]
        
        # Calculate average values for numeric columns
        avg_values = cluster_data[numeric_columns].mean()
        
        # Get dominant values for categorical columns
        dominant_category = cluster_data["top_category"].mode()[0]
        dominant_brand = cluster_data["top_brand"].mode()[0]
        
        # Add cluster summary
        cluster_summary.append(
            f"Cluster {cluster}:\n"
            f"- Average Values:\n"
            f"  - Total Spent: €{avg_values['total_spent']:.2f}\n"
            f"  - Total Orders: {avg_values['total_orders']:.1f}\n"
            f"  - Average Order Value: €{avg_values['avg_order_value']:.2f}\n"
            f"  - Days Since Last Purchase: {avg_values['last_purchase_days_ago']:.1f}\n"
            f"  - Categories Bought: {avg_values['categories_bought']:.1f}\n"
            f"  - Brands Bought: {avg_values['brands_bought']:.1f}\n"
            f"  - Price Preference Range (1=Budget, 2=Regular, 3=Premium): {avg_values['price_preference_range']:.1f}\n"
            f"  - Discount Sensitivity (0.0=Never uses discounts, 1.0=Always seeks discounts): {avg_values['discount_sensitivity']:.2f}\n"
            f"  - Luxury Preference (1=Basic products, 5=Luxury products): {avg_values['luxury_preference_score']:.1f}\n"
            f"- Most Common Category: {dominant_category}\n"
            f"- Most Common Brand: {dominant_brand}\n"
        )
    
    prompt = """You are an expert in customer segmentation and retail analytics.
Based on the following cluster data, create a short, descriptive name and explanation for each customer segment.

The data shows:
- Spending patterns (total spent, orders, average value)
- Shopping frequency (days since last purchase)
- Shopping variety (categories and brands bought)
- Price preference range (1=Budget, 2=Regular, 3=Premium)
- Discount sensitivity (0.0=Never uses discounts, 1.0=Always seeks discounts)
- Luxury orientation (1=Basic products, 5=Luxury products)
- Preferred category and brand

For each cluster, provide EXACTLY this format:
Cluster 0: [2-3 WORD NAME]
[2-3 sentences explaining key characteristics that define this segment]

Example:
Cluster 0: Premium Fashion Enthusiasts
These customers spend heavily on luxury brands, rarely seek discounts, and frequently purchase from multiple categories. They show strong preference for high-end fashion and make regular purchases.

Cluster Data:
"""
    
    return prompt + "\n" + "\n".join(cluster_summary)

def get_cluster_names_from_ai(data):
    """
    Uses OpenAI to generate descriptive names for clusters.
    """
    client = get_openai_client()
    if not client:
        return "Failed to initialize OpenAI client. Please check your API key."
    
    # Prepare cluster statistics
    cluster_stats = {}
    numeric_columns = ['total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                      'categories_bought', 'brands_bought', 'price_preference_range',
                      'discount_sensitivity', 'luxury_preference_score']
    categorical_columns = ['top_category', 'top_brand']
    
    for i in range(len(set(data["cluster"]))):
        cluster_mask = data["cluster"] == i
        cluster_features = data[cluster_mask]
        
        # Calculate statistics only for numeric columns
        mean_values = cluster_features[numeric_columns].mean().to_dict()
        # Get mode for categorical columns
        categorical_values = {col: cluster_features[col].mode().iloc[0] for col in categorical_columns}
        
        cluster_stats[f"Cluster {i}"] = {
            "size": sum(cluster_mask),
            "mean_values": mean_values,
            "categorical_values": categorical_values
        }
    
    # Generate prompt
    prompt = generate_cluster_naming_prompt(data)
    
    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a customer segmentation expert. Provide concise and descriptive cluster names with clear explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    
    # Parse response
    ai_response = response.choices[0].message.content
    
    # Extract cluster names and explanations
    cluster_info = {}
    current_cluster = None
    current_text = []
    
    for line in ai_response.split('\n'):
        if line.startswith('Cluster '):
            # Save previous cluster info if exists
            if current_cluster is not None and current_text:
                cluster_info[current_cluster] = '\n'.join(current_text)
            
            # Start new cluster
            current_cluster = int(line.split(':')[0].split()[1])
            name = line.split(':', 1)[1].strip()
            current_text = [name]
        elif line.strip() and current_cluster is not None:
            current_text.append(line.strip())
    
    # Save last cluster
    if current_cluster is not None and current_text:
        cluster_info[current_cluster] = '\n'.join(current_text)
    
    return cluster_info

def create_named_clustered_dataset(clustered_data, cluster_names):
    """Create a new dataset with AI-suggested cluster names."""
    # Create a copy of the clustered data
    named_clusters = clustered_data.copy()
    
    # Extract names and explanations
    names = {}
    explanations = {}
    
    for cluster, info in cluster_names.items():
        lines = info.split('\n')
        names[cluster] = lines[0]  # First line is the name
        explanations[cluster] = '\n'.join(lines[1:])  # Rest is explanation
    
    # Add new columns
    named_clusters['cluster_name'] = named_clusters['cluster'].map(names)
    named_clusters['cluster_explanation'] = named_clusters['cluster'].map(explanations)
    
    return named_clusters

def select_best_segment(cluster_data, product_info, discount_percent):
    """Selects the best customer segment for the given product and discount."""
    client = get_openai_client()
    if not client:
        return {
            "cluster_name": "Error",
            "match_score": 0,
            "cluster_explanation": "OpenAI API key not available. Please check your configuration.",
            "stats": {
                "avg_spent": 0,
                "avg_order": 0,
                "top_brand": "N/A",
                "top_category": "N/A"
            }
        }

    # Create segment summaries
    segments = {}
    for cluster_name in cluster_data['cluster_name'].unique():
        segment_data = cluster_data[cluster_data['cluster_name'] == cluster_name]
        
        # Calculate segment statistics
        stats = {
            'avg_spent': segment_data['total_spent'].mean(),
            'avg_order': segment_data['avg_order_value'].mean(),
            'top_brand': segment_data['top_brand'].mode().iloc[0],
            'top_category': segment_data['top_category'].mode().iloc[0],
            'avg_discount_sensitivity': segment_data['discount_sensitivity'].mean(),
            'size': len(segment_data)
        }
        
        segments[cluster_name] = {
            'stats': stats,
            'explanation': segment_data['cluster_explanation'].iloc[0]
        }

    # Prepare prompt for AI
    prompt = f"""Analyze these customer segments for promoting this product:

Product:
- Name: {product_info['product_name']}
- Category: {product_info['category']}
- Brand: {product_info['brand']}
- Price: €{product_info['retail_price']:.2f}
- Discount: {discount_percent}%
- Final Price: €{product_info['retail_price'] * (1 - discount_percent/100):.2f}

Customer Segments:
"""
    
    for name, data in segments.items():
        prompt += f"""
{name}:
- Average Spent: €{data['stats']['avg_spent']:.2f}
- Average Order: €{data['stats']['avg_order']:.2f}
- Top Category: {data['stats']['top_category']}
- Top Brand: {data['stats']['top_brand']}
- Customers: {data['stats']['size']}
- Profile: {data['explanation']}
"""

    prompt += """
Select the best segment for this product promotion.
Format your response exactly as:
SEGMENT: [segment name]
SCORE: [0-100]
REASON: [1-2 sentences why this segment is best]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a marketing expert selecting customer segments for targeted promotions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Parse response
        lines = response.choices[0].message.content.split('\n')
        selected_name = lines[0].replace('SEGMENT:', '').strip()
        score = int(lines[1].replace('SCORE:', '').strip())
        reason = ' '.join(lines[2:]).replace('REASON:', '').strip()

        # Get segment stats
        selected_segment = segments[selected_name]

        return {
            "cluster_name": selected_name,
            "match_score": score,
            "cluster_explanation": reason,
            "stats": selected_segment['stats']
        }

    except Exception as e:
        st.error(f"Error selecting segment: {str(e)}")
        return {
            "cluster_name": "Error",
            "match_score": 0,
            "cluster_explanation": f"Error selecting segment: {str(e)}",
            "stats": {
                "avg_spent": 0,
                "avg_order": 0,
                "top_brand": "N/A",
                "top_category": "N/A"
            }
        }

def select_best_customers(cluster_data, cluster_name, product_info, discount_percent):
    """
    Selects the best customers from given segment.
    Args:
        cluster_data: DataFrame with cluster data
        cluster_name: Name of selected cluster
        product_info: Dictionary with product information
        discount_percent: Discount percentage
    Returns:
        list: List of best customers with their scores
    """
    # Get customers from selected segment using cluster_name
    cluster_customers = cluster_data[cluster_data['cluster_name'] == cluster_name].copy()
    
    # Calculate product price tier (1=Budget, 2=Regular, 3=Premium)
    avg_price = cluster_customers['avg_order_value'].mean()
    if product_info['retail_price'] <= avg_price * 0.7:
        product_price_tier = 1  # Budget
    elif product_info['retail_price'] <= avg_price * 1.3:
        product_price_tier = 2  # Regular
    else:
        product_price_tier = 3  # Premium
    
    # Calculate customer scores
    customer_scores = []
    for _, customer in cluster_customers.iterrows():
        score = 0
        reasons = []
        
        # 1. Brand Affinity (20 points)
        brand_score = 0
        if customer['top_brand'] == product_info['brand']:
            brand_score = 20
            reasons.append("✓ Preferred Brand (+20 points)")
        elif product_info['brand'] in str(customer['brands_bought']):
            brand_score = 10
            reasons.append("✓ Previously Purchased Brand (+10 points)")
        score += brand_score
        
        # 2. Category Match (20 points)
        category_score = 0
        if customer['top_category'] == product_info['category']:
            category_score = 20
            reasons.append("✓ Preferred Category (+20 points)")
        elif product_info['category'] in str(customer['categories_bought']):
            category_score = 10
            reasons.append("✓ Previously Purchased Category (+10 points)")
        score += category_score
        
        # 3. Price Preference Match (20 points)
        price_score = 0
        customer_price_pref = customer['price_preference_range']  # 1=Budget, 2=Regular, 3=Premium
        if customer_price_pref == product_price_tier:
            price_score = 20
            reasons.append("✓ Perfect Price Range Match (+20 points)")
        elif abs(customer_price_pref - product_price_tier) == 1:
            price_score = 10
            reasons.append("✓ Close Price Range Match (+10 points)")
        score += price_score
        
        # 4. Luxury Preference Match (20 points)
        luxury_score = 0
        if product_price_tier == 3:  # Premium product
            if customer['luxury_preference_score'] >= 4:
                luxury_score = 20
                reasons.append("✓ High Luxury Preference Match (+20 points)")
            elif customer['luxury_preference_score'] >= 3:
                luxury_score = 10
                reasons.append("✓ Medium Luxury Preference Match (+10 points)")
        elif product_price_tier == 2:  # Regular product
            if 2 <= customer['luxury_preference_score'] <= 4:
                luxury_score = 20
                reasons.append("✓ Perfect Regular Product Match (+20 points)")
            else:
                luxury_score = 10
                reasons.append("✓ Acceptable Regular Product Match (+10 points)")
        else:  # Budget product
            if customer['luxury_preference_score'] <= 2:
                luxury_score = 20
                reasons.append("✓ Budget Product Preference Match (+20 points)")
            elif customer['luxury_preference_score'] <= 3:
                luxury_score = 10
                reasons.append("✓ Acceptable Budget Product Match (+10 points)")
        score += luxury_score
        
        # 5. Discount Sensitivity Match (20 points)
        discount_score = 0
        sensitivity = customer['discount_sensitivity']
        
        # High discount (>20%) - prefer discount-sensitive customers
        if discount_percent > 20:
            if sensitivity >= 0.7:  # High sensitivity (0.7-1.0)
                discount_score = 20
                reasons.append("✓ High Discount Sensitivity for Large Discount (+20 points)")
            elif sensitivity >= 0.5:  # Medium sensitivity (0.5-0.7)
                discount_score = 10
                reasons.append("✓ Medium Discount Sensitivity for Large Discount (+10 points)")
        # Lower discount (<=20%) - prefer less sensitive customers
        else:
            if sensitivity <= 0.3:  # Low sensitivity (0.0-0.3)
                discount_score = 20
                reasons.append("✓ Low Discount Sensitivity for Small Discount (+20 points)")
            elif sensitivity <= 0.5:  # Medium sensitivity (0.3-0.5)
                discount_score = 10
                reasons.append("✓ Medium Discount Sensitivity for Small Discount (+10 points)")
        score += discount_score
        
        customer_scores.append({
            'customer_id': customer['customer_id'],
            'score': score,
            'reasons': reasons,
            'data': customer,
            'scores_breakdown': {
                'brand': brand_score,
                'category': category_score,
                'price': price_score,
                'luxury': luxury_score,
                'discount': discount_score
            }
        })
    
    # Select top 5 matching customers
    return sorted(customer_scores, key=lambda x: x['score'], reverse=True)[:5]

def generate_promotional_email(product_info, customer_info, segment_info):
    """
    Generate a promotional email using AI.
    
    Args:
        product_info: Dictionary with product details
        customer_info: Dictionary with customer details
        segment_info: Dictionary with segment details
        
    Returns:
        dict: Dictionary containing email subject and body
    """
    client = get_openai_client()
    if not client:
        return {
            "subject": "Error: OpenAI Client Not Available",
            "body": "Failed to initialize OpenAI client. Please check your API key."
        }
    
    prompt = f"""Create a stunning, creative promotional email that focuses on the product and offer. Make it visually appealing and persuasive.

Use these details to create the email:

Product Details:
- Name: {product_info['product_name']}
- Brand: {product_info['brand']}
- Category: {product_info['category']}
- Original Price: €{product_info['original_price']:.2f}
- Discount: {product_info['discount_percent']}%
- Final Price: €{product_info['discounted_price']:.2f}

Target Audience Preferences:
- Preferred Brand: {customer_info['profile']['top_brand']}
- Preferred Category: {customer_info['profile']['top_category']}
- Average Purchase Value: €{customer_info['profile']['avg_order_value']:.2f}
- Last Purchase: {customer_info['profile']['last_purchase_days_ago']} days ago
- Discount Sensitivity: {customer_info['profile']['discount_sensitivity']}

Requirements:
1. Create a modern, luxury-style promotional email
2. Use HTML formatting for a beautiful design
3. Include a strong call-to-action
4. Emphasize the limited-time offer. Higlight it
5. Focus on value proposition and savings
6. Keep the design clean and professional
7. Make the discount prominent and higlight it
8. Include a clear call-to-action button
8. No pictures. ALways white backgroung.

Format your response exactly like this:
SUBJECT: Your subject line here
BODY: Your HTML formatted email body here"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a world-class email marketing designer and copywriter. Create stunning, highly personalized promotional emails that convert. Use modern design principles and persuasive writing techniques."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )
    
    content = response.choices[0].message.content
    subject = ""
    body = ""
    
    for line in content.split('\n'):
        if line.startswith('SUBJECT:'):
            subject = line.replace('SUBJECT:', '').strip()
        elif line.startswith('BODY:'):
            body = content[content.index('BODY:') + 5:].strip()
    
    # Clean HTML code
    body = body.strip('`').strip()
    if body.startswith('```html'):
        body = body[7:].strip()
    if body.endswith('```'):
        body = body[:-3].strip()
    
    return {
        'subject': subject,
        'body': body
    }

def send_promotional_email(config, subject, body, receiver_email):
    """
    Send promotional email using SMTP.
    
    Args:
        config: Dictionary with SMTP configuration
        subject: Email subject
        body: HTML email body
        receiver_email: Recipient's email address
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Get SMTP password from environment variable
        smtp_password = os.getenv('SMTP_PASSWORD')
        if not smtp_password:
            return False, "❌ SMTP password not found in environment variables"

        # Create the email message
        msg = MIMEMultipart()
        msg["From"] = config['sender_email']
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()  # Enable encrypted connection
            server.login(config['sender_email'], smtp_password)
            server.sendmail(config['sender_email'], receiver_email, msg.as_string())

        return True, "✅ Email sent successfully!"
    except Exception as e:
        return False, f"❌ Error sending email: {str(e)}"
    
