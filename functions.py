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

# Initialize OpenAI client using environment variable
client = OpenAI()

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
        "retail_price": [round(random.uniform(300, 5000), 2) for _ in range(num_products)],
        "cost_price": [],
        "profit_margin": []
    }

    # Populate category and brand based on product_name
    for product_name in inventory_data["product_name"]:
        split_name = product_name.split(" ")
        inventory_data["brand"].append(split_name[0])
        inventory_data["category"].append(split_name[-1])

    # Calculate cost_price and profit_margin
    for i in range(num_products):
        retail_price = inventory_data["retail_price"][i]
        profit_margin = round(random.uniform(50, 100), 2) / 100
        cost_price = round(retail_price * (1 - profit_margin), 2)
        inventory_data["cost_price"].append(cost_price)
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
    inertia_values = []
    silhouette_scores = []
    
    # Calculate metrics for each k
    for k in range(k_range[0], k_range[1]):
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        
        # Calculate inertia
        inertia_values.append(kmeans.inertia_)
        
        # Calculate silhouette score (only for k >= 2)
        if k >= 2:
            labels = kmeans.labels_
            sil_score = silhouette_score(features, labels)
            silhouette_scores.append(sil_score)
    
    return inertia_values, silhouette_scores

def get_optimal_clusters_from_ai(inertia_values, silhouette_scores):
    """
    Uses OpenAI API to analyze clustering metrics and recommend optimal number of clusters.
    Args:
        inertia_values: List of inertia scores from Elbow method
        silhouette_scores: List of silhouette scores
    Returns:
        str: AI analysis and recommendation
    """
    # Prepare prompt for OpenAI
    prompt = f"""
    Based on the following clustering metrics:
    
    Elbow Method (Inertia Values): {inertia_values}
    Silhouette Scores: {silhouette_scores}
    
    Please analyze and recommend the optimal number of clusters (k).
    Consider both metrics and provide your recommendation in the following format:
    
    - Elbow Method Analysis: [your analysis]
    - Silhouette Score Analysis: [your analysis]
    - Recommended k: [number or range]
    - Explanation: [brief explanation of recommendation]
    """
    
    try:
        print("Sending request to OpenAI API...")
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data science expert specializing in cluster analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        print("Successfully received response from OpenAI")
        # Extract and return the analysis
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Neočekávaná chyba při komunikaci s OpenAI: {str(e)}"
        print(f"Unexpected Error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return error_msg

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
            model="gpt-3.5-turbo",
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
        error_msg = f"Neočekávaná chyba při komunikaci s OpenAI: {str(e)}"
        print(f"Unexpected Error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return error_msg

def perform_kmeans_clustering(data, n_clusters):
    """
    Performs K-means clustering on the input data.
    Args:
        data: DataFrame with normalized features
        n_clusters: Number of clusters to create
    Returns:
        numpy.ndarray: Cluster assignments
    """
    # Get numeric columns for clustering (excluding customer_id)
    numeric_columns = ['price_preference_range', 'discount_sensitivity', 'luxury_preference_score',
                      'total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                      'categories_bought', 'brands_bought']
    
    # Get data for clustering
    clustering_data = data[numeric_columns]
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(clustering_data)
    
    return kmeans.labels_

def create_clustered_datasets(normalized_data, cluster_labels, original_combined_data, original_preference_data, original_behavioral_data):
    """
    Creates two versions of clustered datasets: normalized and denormalized with original values.
    Args:
        normalized_data: DataFrame with normalized features
        cluster_labels: Cluster assignments from K-means
        original_combined_data: Original combined data before normalization
        original_preference_data: Original preference data with text values
        original_behavioral_data: Original behavioral data with emails
    Returns:
        tuple: (normalized dataset with clusters, denormalized dataset with clusters)
    """
    # Create normalized dataset with clusters
    clustered_data = normalized_data.copy()
    clustered_data['cluster'] = cluster_labels
    # Reorder columns to show cluster first
    cols = ['cluster'] + [col for col in clustered_data.columns if col != 'cluster']
    clustered_data = clustered_data[cols]
    
    # Create cluster assignments dataframe
    cluster_assignments = pd.DataFrame({
        'customer_id': normalized_data['customer_id'],
        'cluster': cluster_labels
    })
    
    # Create denormalized dataset with clusters
    # 1. Merge with original combined data
    final_clustered_data = pd.merge(
        original_combined_data,
        cluster_assignments,
        on='customer_id',
        how='inner'
    )
    
    # 2. Get original category and brand values
    final_clustered_data = pd.merge(
        final_clustered_data,
        original_preference_data[['customer_id', 'top_category', 'top_brand']],
        on='customer_id',
        how='left',
        suffixes=('_numeric', '')
    )
    
    # 3. Add email from original behavioral data
    final_clustered_data = pd.merge(
        final_clustered_data,
        original_behavioral_data[['customer_id', 'email']],
        on='customer_id',
        how='left'
    )
    
    # 4. Drop numeric columns and reorder
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

def evaluate_cluster_validation(distance_matrix, features, num_clusters):
    """
    Uses OpenAI to evaluate if the segmentation is usable.
    Args:
        distance_matrix: Matrix of distances between clusters
        features: DataFrame with features (not used)
        num_clusters: Number of clusters
    Returns:
        dict: Basic AI evaluation of segmentation usability
    """
    avg_distance = np.mean(distance_matrix[distance_matrix > 0])
    min_distance = np.min(distance_matrix[distance_matrix > 0])
    max_distance = np.max(distance_matrix)
    
    prompt = f"""
    You are an expert in customer segmentation using K-means clustering.
    Evaluate only the usability of segmentation based on distances between segments:

    Metrics:
    - Average distance: {avg_distance:.3f}
    - Minimum distance: {min_distance:.3f}
    - Maximum distance: {max_distance:.3f}
    - Number of segments: {num_clusters}
    
    Provide ONLY:
    1. Whether the segments are sufficiently different for use in marketing
    2. Brief justification based on numbers
    
    Format the response as:
    SEGMENTATION USABILITY:
    [EXCELLENT 😀 / GOOD 🙂 / AVERAGE 😐 / INSUFFICIENT 😟]

    JUSTIFICATION:
    [1-2 sentences explaining the rating based on numbers]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating segmentation quality. Provide concise and clear evaluations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            'analysis': response.choices[0].message.content,
            'metrics': {
                'avg_distance': avg_distance,
                'min_distance': min_distance,
                'max_distance': max_distance
            }
        }
        
    except Exception as e:
        print(f"Error in AI evaluation: {str(e)}")
        return {
            'analysis': "Failed to get AI analysis. Please try again later.",
            'metrics': None
        }

def generate_cluster_naming_prompt(data):
    """Generate a prompt for OpenAI to name clusters based on customer behavior."""
    cluster_summary = []
    clusters = sorted(data["cluster"].unique())
    
    for cluster in clusters:
        cluster_data = data[data["cluster"] == cluster]
        
        # Calculate average values for numeric columns
        avg_values = cluster_data[
            ["total_spent", "total_orders", "avg_order_value", "last_purchase_days_ago",
             "categories_bought", "brands_bought", "price_preference_range",
             "discount_sensitivity", "luxury_preference_score"]
        ].mean()
        
        # Get dominant values for categorical columns
        dominant_category = cluster_data["top_category"].mode()[0]
        dominant_brand = cluster_data["top_brand"].mode()[0]
        
        # Add cluster summary
        cluster_summary.append(
            f"Cluster {cluster}:\n"
            f"- Average Values:\n"
            f"  - Total Spent: ${avg_values['total_spent']:.2f}\n"
            f"  - Total Orders: {avg_values['total_orders']:.1f}\n"
            f"  - Average Order Value: ${avg_values['avg_order_value']:.2f}\n"
            f"  - Days Since Last Purchase: {avg_values['last_purchase_days_ago']:.1f}\n"
            f"  - Categories Bought: {avg_values['categories_bought']:.1f}\n"
            f"  - Brands Bought: {avg_values['brands_bought']:.1f}\n"
            f"  - Price Preference (1=Low, 2=Mid, 3=High): {avg_values['price_preference_range']:.1f}\n"
            f"  - Discount Sensitivity (0=Low, 1=High): {avg_values['discount_sensitivity']:.2f}\n"
            f"  - Luxury Preference (1-5): {avg_values['luxury_preference_score']:.1f}\n"
            f"- Most Common Category: {dominant_category}\n"
            f"- Most Common Brand: {dominant_brand}\n"
        )
    
    prompt = """You are an expert in customer segmentation and retail analytics.
Based on the following cluster data, create a short, descriptive name and explanation for each customer segment.

The data shows:
- Spending patterns (total spent, orders, average value)
- Shopping frequency (days since last purchase)
- Shopping variety (categories and brands bought)
- Price sensitivity (1=Low price, 2=Mid price, 3=High price)
- Discount seeking behavior (0=Rarely uses discounts, 1=Always seeks discounts)
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
    """Get cluster names and explanations from OpenAI."""
    try:
        # Generate prompt
        prompt = generate_cluster_naming_prompt(data)
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
        
    except Exception as e:
        print(f"Error getting AI cluster names: {str(e)}")
        return None

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
    