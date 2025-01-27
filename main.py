import streamlit as st
import matplotlib.pyplot as plt
from secure_config import verify_password, get_api_key
from openai import OpenAI
from streamlit_quill import st_quill

# Page settings must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Customer Data Analysis")

# Initialize OpenAI client
client = None
if "openai_api_key" in st.session_state:
    client = OpenAI(api_key=st.session_state.openai_api_key)

# Initialize session state
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

# Login check and API key decryption
def check_password():
    """Returns `True` if the password is correct and sets the API key."""
    def password_entered():
        """Verifies password and retrieves API key."""
        try:
            if "password" in st.session_state and verify_password(st.session_state["password"]):
                api_key = get_api_key(st.session_state["password"])
                if api_key:
                    st.session_state["password_correct"] = True
                    st.session_state["openai_api_key"] = api_key
                    del st.session_state["password"]  # Delete password from session state
                    st.success("✅ Login successful! Application is starting...")
                else:
                    st.session_state["password_correct"] = False
                    st.error("❌ Incorrect password")
            else:
                st.session_state["password_correct"] = False
                st.error("❌ Incorrect password")
        except Exception as e:
            st.session_state["password_correct"] = False
            st.error(f"❌ Error during password verification: {str(e)}")

    if not st.session_state["password_correct"]:
        # First display of login screen
        st.markdown("""
            <style>
                .stTextInput > div > div > input {
                    width: 250px;
                }
            </style>""", unsafe_allow_html=True)
        
        st.text_input(
            "Enter password to access the application", 
            type="password",
            key="password",
            on_change=password_entered
        )
        st.info("Please enter the password to continue")
        return False
    
    return st.session_state["password_correct"]

# Kontrola hesla
if not check_password():
    st.stop()  # Zastaví aplikaci, pokud není zadáno správné heslo

import pandas as pd
from functions import (
    generate_behavioral_data,
    generate_preference_data,
    generate_inventory_data,
    check_data_consistency,
    calculate_clustering_metrics,
    get_optimal_clusters_from_ai,
    CATEGORIES,
    BRANDS,
    perform_kmeans_clustering,
    create_clustered_datasets,
    calculate_validation_metrics,
    plot_validation_metrics,
    evaluate_cluster_validation,
    get_cluster_names_from_ai,
    create_named_clustered_dataset,
    select_best_segment,
    select_best_customers
)
import toml
from components.section import create_section
from components.template_loader import load_template

# Load CSS
with open("styles/main.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Handle navigation from buttons
if 'go_to_clustering' in st.session_state and st.session_state['go_to_clustering']:
    st.session_state['go_to_clustering'] = False
    st.session_state.menu_selection = "Clustering"

if 'go_to_data' in st.session_state and st.session_state['go_to_data']:
    st.session_state['go_to_data'] = False
    st.session_state.menu_selection = "Data"

# Handle navigation to Inventory & Customer Selection
if 'go_to_inventory' in st.session_state and st.session_state['go_to_inventory']:
    st.session_state['go_to_inventory'] = False
    st.session_state.menu_selection = "Inventory & Customer Selection"

# Handle navigation to Email Design
if 'go_to_email' in st.session_state and st.session_state['go_to_email']:
    st.session_state['go_to_email'] = False
    st.session_state.menu_selection = "Email Design"

# Load config file
try:
    with open('.streamlit/config.toml', 'r') as f:
        config = toml.load(f)
except Exception as e:
    pass  # Silent fail, no debug output needed

# Sidebar menu
if 'menu_selection' not in st.session_state:
    st.session_state.menu_selection = "Instructions"

menu_selection = st.sidebar.radio(
    "Navigation",
    ["Instructions", "Data", "Clustering", "Inventory & Customer Selection", "Email Design"],
    key="menu_selection"
)

if menu_selection == "Instructions":
    st.title("📚 User Guide")
    st.write("Welcome to the Customer Data Analysis Tool!")
    
    # Application Goal section
    application_goal_content = load_template("application-goal")
    st.markdown(create_section(
        "APPLICATION GOAL",
        application_goal_content,
        "🎯"
    ), unsafe_allow_html=True)
    
    # Workflow section
    workflow_content = load_template("workflow")
    st.markdown(create_section(
        "WORKFLOW",
        workflow_content,
        "📋"
    ), unsafe_allow_html=True)
    
    # Getting Started section
    getting_started_content = load_template("getting-started")
    st.markdown(create_section(
        "GETTING STARTED",
        getting_started_content,
        "🚀"
    ), unsafe_allow_html=True)
    
    # Navigation button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Start with Data Section ➡️", type="primary", use_container_width=True):
            st.session_state['go_to_data'] = True
            st.rerun()

elif menu_selection == "Data":
    st.title("📊 Customer Data Analysis")

    # Load Data section
    with st.expander("📥 Load Data" + (" ✅ COMPLETED" if 'data_loaded' in st.session_state else ""), expanded=True):
        st.markdown('<p class="big-header">Load Data</p>', unsafe_allow_html=True)
        
        load_col = st.container()
        with load_col:
            if 'data_loaded' not in st.session_state:
                st.markdown('<div class="load-button">', unsafe_allow_html=True)
                if st.button("Load Datasets"):
                    # Generate and store original datasets
                    st.session_state.original_behavioral_data = generate_behavioral_data(num_customers=2000)
                    st.session_state.original_preference_data = generate_preference_data(num_customers=2000)
                    st.session_state.original_inventory_data = generate_inventory_data(num_products=1000)
                    
                    st.session_state.data_loaded = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="loaded-button">', unsafe_allow_html=True)
                st.button("✅ Load Datasets")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Show success messages if data is loaded
        if 'data_loaded' in st.session_state:
            st.success("✅ Behavioral dataset loaded successfully")
            st.success("✅ Preference dataset loaded successfully")
            st.success("✅ Inventory dataset loaded successfully")
        
            show_data = st.toggle("Show Original Data", value=False)
            
            if show_data:
                st.subheader("Original Datasets:")
                
                # Display Original Behavioral Data
                st.subheader(f"Original Behavioral Data -- Number of records: {len(st.session_state.original_behavioral_data)}")
                st.dataframe(
                    st.session_state.original_behavioral_data,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Display Original Preference Data
                st.subheader(f"Original Preference Data -- Number of records: {len(st.session_state.original_preference_data)}")
                st.dataframe(
                    st.session_state.original_preference_data,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Display Original Inventory Data
                st.subheader(f"Original Inventory Data -- Number of records: {len(st.session_state.original_inventory_data)}")
                st.dataframe(
                    st.session_state.original_inventory_data,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
    
    if 'data_loaded' in st.session_state:
        # Data Consistency Check section
        with st.expander("🔍 Data Consistency Check ✅ COMPLETED", expanded=False):
            st.markdown('<p class="big-header">Data Consistency Check</p>', unsafe_allow_html=True)
            
            problems, problem_rows = check_data_consistency(st.session_state.original_behavioral_data)
            
            # Remove problematic rows and save clean data
            if problem_rows:
                st.session_state.clean_behavioral_data = st.session_state.original_behavioral_data.drop(problem_rows).reset_index(drop=True)
            else:
                st.session_state.clean_behavioral_data = st.session_state.original_behavioral_data.copy()
            
            # Display problems and their resolution
            for problem in problems:
                st.warning(f"• {problem} --- ✅ resolved")
            
            if not problems:
                st.success("✅ No issues found in the dataset")
            
            st.success(f"📊 Records after cleaning: {len(st.session_state.clean_behavioral_data)}")

        # Data Privacy section
        with st.expander("🔒 Data Privacy & Anonymization ✅ COMPLETED", expanded=False):
            st.markdown('<p class="big-header">Data Privacy & Anonymization</p>', unsafe_allow_html=True)
            
            # Create anonymized dataset by removing email column
            st.session_state.anonymized_behavioral_data = st.session_state.clean_behavioral_data.drop(columns=['email'])
            st.success("✅ Email addresses removed from dataset for AI analysis")
            st.success("ℹ️ Anonymized dataset ready for processing")

        # String to Numeric Conversion section
        with st.expander("🔢 String to Numeric Conversion ✅ COMPLETED", expanded=False):
            st.markdown('<p class="big-header">String to Numeric Conversion</p>', unsafe_allow_html=True)
            
            # Create mapping tables for 'top_category' and 'top_brand'
            category_mapping = {category: idx for idx, category in enumerate(CATEGORIES)}
            brand_mapping = {brand: idx for idx, brand in enumerate(BRANDS)}
            
            # Create a copy of preference data for conversion
            st.session_state.numeric_preference_data = st.session_state.original_preference_data.copy()
            
            # Convert categorical columns to numeric
            st.session_state.numeric_preference_data["top_category"] = st.session_state.original_preference_data["top_category"].map(category_mapping)
            st.session_state.numeric_preference_data["top_brand"] = st.session_state.original_preference_data["top_brand"].map(brand_mapping)
            
            # Save mappings for reference
            st.session_state.category_mapping = pd.DataFrame(list(category_mapping.items()), columns=["category_name", "category_id"])
            st.session_state.brand_mapping = pd.DataFrame(list(brand_mapping.items()), columns=["brand_name", "brand_id"])
            
            st.success("✅ Categorical values converted to numeric format")
            
            show_details = st.toggle("Show Conversion Details", value=False)
            if show_details:
                st.subheader("Category Mapping")
                st.dataframe(
                    st.session_state.category_mapping,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.subheader("Brand Mapping")
                st.dataframe(
                    st.session_state.brand_mapping,
                    use_container_width=True,
                    hide_index=True
                )

        # Final Datasets section
        with st.expander("📋 Final Datasets Overview ✅ COMPLETED", expanded=False):
            st.markdown('<p class="big-header">Final Datasets</p>', unsafe_allow_html=True)
            
            # Merge datasets
            st.session_state.combined_raw_data = pd.merge(
                st.session_state.numeric_preference_data,
                st.session_state.anonymized_behavioral_data,
                on="customer_id",
                how="inner"
            )
            
            # Normalize data for K-means
            from sklearn.preprocessing import StandardScaler
            
            # Select numeric columns for normalization
            numeric_columns = ['price_preference_range', 'discount_sensitivity', 'luxury_preference_score',
                             'total_spent', 'total_orders', 'avg_order_value', 'last_purchase_days_ago',
                             'categories_bought', 'brands_bought']
            
            # Create scaler
            scaler = StandardScaler()
            
            # Create copy for normalization
            st.session_state.normalized_kmeans_data = st.session_state.combined_raw_data.copy()
            
            # Normalize numeric columns
            st.session_state.normalized_kmeans_data[numeric_columns] = scaler.fit_transform(
                st.session_state.combined_raw_data[numeric_columns]
            )
            
            st.success("✅ Datasets merged and normalized for K-means analysis")
            st.success(f"✅ Final dataset contains {len(st.session_state.normalized_kmeans_data)} records")
            
            show_final_data = st.toggle("Show Final Dataset", value=False)
            if show_final_data:
                st.subheader(f"Final Dataset for K-means Clustering -- Number of records: {len(st.session_state.normalized_kmeans_data)}")
                st.dataframe(
                    st.session_state.normalized_kmeans_data,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )

        # Final success message
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 2rem; margin: 2rem 0; background-color: #1a1a1a; border-radius: 10px; border: 1px solid #333;'>
                <h2 style='color: #00cc00; margin-bottom: 1rem;'>✅ All Data Successfully Processed</h2>
                <p style='color: #ffffff; font-size: 1.2rem; margin-bottom: 1.5rem;'>
                    All datasets have been loaded, cleaned, and prepared for analysis. You can now proceed to customer segmentation.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the button using columns
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Continue to Clustering Analysis ➡️", type="primary", use_container_width=True):
                # Instead of directly modifying menu_selection, set a flag
                st.session_state['go_to_clustering'] = True
                st.rerun()

elif menu_selection == "Clustering":
    st.title("🎯 Customer Clustering")
    
    # Step 1: Determine Optimal Number of Clusters
    with st.expander("1️⃣ Determine the optimal number of clusters", expanded=True):
        st.markdown('<p class="big-header">Determine Optimal Number of Clusters</p>', unsafe_allow_html=True)
        
        # Check if normalized data exists
        if "normalized_kmeans_data" not in st.session_state:
            st.warning("⚠️ Please prepare the data in the DATA section first.")
            st.info("Go to the DATA section and complete data preparation for clustering.")
            
            # Navigation button
            if st.button("Go to DATA section ➡️", type="primary"):
                st.session_state['go_to_data'] = True
                st.rerun()
        else:
            # Toggle for showing normalized data
            show_data = st.toggle("Show normalized data for clustering")
            if show_data:
                st.dataframe(st.session_state.normalized_kmeans_data)
            
            features = st.session_state.normalized_kmeans_data.values
            
            # Button to trigger AI analysis
            if "clustering_analysis" not in st.session_state:
                if st.button("Run AI Analysis for Optimal Number of Clusters", type="primary"):
                    # Calculate metrics if not already calculated
                    if "inertia_values" not in st.session_state or "silhouette_scores" not in st.session_state:
                        with st.spinner("Calculating clustering metrics..."):
                            st.session_state.inertia_values, st.session_state.silhouette_scores = calculate_clustering_metrics(features)
                    
                    # Get AI recommendation
                    with st.spinner("Getting AI analysis..."):
                        st.session_state.clustering_analysis = get_optimal_clusters_from_ai(
                            st.session_state.inertia_values,
                            st.session_state.silhouette_scores
                        )
                        
                        # Extract recommended k range and set initial slider value
                        analysis = st.session_state.clustering_analysis
                        import re
                        k_range = re.findall(r'Recommended k: (\d+)(?:-(\d+))?', analysis)
                        if k_range:
                            start = int(k_range[0][0])
                            end = int(k_range[0][1]) if k_range[0][1] else start
                            st.session_state.optimal_k = (start + end) // 2
                        else:
                            st.session_state.optimal_k = 5  # Default value if no range found
                        
                        # Perform initial clustering with recommended k
                        cluster_labels = perform_kmeans_clustering(
                            st.session_state.normalized_kmeans_data,
                            st.session_state.optimal_k
                        )
                        
                        # Create clustered datasets
                        st.session_state.clustered_data, st.session_state.final_clustered_data = create_clustered_datasets(
                            st.session_state.normalized_kmeans_data,
                            cluster_labels,
                            st.session_state.combined_raw_data,
                            st.session_state.original_preference_data,
                            st.session_state.original_behavioral_data
                        )
                        
                        # Calculate initial validation metrics
                        validation_metrics = calculate_validation_metrics(
                            st.session_state.clustered_data,
                            st.session_state.clustered_data['cluster']
                        )
                        st.session_state.validation_metrics = validation_metrics
                        
                        # Get initial AI evaluation
                        ai_evaluation = evaluate_cluster_validation(
                            validation_metrics['distance_matrix'],
                            validation_metrics['features'],
                            len(set(st.session_state.clustered_data['cluster']))
                        )
                        st.session_state.ai_validation = ai_evaluation
                        
                        st.rerun()
            
            # Display plots and analysis if available
            if "clustering_analysis" in st.session_state:
                col1, col2 = st.columns(2)
                
                plt.rcParams['figure.figsize'] = [6, 3]
                
                # Elbow Method Plot
                with col1:
                    st.write("**Elbow Method**")
                    fig_elbow = plt.figure()
                    plt.plot(range(1, 11), st.session_state.inertia_values, 'bx-')
                    plt.xlabel('Number of Clusters (k)')
                    plt.ylabel('Inertia')
                    plt.title('Elbow Method')
                    plt.axvline(x=st.session_state.optimal_k, color='r', linestyle='--', alpha=0.5)
                    plt.text(st.session_state.optimal_k, plt.ylim()[1], f'k={st.session_state.optimal_k}', 
                            rotation=0, ha='right', va='bottom')
                    st.pyplot(fig_elbow)
                    plt.close()
                
                # Silhouette Score Plot
                with col2:
                    st.write("**Silhouette Score**")
                    fig_silhouette = plt.figure()
                    plt.plot(range(2, 11), st.session_state.silhouette_scores, 'rx-')
                    plt.xlabel('Number of Clusters (k)')
                    plt.ylabel('Silhouette Score')
                    plt.title('Silhouette Score')
                    plt.axvline(x=st.session_state.optimal_k, color='r', linestyle='--', alpha=0.5)
                    plt.text(st.session_state.optimal_k, plt.ylim()[1], f'k={st.session_state.optimal_k}', 
                            rotation=0, ha='right', va='bottom')
                    st.pyplot(fig_silhouette)
                    plt.close()
                
                # Display AI analysis
                st.write("**AI Analysis**")
                st.write(st.session_state.clustering_analysis)
                
                # Slider for selecting number of clusters
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    selected_k = st.slider("Počet clusterů (k)", 
                                         min_value=2, 
                                         max_value=10,
                                         value=st.session_state.optimal_k)
                    if selected_k != st.session_state.optimal_k:
                        st.session_state.optimal_k = selected_k
                        
                        # Reset cluster names when changing number of clusters
                        if 'cluster_names' in st.session_state:
                            del st.session_state.cluster_names
                        if 'final_named_clusters' in st.session_state:
                            del st.session_state.final_named_clusters
                        
                        # Perform clustering with new k
                        cluster_labels = perform_kmeans_clustering(
                            st.session_state.normalized_kmeans_data,
                            selected_k
                        )
                        
                        # Update clustered datasets
                        st.session_state.clustered_data, st.session_state.final_clustered_data = create_clustered_datasets(
                            st.session_state.normalized_kmeans_data,
                            cluster_labels,
                            st.session_state.combined_raw_data,
                            st.session_state.original_preference_data,
                            st.session_state.original_behavioral_data
                        )
                        
                        # Calculate new validation metrics
                        validation_metrics = calculate_validation_metrics(
                            st.session_state.clustered_data,
                            st.session_state.clustered_data['cluster']
                        )
                        st.session_state.validation_metrics = validation_metrics
                        
                        # Get new AI evaluation
                        ai_evaluation = evaluate_cluster_validation(
                            validation_metrics['distance_matrix'],
                            validation_metrics['features'],
                            len(set(st.session_state.clustered_data['cluster']))
                        )
                        st.session_state.ai_validation = ai_evaluation
                        
                        st.rerun()

    # Step 2: K-means Clustering
    if "clustering_analysis" in st.session_state:
        with st.expander("2️⃣ K-means Clustering", expanded=True):
            st.markdown('<p class="big-header">K-means Clustering</p>', unsafe_allow_html=True)
            
            if "optimal_k" not in st.session_state:
                st.warning("⚠️ Please determine the optimal number of clusters first.")
                st.info("Complete the analysis in the previous section.")
            else:
                # Perform clustering
                cluster_labels = perform_kmeans_clustering(
                    st.session_state.normalized_kmeans_data,
                    st.session_state.optimal_k
                )
                
                # Create clustered datasets
                st.session_state.clustered_data, st.session_state.final_clustered_data = create_clustered_datasets(
                    st.session_state.normalized_kmeans_data,
                    cluster_labels,
                    st.session_state.combined_raw_data,
                    st.session_state.original_preference_data,
                    st.session_state.original_behavioral_data
                )
                
                st.success(f"✅ Data successfully divided into {st.session_state.optimal_k} clusters")
                
                # Store the toggle states in session state
                if 'show_clusters_toggle' not in st.session_state:
                    st.session_state.show_clusters_toggle = False
                if 'show_final_clusters_toggle' not in st.session_state:
                    st.session_state.show_final_clusters_toggle = False
                
                # Toggle for showing normalized clustered data
                st.write("**Normalized Data with Clusters:**")
                st.session_state.show_clusters_toggle = st.toggle(
                    "Show normalized data with assigned clusters",
                    value=st.session_state.show_clusters_toggle
                )
                
                if st.session_state.show_clusters_toggle:
                    # Reorder columns for normalized data
                    cols = ['customer_id', 'cluster'] + [
                        col for col in st.session_state.clustered_data.columns 
                        if col not in ['customer_id', 'cluster']
                    ]
                    
                    st.dataframe(
                        st.session_state.clustered_data[cols],
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                
                # Toggle for showing final denormalized data
                st.write("**Original Data with Clusters:**")
                st.session_state.show_final_clusters_toggle = st.toggle(
                    "Show original data with assigned clusters",
                    value=st.session_state.show_final_clusters_toggle
                )
                
                if st.session_state.show_final_clusters_toggle:
                    # Reorder columns for final data
                    cols = ['customer_id', 'cluster'] + [col for col in st.session_state.final_clustered_data.columns 
                                                       if col not in ['customer_id', 'cluster']]
                    st.dataframe(
                        st.session_state.final_clustered_data[cols],
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )

        # Step 3: Cluster Validation
        with st.expander("3️⃣ Cluster Validation", expanded=True):
            if 'clustered_data' not in st.session_state:
                st.warning("⚠️ Please perform clustering in the section above first.")
            else:
                # Display validation metrics and AI evaluation from session state
                if 'validation_metrics' in st.session_state and 'ai_validation' in st.session_state:
                    plt.rcParams['figure.figsize'] = [12, 1]
                    fig = plot_validation_metrics(
                        st.session_state.validation_metrics['distance_matrix'],
                        st.session_state.validation_metrics['features'],
                        st.session_state.clustered_data['cluster'],
                        st.session_state.validation_metrics['feature_names']
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("### 🤖 AI Segmentation Analysis")
                    if st.session_state.ai_validation['analysis'] != "Failed to get AI analysis. Please try again later.":
                        st.markdown(st.session_state.ai_validation['analysis'])
                    else:
                        st.error(st.session_state.ai_validation['analysis'])
                else:
                    st.info("Please adjust the number of clusters to perform validation.")

        # Step 4: Name Clusters
        with st.expander("4️⃣ Cluster Naming", expanded=True):
            if 'clustered_data' not in st.session_state:
                st.warning("⚠️ Please perform clustering in the section above first.")
            else:
                st.markdown('<p class="big-header">Name Clusters</p>', unsafe_allow_html=True)
                
                if 'cluster_names' not in st.session_state:
                    if st.button("Get AI Suggestions for Cluster Names", type="primary"):
                        with st.spinner("Getting AI suggestions for cluster names..."):
                            cluster_names = get_cluster_names_from_ai(st.session_state.final_clustered_data)
                            if cluster_names:
                                st.session_state.cluster_names = cluster_names
                            else:
                                st.error("Failed to get AI suggestions. Please try again.")
                
                if 'cluster_names' in st.session_state:
                    st.write("Edit cluster names and explanations below:")
                    
                    edited_names = {}
                    edited_explanations = {}
                    
                    for cluster in sorted(st.session_state.final_clustered_data['cluster'].unique()):
                        info = st.session_state.cluster_names.get(cluster, '')
                        if info:
                            lines = info.split('\n')
                            default_name = lines[0]
                            default_explanation = '\n'.join(lines[1:])
                        else:
                            default_name = f"Cluster {cluster}"
                            default_explanation = ""
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            edited_names[cluster] = st.text_input(
                                f"Name for Cluster {cluster}",
                                value=default_name,
                                key=f"name_{cluster}"
                            )
                        with col2:
                            edited_explanations[cluster] = st.text_area(
                                f"Explanation for Cluster {cluster}",
                                value=default_explanation,
                                key=f"explanation_{cluster}",
                                height=100
                            )
                    
                    if st.button("Save Cluster Names", type="primary"):
                        new_cluster_info = {
                            cluster: f"{name}\n{edited_explanations[cluster]}"
                            for cluster, name in edited_names.items()
                        }
                        
                        st.session_state.final_named_clusters = create_named_clustered_dataset(
                            st.session_state.final_clustered_data,
                            new_cluster_info
                        )
                        
                        st.success("✅ Cluster names saved successfully!")
                
                # Přesunuto mimo podmínku tlačítka Save
                if 'final_named_clusters' in st.session_state:
                    # Toggle for showing final named clusters
                    if 'show_named_clusters_toggle' not in st.session_state:
                        st.session_state.show_named_clusters_toggle = False
                    
                    st.write("**Final Dataset with Named Clusters:**")
                    show_named_clusters = st.toggle(
                        "Show final dataset with cluster names",
                        value=st.session_state.show_named_clusters_toggle,
                        key="show_named_clusters"
                    )
                    
                    if show_named_clusters:
                        try:
                            # Reorder columns for final named data
                            cols = ['customer_id', 'cluster', 'cluster_name'] + [
                                col for col in st.session_state.final_named_clusters.columns 
                                if col not in ['customer_id', 'cluster', 'cluster_name', 'cluster_explanation']
                            ] + ['cluster_explanation']
                            
                            st.dataframe(
                                st.session_state.final_named_clusters[cols],
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                        except Exception as e:
                            st.error(f"Chyba při zobrazování dat: {str(e)}")
                            st.write("Dostupné sloupce:")
                            st.write(st.session_state.final_named_clusters.columns.tolist())

        # Final success message
        if 'final_named_clusters' in st.session_state:
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; padding: 2rem; margin: 2rem 0; background-color: #1a1a1a; border-radius: 10px; border: 1px solid #333;'>
                    <h2 style='color: #00cc00; margin-bottom: 1rem;'>✅ Clustering Analysis Complete</h2>
                    <p style='color: #ffffff; font-size: 1.2rem; margin-bottom: 1.5rem;'>
                        Customer segmentation is now complete with named clusters. You can proceed to inventory and customer selection.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation button
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Continue to Inventory & Customer Selection ➡️", type="primary", use_container_width=True):
                    st.session_state['go_to_inventory'] = True
                    st.rerun()

elif menu_selection == "Inventory & Customer Selection":
    st.title("📦 Inventory & Customer Selection")
    
    # Filter and Sort section
    with st.expander("🔍 Filter and Sort Products", expanded=True):
        st.markdown('<p class="big-header">Filter and Sort Products</p>', unsafe_allow_html=True)
        
        # Multiple filter options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Filter Options:**")
            show_high_stock = st.checkbox("Show High Stock Products", value=False)
            show_high_margin = st.checkbox("Show High Margin Products", value=False)
            show_high_price = st.checkbox("Show High Price Products", value=False)
            
        with col2:
            num_products = st.slider(
                "Number of products to display:",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="num_products"
            )
        
        # Apply filters
        df = st.session_state.original_inventory_data.copy()
        filtered_products = set()
        
        if show_high_stock:
            high_stock = set(df.nlargest(num_products, "stock_quantity")["product_id"])
            filtered_products.update(high_stock)
            
        if show_high_margin:
            high_margin = set(df.nlargest(num_products, "profit_margin")["product_id"])
            filtered_products.update(high_margin)
            
        if show_high_price:
            high_price = set(df.nlargest(num_products, "retail_price")["product_id"])
            filtered_products.update(high_price)
            
        # If no filters selected, show all products
        if not filtered_products:
            filtered_df = df.head(num_products)
        else:
            filtered_df = df[df["product_id"].isin(filtered_products)]
        
        # Display filtered data
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "product_id": "ID",
                "product_name": "Product Name",
                "category": "Category",
                "brand": "Brand",
                "stock_quantity": st.column_config.NumberColumn("Stock", format="%d"),
                "retail_price": st.column_config.NumberColumn("Retail Price", format="$%.2f"),
                "cost_price": st.column_config.NumberColumn("Cost Price", format="$%.2f"),
                "profit_margin": st.column_config.NumberColumn("Profit Margin", format="%.1f%%")
            }
        )
    
    # Product Selection section
    with st.expander("🎯 Select Product for Promotion", expanded=True):
        st.markdown('<p class="big-header">Select Product for Promotion</p>', unsafe_allow_html=True)
        
        # Let user select a product
        selected_product = st.selectbox(
            "Select a product to promote:",
            options=[""] + [
                f"{row['product_name']} | {row['category']} | ${row['retail_price']:.2f} | Margin: {row['profit_margin']:.1f}% | Stock: {row['stock_quantity']}"
                for _, row in filtered_df.iterrows()
            ],
            key="selected_product"
        )
        
        if selected_product:
            # Extract product name from the selection
            selected_product_name = selected_product.split(" | ")[0]
            # Store selected product info in session state
            selected_product_info = filtered_df[filtered_df["product_name"] == selected_product_name].iloc[0]
            
            # Display selected product details
            col1, col2 = st.columns(2)
            with col1:
                st.info("Product Details")
                st.write(f"**Name:** {selected_product_info['product_name']}")
                st.write(f"**Category:** {selected_product_info['category']}")
                st.write(f"**Brand:** {selected_product_info['brand']}")
            
            with col2:
                st.info("Business Metrics")
                st.write(f"**Stock:** {selected_product_info['stock_quantity']} units")
                st.write(f"**Retail Price:** ${selected_product_info['retail_price']:.2f}")
                st.write(f"**Cost Price:** ${selected_product_info['cost_price']:.2f}")
                st.write(f"**Profit Margin:** {selected_product_info['profit_margin']:.1f}%")
            
            # Discount Selection
            st.markdown("### 💰 Promotion Discount")
            col1, col2 = st.columns(2)
            
            with col1:
                discount_percent = st.number_input(
                    "Select discount percentage:",
                    min_value=0,
                    max_value=90,
                    value=20,
                    step=1,
                    key="discount_percent"
                )
            
            with col2:
                original_price = selected_product_info['retail_price']
                discounted_price = original_price * (1 - discount_percent/100)
                original_profit = original_price - selected_product_info['cost_price']
                new_profit = discounted_price - selected_product_info['cost_price']
                original_margin = (original_profit / original_price) * 100
                new_margin = (new_profit / discounted_price) * 100
                
                st.info("Discount Analysis")
                st.write(f"**Original Price:** ${original_price:.2f}")
                st.write(f"**Discounted Price:** ${discounted_price:.2f}")
                st.write(f"**Original Profit:** ${original_profit:.2f} ({original_margin:.1f}%)")
                st.write(f"**New Profit:** ${new_profit:.2f} ({new_margin:.1f}%)")
            
            # Store promotion details in session state
            promotion_details = {
                "product_id": selected_product_info['product_id'],
                "product_name": selected_product_info['product_name'],
                "category": selected_product_info['category'],
                "brand": selected_product_info['brand'],
                "original_price": original_price,
                "discount_percent": discount_percent,
                "discounted_price": discounted_price,
                "original_profit": original_profit,
                "new_profit": new_profit,
                "original_margin": original_margin,
                "new_margin": new_margin,
                "stock_quantity": selected_product_info['stock_quantity']
            }
            st.session_state.promotion_details = promotion_details
            
            # Customer Cluster Selection
            if 'final_named_clusters' not in st.session_state:
                st.warning("⚠️ Customer segmentation data not available. Please complete the clustering step first.")
                st.stop()
            
            # Load cluster data
            cluster_data = st.session_state.final_named_clusters.copy()
            
            # Check if we already have segment analysis for this product and discount
            if ('selected_segment' not in st.session_state or 
                st.session_state.get('last_analyzed_product') != selected_product or 
                st.session_state.get('last_analyzed_discount') != discount_percent):
                
                try:
                    # Select best segment using the new function
                    st.session_state.selected_segment = select_best_segment(
                        cluster_data,
                        selected_product_info,
                        discount_percent
                    )
                    
                    st.session_state.last_analyzed_product = selected_product
                    st.session_state.last_analyzed_discount = discount_percent
                    
                except Exception as e:
                    st.error(f"Error analyzing customer segments: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"Exception type: {type(e)}")
                    st.write(f"Exception args: {e.args}")
                    st.stop()
            
            # Display AI selection results
            st.markdown("#### 🎯 Selected Customer Cluster")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"""
                **{st.session_state.selected_segment['cluster_name']}**
                Match: {st.session_state.selected_segment['match_score']}%

                Why this cluster:
                {st.session_state.selected_segment['cluster_explanation']}
                """)
            
            with col2:
                st.info("Cluster Statistics:")
                stats = st.session_state.selected_segment['stats']
                st.write(f"""
                - Average Spent: ${stats['avg_spent']:.2f}
                - Average Order Value: ${stats['avg_order']:.2f}
                - Most Common Brand: {stats['top_brand']}
                - Most Common Category: {stats['top_category']}
                """)
            
            try:
                # Customer selection from the chosen segment
                st.markdown("#### 👤 Customer Selection")
                
                # Select best customers using the new function
                top_customers = select_best_customers(
                    cluster_data,
                    st.session_state.selected_segment['cluster_name'],
                    selected_product_info,
                    discount_percent
                )
                
                # Create dropdown for customer selection
                selected_customer_id = st.selectbox(
                    "Select a customer from top 5 best matches:",
                    options=[f"ID {c['customer_id']} - Match {c['score']}%" for c in top_customers],
                    key="selected_customer_dropdown"
                )
                
                # Get selected customer details
                best_customer = next(c for c in top_customers if str(c['customer_id']) == selected_customer_id.split()[1])
                
                # Display selected customer
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"""
                    🏆 Best Match in Segment
                    Customer ID: {best_customer['customer_id']} | Total Match: {best_customer['score']}%
                    
                    **Why this customer?**
                    ✓ Preferred Category (+30 points)
                    ✓ Above Average Spending (+10 points)
                    ✓ Active Customer (+10 points)
                    """)
                
                with col2:
                    st.info("Customer Profile:")
                    st.write(f"""
                    - Total Spent: ${best_customer['data']['total_spent']:.2f}
                    - Average Order Value: ${best_customer['data']['avg_order_value']:.2f}
                    - Favorite Brand: {best_customer['data']['top_brand']}
                    - Favorite Category: {best_customer['data']['top_category']}
                    - Last Purchase: {best_customer['data']['last_purchase_days_ago']} days ago
                    """)
                
                # Store complete promotion information in session state
                st.session_state.promotion_campaign = {
                    'product': st.session_state.promotion_details,
                    'target_segment': st.session_state.selected_segment,
                    'selected_customer': {
                        'customer_id': best_customer['customer_id'],
                        'match_score': best_customer['score'],
                        'profile': {
                            'total_spent': best_customer['data']['total_spent'],
                            'avg_order_value': best_customer['data']['avg_order_value'],
                            'top_brand': best_customer['data']['top_brand'],
                            'top_category': best_customer['data']['top_category'],
                            'last_purchase_days_ago': best_customer['data']['last_purchase_days_ago'],
                            'discount_sensitivity': best_customer['data']['discount_sensitivity']
                        },
                        'selection_reasons': best_customer['reasons'],
                        'scores': best_customer['scores_breakdown']
                    }
                }

            except Exception as e:
                st.error(f"Error selecting customer: {str(e)}")
                st.write("Debug info:")
                st.write(f"Exception type: {type(e)}")
                st.write(f"Exception args: {e.args}")

            # Navigation section
            if 'promotion_campaign' in st.session_state:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    if st.button("Continue to Email Design ➡️", type="primary", use_container_width=True):
                        st.session_state['go_to_email'] = True
                        st.rerun()
    
elif menu_selection == "Email Design":
    st.title("✉️ Email Campaign Design")
    
    # AI Email Generation section
    with st.expander("✨ AI Email Generation", expanded=True):
        if 'promotion_campaign' not in st.session_state:
            st.warning("⚠️ Nejdříve prosím dokončete výběr produktu a zákazníka.")
            st.stop()
        
        # Get data from session state
        campaign = st.session_state.promotion_campaign
        product = campaign['product']
        customer = campaign['selected_customer']
        segment = campaign['target_segment']
        
        # Show input data for AI
        st.markdown("### 📋 Informace pro generování emailu")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Informace o produktu:**")
            st.write(f"- Název: {product['product_name']}")
            st.write(f"- Značka: {product['brand']}")
            st.write(f"- Kategorie: {product['category']}")
            st.write(f"- Původní cena: ${product['original_price']:.2f}")
            st.write(f"- Sleva: {product['discount_percent']}%")
            st.write(f"- Konečná cena: ${product['discounted_price']:.2f}")
        
        with col2:
            st.write("**Informace o zákazníkovi:**")
            st.write(f"- Segment: {segment['cluster_name']}")
            st.write(f"- Oblíbená značka: {customer['profile']['top_brand']}")
            st.write(f"- Oblíbená kategorie: {customer['profile']['top_category']}")
            st.write(f"- Průměrná hodnota objednávky: ${customer['profile']['avg_order_value']:.2f}")
            st.write(f"- Poslední nákup: před {customer['profile']['last_purchase_days_ago']} dny")
        
        st.markdown("---")

        # AI Prompt for email generation
        prompt = f"""Create a stunning, creative promotional email that focuses on the product and offer. Make it visually appealing and persuasive.

Use these details to create the email:

Product Details:
- Name: {product['product_name']}
- Brand: {product['brand']}
- Category: {product['category']}
- Original Price: ${product['original_price']:.2f}
- Discount: {product['discount_percent']}%
- Final Price: ${product['discounted_price']:.2f}

Target Audience Preferences:
- Preferred Brand: {customer['profile']['top_brand']}
- Preferred Category: {customer['profile']['top_category']}
- Average Purchase Value: ${customer['profile']['avg_order_value']:.2f}

Requirements:
1. Create a modern, visually stunning email design using HTML and inline CSS
2. Use gradients, shadows, and modern typography
3. Make it mobile-responsive
4. Include a compelling subject line that creates urgency
5. Highlight the exclusive discount
6. Focus on product benefits and value
7. Include a strong call-to-action button
8. Add a professional footer
9. Create a sense of urgency (4-day limited offer)
10. DO NOT include any personal greetings or segment information

Format the response as:
SUBJECT: [Write a compelling subject line focusing on the offer]
BODY: [Complete HTML email with inline styles]"""

        # Button to generate email
        if st.button("🎨 Vygenerovat kreativní email", type="primary"):
            with st.spinner("🤖 AI vytváří váš personalizovaný email..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
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
                
                # Očistit HTML kód
                body = body.strip('`').strip()
                if body.startswith('```html'):
                    body = body[7:].strip()
                if body.endswith('```'):
                    body = body[:-3].strip()
                
                st.session_state.email_content = {
                    'subject': subject,
                    'body': body
                }
                st.rerun()
        
        # Display preview if email is generated
        if 'email_content' in st.session_state:
            st.markdown("### 👀 Náhled emailu")
            st.markdown("""
                <style>
                    .email-container {
                        max-width: 800px;
                        margin: 0 auto;
                        background-color: #1a1a1a;
                        padding: 20px;
                        border-radius: 10px;
                    }
                    .email-subject {
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 0 0 20px 0;
                        font-weight: bold;
                        color: black;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                        text-align: center;
                    }
                    .email-preview {
                        background-color: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 0;
                        color: black;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        font-family: Arial, sans-serif;
                    }
                    .email-preview img {
                        max-width: 100%;
                        height: auto;
                        border-radius: 8px;
                        margin: 10px 0;
                    }
                    .email-preview a {
                        text-decoration: none;
                        color: inherit;
                    }
                    .email-preview h1 {
                        font-size: 32px;
                        margin: 20px 0;
                        font-weight: 300;
                        color: #1a1a1a;
                    }
                    .email-preview h2 {
                        font-size: 24px;
                        margin: 15px 0;
                        color: #1a1a1a;
                    }
                    .email-preview p {
                        margin: 10px 0;
                        line-height: 1.5;
                    }
                    .email-preview div {
                        margin: 10px 0;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="email-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="email-subject">Předmět: {st.session_state.email_content["subject"]}</div>', unsafe_allow_html=True)
            st.components.v1.html(st.session_state.email_content["body"], height=800, scrolling=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save button
            if st.button("💾 Uložit email", type="primary"):
                st.session_state.promotion_campaign['email'] = {
                    'subject': st.session_state.email_content["subject"],
                    'body': st.session_state.email_content["body"],
                    'generated_at': pd.Timestamp.now().isoformat()
                }
                st.success("✅ Email byl úspěšně uložen!")
