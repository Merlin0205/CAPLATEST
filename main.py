import streamlit as st
import matplotlib.pyplot as plt
from secure_config import verify_password, get_api_key

# Page settings must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Customer Data Analysis")

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
    create_named_clustered_dataset
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

# Handle navigation to Inventory Selection
if 'go_to_inventory' in st.session_state and st.session_state['go_to_inventory']:
    st.session_state['go_to_inventory'] = False
    st.session_state.menu_selection = "Inventory Selection"

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
    ["Instructions", "Data", "Clustering", "Inventory Selection", "Email Design"],
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
                    cols = ['customer_id', 'cluster'] + [col for col in st.session_state.clustered_data.columns 
                                                       if col not in ['customer_id', 'cluster']]
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
                        Customer segmentation is now complete with named clusters. You can proceed to inventory selection.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation button
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Continue to Inventory Selection ➡️", type="primary", use_container_width=True):
                    st.session_state['go_to_inventory'] = True
                    st.rerun()

elif menu_selection == "Inventory Selection":
    st.title("📦 Inventory Selection")
    st.success("This feature is coming soon!")
    
elif menu_selection == "Email Design":
    st.title("✉️ Email Campaign Design")
    st.success("This feature is coming soon!")
